/**
 * @file Boxxer2D.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The Boxxer2D class definition
 */

#include <omp.h>
#include "OMPExceptionCatcher/OMPExceptionCatcher.h"
#include "Boxxer/BoxxerError.h"
#include "Boxxer/GaussFilter.h"
#include "Boxxer/Maxima.h"
#include "Boxxer/Boxxer2D.h"

namespace boxxer {

/* Static member variables */
template<class FloatT, class IdxT>
const IdxT Boxxer2D<FloatT,IdxT>::dim = 2;
template<class FloatT, class IdxT>
const FloatT Boxxer2D<FloatT,IdxT>::DefaultSigmaRatio = 1.1;


template<class FloatT, class IdxT>
Boxxer2D<FloatT,IdxT>::Boxxer2D(const IVecT &imsize, const MatT &_sigma)
    : nScales(_sigma.n_cols), imsize(imsize), sigma(_sigma), sigma_ratio(DefaultSigmaRatio)
{
    if(nScales<1) throw ParameterValueError("Non-positive number of scales.");
    if(imsize.n_elem!=dim){
        std::ostringstream msg;
        msg<<"Got image size with incorrect number of elements(dim="<<dim<<"): "<<imsize.n_elem;
        throw ParameterShapeError(msg.str());
    }
    if(sigma.n_rows!=dim){
        std::ostringstream msg;
        msg<<"Got sigmas with incorrect number of elements(dim="<<dim<<"): #rows"<<sigma.n_rows;
        throw ParameterShapeError(msg.str());
    }
}

template<class FloatT, class IdxT>
void Boxxer2D<FloatT,IdxT>::setDoGSigmaRatio(FloatT _sigma_ratio)
{
    if(_sigma_ratio<=1) {
        std::ostringstream msg;
        msg<<"Got bad sigma ratio: "<<_sigma_ratio;
        throw ParameterShapeError(msg.str());
    }
    sigma_ratio=_sigma_ratio;
}

template<class FloatT, class IdxT>
void Boxxer2D<FloatT,IdxT>::filterScaledLoG(const ImageStackT &im, ScaledImageStackT &fim) const
{
    IdxT nT=static_cast<IdxT>(fim.n_slices);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        //Each LoGFilter2D object has internal storage and so each thread must have its own copy.
        std::vector<LoGFilter2D<FloatT,IdxT>> filters;
        for(IdxT s=0; s<nScales; s++) filters.push_back(LoGFilter2D<FloatT,IdxT>(imsize,sigma.col(s)));
        #pragma omp for
        for(IdxT n=0; n<nT; n++) for(IdxT s=0; s<nScales; s++)
            catcher.run([&]{
                filters[s].filter(im.slice(n),fim.slice(n).slice(s));
            });
    }
    catcher.rethrow(); //Rethrow any caught exceptions
}

template<class FloatT, class IdxT>
void Boxxer2D<FloatT,IdxT>::filterScaledDoG(const ImageStackT &im, ScaledImageStackT &fim) const
{
    IdxT nT=static_cast<IdxT>(fim.n_slices);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        //Each LoGFilter2D object has internal storage and so each thread must have its own copy.
        std::vector<DoGFilter2D<FloatT,IdxT>> filters;
        for(IdxT s=0; s<nScales; s++)
            filters.push_back(DoGFilter2D<FloatT,IdxT>(imsize,sigma.col(s),sigma_ratio));
        #pragma omp for
        for(IdxT n=0; n<nT; n++) for(IdxT s=0; s<nScales; s++)
            catcher.run([&]{
                filters[s].filter(im.slice(n),fim.slice(n).slice(s));
            });
    }
    catcher.rethrow(); //Rethrow any caught exceptions
}

/**
 * 
 * Get the maxima over all scales and all frames.  Scale and maxfind on each frame individually to
 * cut down on memory size (otherwise it would be easier to decouple the filtering and maxfinding.
 */
template<class FloatT, class IdxT>
IdxT Boxxer2D<FloatT,IdxT>::scaleSpaceLoGMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals,
                                   IdxT neighborhood_size, IdxT scale_neighborhood_size) const
{
    IdxT nT=static_cast<IdxT>(im.n_slices);
    arma::field<IMatT> frame_maxima(nT); //These will come back 3xN
    arma::field<VecT> frame_max_vals(nT);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        std::vector<LoGFilter2D<FloatT,IdxT>> filters;
        for(IdxT s=0; s<nScales; s++) filters.push_back(LoGFilter2D<FloatT,IdxT>(imsize,sigma.col(s)));
        auto sim = make_scaled_image();
        #pragma omp for
        for(IdxT n=0; n<nT; n++) {
            catcher.run([&]{
                for(IdxT s=0; s<nScales; s++) filters[s].filter(im.slice(n),sim.slice(s));
                    scaleSpaceFrameMaxima(sim, frame_maxima(n), frame_max_vals(n), neighborhood_size, scale_neighborhood_size);
            });
        }
    }
    catcher.rethrow(); //Rethrow any caught exceptions
    return combine_maxima(frame_maxima, frame_max_vals, maxima, max_vals);
}

template<class FloatT, class IdxT>
IdxT Boxxer2D<FloatT,IdxT>::scaleSpaceDoGMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals,
                                      IdxT neighborhood_size, IdxT scale_neighborhood_size) const
{
    IdxT nT=static_cast<IdxT>(im.n_slices);
    arma::field<IMatT> frame_maxima(nT); //These will come back 3xN
    arma::field<VecT> frame_max_vals(nT);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        std::vector<DoGFilter2D<FloatT,IdxT>> filters;
        for(IdxT s=0; s<nScales; s++) filters.push_back(DoGFilter2D<FloatT,IdxT>(imsize,sigma.col(s),sigma_ratio));
        auto sim = make_scaled_image();
        #pragma omp for
        for(IdxT n=0; n<nT; n++) {
            catcher.run([&]{
                for(IdxT s=0; s<nScales; s++) filters[s].filter(im.slice(n),sim.slice(s));
                    scaleSpaceFrameMaxima(sim, frame_maxima(n), frame_max_vals(n), neighborhood_size, scale_neighborhood_size);
            });
        }
    }
    catcher.rethrow(); //Rethrow any caught exceptions
    return combine_maxima(frame_maxima, frame_max_vals, maxima, max_vals);
}


/**
 * Get the scale maxima for a single frame
 */
template<class FloatT, class IdxT>
IdxT Boxxer2D<FloatT,IdxT>::scaleSpaceFrameMaxima(const ScaledImageT &sim, IMatT &maxima, VecT &max_vals,
                                   IdxT neighborhood_size, IdxT scale_neighborhood_size) const
{
    arma::field<IMatT> scale_maxima(nScales);
    arma::field<VecT> scale_max_vals(nScales);
    Maxima2D<FloatT,IdxT> maxima2D(imsize, neighborhood_size);
    for(IdxT s=0; s<nScales; s++)
        maxima2D.find_maxima(sim.slice(s), scale_maxima(s), scale_max_vals(s));
    combine_maxima(scale_maxima, scale_max_vals, maxima, max_vals);
    return scaleSpaceFrameMaximaRefine(sim, maxima, max_vals, scale_neighborhood_size);
}

/**
 * Given a scaled image and scale maxima, refine to remove overlapping scale maxima 
 */
template<class FloatT, class IdxT>
IdxT
Boxxer2D<FloatT,IdxT>::scaleSpaceFrameMaximaRefine(const ScaledImageT &im, IMatT &maxima, VecT &max_vals,
                                              IdxT scale_neighborhood_size) const
{
    using std::max;
    using std::min;
    IMatT new_maxima(maxima.n_rows, maxima.n_cols);
    VecT new_max_vals(max_vals.n_elem);
    IdxT nMaxima = static_cast<IdxT>(maxima.n_cols);
    IdxT nNewMaxima=0;
    IdxT delta = static_cast<IdxT>((scale_neighborhood_size-1)/2);
    for(IdxT n=0; n<nMaxima; n++) {
        IVecT mx = maxima.col(n);
        FloatT mxv = max_vals(n);
        if ( (mx(0) < delta || mx(0)+delta>=imsize(0)) ||
             (mx(1) < delta || mx(1)+delta>=imsize(1))) {
            for(IdxT s=0; s<nScales; s++)
                for(IdxT j = (mx(1)<=delta ? 0 : mx(1)-delta); j<imsize(1) && j<=mx(1)+delta; j++)
                    for(IdxT i = (mx(0)<=delta ? 0 : mx(0)-delta); i<imsize(0) && i<=mx(0)+delta; i++)
                        if( im(i,j,s) > mxv)  goto scale_maxima_reject;
        } else {
            for(IdxT s=0; s<nScales; s++)
                for(IdxT j=mx(1)-delta; j<=mx(1)+delta; j++)
                    for(IdxT i=mx(0)-delta; i<=mx(0)+delta; i++)
                        if( im(i,j,s) > mxv) goto scale_maxima_reject;
        }
        new_maxima.col(nNewMaxima) = mx;
        new_max_vals(nNewMaxima) = mxv;
        nNewMaxima++;
scale_maxima_reject: ;//Go here when scale maxima is not valid
    }
    if(nNewMaxima==0) {
        maxima.set_size(maxima.n_rows,0);
        max_vals.reset();
    } else {
        maxima = new_maxima(arma::span::all, arma::span(0,nNewMaxima-1));
        max_vals = new_max_vals(arma::span(0,nNewMaxima-1));
    }
    return nNewMaxima;
}


/* Static Methods */

template<class FloatT, class IdxT>
void Boxxer2D<FloatT,IdxT>::filterLoG(const ImageStackT &im, ImageStackT &fim, const VecT &sigma)
{
    IdxT nT=static_cast<IdxT>(fim.n_slices);
    IVecT imsize={static_cast<IdxT>(im.n_rows),static_cast<IdxT>(im.n_cols)};
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        LoGFilter2D<FloatT,IdxT> filter(imsize,sigma);
        #pragma omp for
        for(IdxT n=0; n<nT; n++)
            catcher.run([&]{
                filter.filter(im.slice(n),fim.slice(n));
            });
    }
    catcher.rethrow(); //Rethrow any caught exceptions
}

template<class FloatT, class IdxT>
void Boxxer2D<FloatT,IdxT>::filterDoG(const ImageStackT &im, ImageStackT &fim, const VecT &sigma, FloatT sigma_ratio)
{
    IdxT nT=static_cast<IdxT>(fim.n_slices);
    IVecT imsize={static_cast<IdxT>(im.n_rows),static_cast<IdxT>(im.n_cols)};
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        DoGFilter2D<FloatT,IdxT> filter(imsize,sigma,sigma_ratio);
        #pragma omp for
        for(IdxT n=0; n<nT; n++)
            catcher.run([&]{
                filter.filter(im.slice(n),fim.slice(n));
            });
    }
    catcher.rethrow(); //Rethrow any caught exceptions
}

template<class FloatT, class IdxT>
void Boxxer2D<FloatT,IdxT>::filterGauss(const ImageStackT &im, ImageStackT &fim, const VecT &sigma)
{
    IdxT nT=static_cast<IdxT>(im.n_slices);
    IVecT imsize={static_cast<IdxT>(im.n_rows),static_cast<IdxT>(im.n_cols)};
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        GaussFilter2D<FloatT,IdxT> filter(imsize,sigma);
        #pragma omp for
        for(IdxT n=0; n<nT; n++)
            catcher.run([&]{
                filter.filter(im.slice(n),fim.slice(n));
            });
    }
    catcher.rethrow(); //Rethrow any caught exceptions
}

/**
 * This finds local maxima over an image stack in parallel.
 */
template<class FloatT, class IdxT>
IdxT Boxxer2D<FloatT,IdxT>::enumerateImageMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals, IdxT neighborhood_size)
{
    IdxT nT=static_cast<IdxT>(im.n_slices);
    arma::field<IMatT> frame_maxima(nT);
    arma::field<VecT> frame_max_vals(nT);
    IVecT imsize={static_cast<IdxT>(im.n_rows),static_cast<IdxT>(im.n_cols)};
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        Maxima2D<FloatT,IdxT> maxima2D(imsize, neighborhood_size);
        #pragma omp for
        for(IdxT n=0; n<nT; n++)
            catcher.run([&]{
                maxima2D.find_maxima(im.slice(n), frame_maxima(n), frame_max_vals(n));
        });
    }
    catcher.rethrow(); //Rethrow any caught exceptions
    return combine_maxima(frame_maxima, frame_max_vals, maxima, max_vals);
}

template<class FloatT, class IdxT>
IdxT Boxxer2D<FloatT,IdxT>::combine_maxima(const arma::field<IMatT> &frame_maxima,
                                     const arma::field<VecT> &frame_max_vals,
                                     IMatT &maxima, VecT &max_vals)
{
    IdxT Nmaxima=0;
    for(IdxT n=0; n<frame_max_vals.n_elem; n++) Nmaxima += frame_max_vals(n).n_elem;
    IdxT nrows = frame_maxima(0).n_rows;
    maxima.resize(nrows+1,Nmaxima);
    max_vals.resize(Nmaxima);
    IdxT Nsaved=0;
    for(IdxT n=0; n<frame_max_vals.n_elem; n++) {
        IdxT NFrameMaxima=frame_max_vals(n).n_elem;
        if(NFrameMaxima>0){
            maxima(arma::span(0,nrows-1), arma::span(Nsaved,Nsaved+NFrameMaxima-1)) = frame_maxima(n);
            maxima(arma::span(nrows,nrows), arma::span(Nsaved,Nsaved+NFrameMaxima-1)).fill(n);
            max_vals.rows(Nsaved,Nsaved+NFrameMaxima-1) = frame_max_vals(n);
            Nsaved += NFrameMaxima;
        }
    }
    return Nmaxima;
}

template<class FloatT, class IdxT>
void Boxxer2D<FloatT,IdxT>::checkMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals)
{
    IdxT Nmaxima=static_cast<IdxT>(maxima.n_cols);
    for(IdxT n=0; n<Nmaxima; n++){
        FloatT val=im(maxima(0,n), maxima(1,n), maxima(2,n));
        if (val!=max_vals(n)) {
            printf(" (%i,%i,%i):%.9f!= %.9f\n",maxima(0,n), maxima(1,n), maxima(2,n), val, max_vals(n));
        }
    }
}

/* Explicit Template Instantiation */
template class Boxxer2D<float,uint32_t>;
template class Boxxer2D<double,uint32_t>;

} /* namespace boxxer */
