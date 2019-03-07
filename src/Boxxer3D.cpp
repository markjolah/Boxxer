/**
 * @file Boxxer3D.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class method definitions for Boxxer3D.
 */

#include <omp.h>
#include "OMPExceptionCatcher/OMPExceptionCatcher.h"
#include "Boxxer/BoxxerError.h"
#include "Boxxer/GaussFilter.h"
#include "Boxxer/Maxima.h"
#include "Boxxer/Boxxer3D.h"

namespace boxxer {

/* Static member variables */
template<class FloatT, class IdxT>
const IdxT Boxxer3D<FloatT,IdxT>::dim = 3;
template<class FloatT, class IdxT>
const FloatT Boxxer3D<FloatT,IdxT>::DefaultSigmaRatio = 1.1;

template<class FloatT, class IdxT>
Boxxer3D<FloatT,IdxT>::Boxxer3D(const IVecT &imsize, const MatT &_sigma)
    : nScales(_sigma.n_cols),imsize(imsize), sigma(_sigma), sigma_ratio(DefaultSigmaRatio)
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
void Boxxer3D<FloatT,IdxT>::setDoGSigmaRatio(FloatT _sigma_ratio)
{
    if(_sigma_ratio<=1) {
        std::ostringstream msg;
        msg<<"Got bad sigma ratio: "<<_sigma_ratio;
        throw ParameterShapeError(msg.str());
    }
    sigma_ratio=_sigma_ratio;
}

template<class FloatT, class IdxT>
void Boxxer3D<FloatT,IdxT>::filterScaledLoG(const ImageT &im, ScaledImageT &fim)
{
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel for
    for(IdxT s=0; s<nScales; s++) {
        catcher.run([&]{
            LoGFilter3D<FloatT,IdxT> scale_filter(imsize,sigma.col(s));
            scale_filter.filter(im,fim.slice(s));
        });
    }
    catcher.rethrow(); //Rethrow any caught exceptions
}

template<class FloatT, class IdxT>
void Boxxer3D<FloatT,IdxT>::filterScaledDoG(const ImageT &im, ScaledImageT &fim)
{
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel for
    for(IdxT s=0; s<nScales; s++)
        catcher.run([&]{
            DoGFilter3D<FloatT,IdxT> scale_filter(imsize,sigma.col(s),sigma_ratio);
            scale_filter.filter(im,fim.slice(s));
        });
    catcher.rethrow(); //Rethrow any caught exceptions
}

/**
 * 
 * Get the maxima over all scales and all frames.  Scale and maxfind on each frame individually to
 * cut down on memory size (otherwise it would be easier to decouple the filtering and maxfinding.
 */
template<class FloatT, class IdxT>
IdxT Boxxer3D<FloatT,IdxT>::scaleSpaceLoGMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals,
                                                IdxT neighborhood_size, IdxT scale_neighborhood_size)
{
    IdxT nT=static_cast<IdxT>(im.n_slices);
    arma::field<IMatT> frame_maxima(nT); //These will come back 3xN
    arma::field<VecT> frame_max_vals(nT);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        auto sim = make_scaled_image();
        std::vector<LoGFilter3D<FloatT,IdxT>> scale_filters;
        for(IdxT s=0; s<nScales; s++) scale_filters.push_back(LoGFilter3D<FloatT,IdxT>(imsize,sigma.col(s)));
        #pragma omp for
        for(IdxT n=0; n<nT; n++)
            catcher.run([&]{
                for(IdxT s=0; s<nScales; s++) scale_filters[s].filter(im.slice(n),sim.slice(s));
                    scaleSpaceFrameMaxima(sim, frame_maxima(n), frame_max_vals(n), neighborhood_size, scale_neighborhood_size);
            });
    }
    catcher.rethrow(); //Rethrow any caught exceptions
    return combine_maxima(frame_maxima, frame_max_vals, maxima, max_vals);
}

template<class FloatT, class IdxT>
IdxT Boxxer3D<FloatT,IdxT>::scaleSpaceDoGMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals,
                                                IdxT neighborhood_size, IdxT scale_neighborhood_size)
{
    IdxT nT=static_cast<IdxT>(im.n_slices);
    arma::field<IMatT> frame_maxima(nT); //These will come back 3xN
    arma::field<VecT> frame_max_vals(nT);
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        auto sim = make_scaled_image();
        std::vector<DoGFilter3D<FloatT,IdxT>> scale_filters;
        for(IdxT s=0; s<nScales; s++) scale_filters.push_back(DoGFilter3D<FloatT,IdxT>(imsize,sigma.col(s),sigma_ratio));
        #pragma omp for
        for(IdxT n=0; n<nT; n++)
            catcher.run([&]{
                for(IdxT s=0; s<nScales; s++) scale_filters[s].filter(im.slice(n),sim.slice(s));
                    scaleSpaceFrameMaxima(sim, frame_maxima(n), frame_max_vals(n), neighborhood_size, scale_neighborhood_size);
            });
    }
    catcher.rethrow(); //Rethrow any caught exceptions
    return combine_maxima(frame_maxima, frame_max_vals, maxima, max_vals);
}

/**
 * Get the scale maxima for a single frame
 */
template<class FloatT, class IdxT>
IdxT Boxxer3D<FloatT,IdxT>::scaleSpaceFrameMaxima(const ScaledImageT &sim, IMatT &maxima, VecT &max_vals,
                                                  IdxT neighborhood_size, IdxT scale_neighborhood_size) const
{
    arma::field<IMatT> scale_maxima(nScales);
    arma::field<VecT> scale_max_vals(nScales);
    Maxima3D<FloatT,IdxT> maxima3D(imsize, neighborhood_size);
    for(IdxT s=0; s<nScales; s++)
        maxima3D.find_maxima(sim.slice(s), scale_maxima(s), scale_max_vals(s));
    combine_maxima(scale_maxima, scale_max_vals, maxima, max_vals);
    return scaleSpaceFrameMaximaRefine(sim, maxima, max_vals, scale_neighborhood_size);
}

/**
 * Given a scaled image and scale maxima, refine to remove overlapping scale maxima 
 */
template<class FloatT, class IdxT>
IdxT
Boxxer3D<FloatT,IdxT>::scaleSpaceFrameMaximaRefine(const ScaledImageT &im, IMatT &maxima, VecT &max_vals,
                                                   IdxT scale_neighborhood_size) const
{
    using std::min;
    using std::max;
    IMatT new_maxima(maxima.n_rows, maxima.n_cols);
    VecT new_max_vals(max_vals.n_elem);
    IdxT nMaxima = static_cast<IdxT>(maxima.n_cols);
    IdxT nNewMaxima=0;
    IdxT delta = static_cast<IdxT>((scale_neighborhood_size-1)/2);
    for(IdxT n=0; n<nMaxima; n++) {
        IVecT mx = maxima.col(n);
        double mxv = max_vals(n);
        bool ok=true;
        if ( (mx(0) < delta || mx(0)+delta>=imsize(0)) ||
             (mx(1) < delta || mx(1)+delta>=imsize(1)) ||
             (mx(2) < delta || mx(2)+delta>=imsize(2))) {
            for(IdxT s=0; s<nScales; s++)
                for(IdxT k= (mx(2) <= delta ? 0 : mx(2)-delta); k<imsize(2) && k<=mx(2)+delta; k++)
                    for(IdxT j= (mx(1) <= delta ? 0 : mx(1)-delta); j<imsize(1) && j<=mx(1)+delta; j++)
                        for(IdxT i= (mx(0) <= delta ? 0 : mx(0)-delta); i<imsize(0) && i<=mx(0)+delta; i++) {
                            if( im(i,j,k,s) > mxv) {ok=false; goto done;}
            }
        } else {
            for(IdxT s=0; s<nScales; s++)
                for(IdxT k=mx(2)-delta; k<=mx(2)+delta; k++)
                    for(IdxT j=mx(1)-delta; j<=mx(1)+delta; j++)
                        for(IdxT i=mx(0)-delta; i<=mx(0)+delta; i++) {
                            if( im(i,j,k,s) > mxv) {ok=false; goto done;}
            }
        }
        done:
        if(ok){
            new_maxima.col(nNewMaxima)=mx;
            new_max_vals(nNewMaxima)=mxv;
            nNewMaxima++;
        }
    }
    maxima = new_maxima(arma::span::all, arma::span(0,nNewMaxima-1));
    max_vals = new_max_vals(arma::span(0,nNewMaxima-1));
    return nNewMaxima;
}

/* Static Methods */


template<class FloatT, class IdxT>
void Boxxer3D<FloatT,IdxT>::filterLoG(const ImageStackT &im, ImageStackT &fim, const VecT &sigma)
{
    IdxT nT=static_cast<IdxT>(fim.n_slices);
    IVecT imsize = {static_cast<IdxT>(im.sX), static_cast<IdxT>(im.sY), static_cast<IdxT>(im.sZ)};
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        LoGFilter3D<FloatT,IdxT> filter(imsize,sigma);
        #pragma omp for
        for(IdxT n=0; n<nT; n++)
            catcher.run([&]{
                filter.filter(im.slice(n),fim.slice(n));
            });
    }
    catcher.rethrow(); //Rethrow any caught exceptions
}

template<class FloatT, class IdxT>
void Boxxer3D<FloatT,IdxT>::filterDoG(const ImageStackT &im, ImageStackT &fim, const VecT &sigma, FloatT sigma_ratio)
{
    IdxT nT=static_cast<IdxT>(fim.n_slices);
    IVecT imsize = {static_cast<IdxT>(im.sX), static_cast<IdxT>(im.sY), static_cast<IdxT>(im.sZ)};
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        DoGFilter3D<FloatT,IdxT> filter(imsize,sigma,sigma_ratio);
        #pragma omp for
        for(IdxT n=0; n<nT; n++)
            catcher.run([&]{
                filter.filter(im.slice(n),fim.slice(n));
            });
    }
    catcher.rethrow(); //Rethrow any caught exceptions
}

template<class FloatT, class IdxT>
void Boxxer3D<FloatT,IdxT>::filterGauss(const ImageStackT &im, ImageStackT &fim, const VecT &sigma)
{
    IdxT nT=static_cast<IdxT>(fim.n_slices);
    IVecT imsize = {static_cast<IdxT>(im.sX), static_cast<IdxT>(im.sY), static_cast<IdxT>(im.sZ)};
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        GaussFilter3D<FloatT,IdxT> filter(imsize,sigma);
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
IdxT Boxxer3D<FloatT,IdxT>::enumerateImageMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals,
                                                 IdxT neighborhood_size)
{
    IdxT nT=static_cast<IdxT>(im.n_slices);
    arma::field<IMatT> frame_maxima(nT);
    arma::field<VecT> frame_max_vals(nT);
    IVecT imsize = {static_cast<IdxT>(im.sX), static_cast<IdxT>(im.sY), static_cast<IdxT>(im.sZ)};
    omp_exception_catcher::OMPExceptionCatcher catcher;
    #pragma omp parallel
    {
        Maxima3D<FloatT,IdxT> maxima3D(imsize, neighborhood_size);
        #pragma omp for
        for(IdxT n=0; n<nT; n++)
            catcher.run([&]{
                maxima3D.find_maxima(im.slice(n), frame_maxima(n), frame_max_vals(n));
            });
    }
    catcher.rethrow(); //Rethrow any caught exceptions
    return combine_maxima(frame_maxima, frame_max_vals, maxima, max_vals);
}

template<class FloatT, class IdxT>
IdxT Boxxer3D<FloatT,IdxT>::combine_maxima(const arma::field<IMatT> &frame_maxima,
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
        IdxT NFrameMaxima = frame_max_vals(n).n_elem;
        if(NFrameMaxima>0){
            maxima(arma::span(0,nrows-1), arma::span(Nsaved,Nsaved+NFrameMaxima-1))=frame_maxima(n);
            maxima(arma::span(nrows,nrows), arma::span(Nsaved,Nsaved+NFrameMaxima-1)).fill(n);
            max_vals.rows(Nsaved,Nsaved+NFrameMaxima-1)=frame_max_vals(n);
            Nsaved+=NFrameMaxima;
        }
    }
    return Nmaxima;
}

template<class FloatT, class IdxT>
void Boxxer3D<FloatT,IdxT>::checkMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals)
{
    IdxT Nmaxima=static_cast<IdxT>(maxima.n_cols);
    for(IdxT n=0; n<Nmaxima; n++){
        FloatT val=im(maxima(0,n), maxima(1,n), maxima(2,n), maxima(3,n));
        if (val!=max_vals(n)) {
            printf(" (%i,%i,%i,%i):%.9f!= %.9f\n",maxima(0,n), maxima(1,n), maxima(2,n), maxima(3,n), val, max_vals(n));
        }
    }
}

/* Explicit Template Instantiation */
template class Boxxer3D<float,uint32_t>;
template class Boxxer3D<double,uint32_t>;

} /* namespace boxxer */
