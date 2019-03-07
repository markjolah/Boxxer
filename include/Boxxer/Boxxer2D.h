/**
* @file Boxxer2D.h
* @author Mark J. Olah (mjo\@cs.unm DOT edu)
* @date 2014-2019
* @brief The class declaration for Boxxer2D.
*/
#ifndef BOXXER_BOXXER2D_H
#define BOXXER_BOXXER2D_H

#include <armadillo>
#include "Boxxer/Hypercube/Hypercube.h"

namespace boxxer {

/**
 * @class Boxxer2D
 * 
 * In this class we make the assumption that images are stored in column-major format
 * and that x=rows, y=cols, t=slices.  This relationship is important in the choice of imsize and sigma
 * parameters.  
 * 
 * imsize = [nrows, ncols, nframes];
 * sigma = [ sigma_rows (X scale=1), sigma_rows (X scale=2); 
 *           sigma_cols(Y scale=1), sigma_cols (Y scale=2)]
 *
 * This is contrary to normal image coordinates in matlab, but for this low level it is easier to think
 * about x as the first index into an image and understand that the meaning for "X" and "Y" will be reversed
 * from the matlab interpretation, but only internally within the Boxxer_IFace MexIFace class.
 * 
 */
template<class FloatT=float, class IdxT=unit32_t>
class Boxxer2D
{
public:
    using IVecT = arma::Col<IdxT>;
    using IMatT = arma::Mat<IdxT>;
    using VecT = arma::Col<FloatT>;
    using MatT = arma::Mat<FloatT>;
    using ImageT = arma::Mat<FloatT>;
    using ImageStackT = arma::Cube<FloatT>;
    using ScaledImageT = arma::Cube<FloatT>;
    using ScaledImageStackT = hypercube::Hypercube<FloatT>;
 
    static const FloatT DefaultSigmaRatio;
    static const IdxT dim;
    
    IdxT nScales;
    IVecT imsize; // [nrows x ncols] size of an individual frame
    MatT sigma; // size: [2 x nScales] row1=sigmaX (rows), row2=sigmaY (cols)
    FloatT sigma_ratio;
    Boxxer2D(const IVecT &imsize, const MatT &sigma);

    void setDoGSigmaRatio(FloatT sigma_ratio);

    void filterScaledLoG(const ImageStackT &im, ScaledImageStackT &fim) const;
    void filterScaledDoG(const ImageStackT &im, ScaledImageStackT &fim) const;
    IdxT scaleSpaceLoGMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals, IdxT neighborhood_size, IdxT scale_neighborhood_size) const;
    IdxT scaleSpaceDoGMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals, IdxT neighborhood_size, IdxT scale_neighborhood_size) const;

    ImageT make_image() const;
    ImageStackT make_image_stack(IdxT nT) const;
    ScaledImageT make_scaled_image() const;
    ScaledImageStackT make_scaled_image_stack(IdxT nT) const;

    /* Static Methods */
    static void filterLoG(const ImageStackT &im, ImageStackT &fim, const VecT &sigma);
    static void filterDoG(const ImageStackT &im, ImageStackT &fim, const VecT &sigma, FloatT sigma_ratio);
    static void filterGauss(const ImageStackT &im, ImageStackT &fim, const VecT &sigma);
    static void checkMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals);
    static IdxT enumerateImageMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals, IdxT neighborhood_size);

private:
    IdxT scaleSpaceFrameMaximaRefine(const ScaledImageT &im, IMatT &maxima, VecT &max_vals, IdxT scale_neighborhood_size) const;
    IdxT scaleSpaceFrameMaxima(const ScaledImageT &im, IMatT &maxima, VecT &max_vals,
                                   IdxT neighborhood_size, IdxT scale_neighborhood_size) const;
    static IdxT combine_maxima(const arma::field<IMatT> &frame_maxima, const arma::field<VecT> &frame_max_vals,
                       IMatT &maxima, VecT &max_vals);
    static void computeDoGSigmas(const MatT &sigma, FloatT sigma_ratio, MatT &gauss_sigmaE, MatT &gauss_sigmaI);
};


template<class FloatT, class IdxT>
inline
typename Boxxer2D<FloatT,IdxT>::ImageT
Boxxer2D<FloatT,IdxT>::make_image() const
{
    return ImageT(imsize(0),imsize(1));
}


template<class FloatT, class IdxT>
inline
typename Boxxer2D<FloatT,IdxT>::ImageStackT
Boxxer2D<FloatT,IdxT>::make_image_stack(IdxT nT) const
{
    return ImageStackT(imsize(0),imsize(1),nT);
}

template<class FloatT, class IdxT>
inline
typename Boxxer2D<FloatT,IdxT>::ScaledImageT
Boxxer2D<FloatT,IdxT>::make_scaled_image() const
{
    return ScaledImageT(imsize(0),imsize(1),nScales);
}

template<class FloatT, class IdxT>
inline
typename Boxxer2D<FloatT,IdxT>::ScaledImageStackT
Boxxer2D<FloatT,IdxT>::make_scaled_image_stack(IdxT nT) const
{
    return ScaledImageStackT(imsize(0),imsize(1),nScales,nT);
}

} /* namespace boxxer */

#endif /* BOXXER_BOXXER2D_H */
