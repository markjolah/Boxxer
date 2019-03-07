/**
 * @file Boxxer3D.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class declaration for Boxxer3D.
 */
#ifndef BOXXER_BOXXER3D_H
#define BOXXER_BOXXER3D_H

#include <cstdint>
#include <armadillo>
#include "Boxxer/Hypercube/Hypercube.h"

namespace boxxer {

/** A box finding algorithm for 3D hyper-spectral microscopy data.
 *
 * Estimates the center coordinates of Gaussian blobs with anisotropic sigmas.
 *
 * All image data manipulated is stored as column-major FloatT arrays with dimension ordering [L Y X T].
 *
 * The Boxxer3D class makes uses of lower level class which are agnostic about the data source being hyperspectral,
 * they don't care what the coordinate dimensions represent scientifically, but this class is associated with the
 * Matlab Boxxer3D class and so maintains the knowledge that the actual coordinates are [L Y X T].
 */
template<class FloatT=float, class IdxT=uint32_t>
class Boxxer3D
{
public:
    using IVecT = arma::Col<IdxT>;
    using IMatT = arma::Mat<IdxT>;
    using VecT = arma::Col<FloatT>;
    using MatT = arma::Mat<FloatT>;
    using ImageT = arma::Cube<FloatT>;
    using ImageStackT = hypercube::Hypercube<FloatT>;
    using ScaledImageT = hypercube::Hypercube<FloatT>;

    static const FloatT DefaultSigmaRatio;
    static const IdxT dim;

    IdxT nScales;
    IVecT imsize; // Size of each dimension for the column-major data. HSData is [L Y X] this is [row, col, slice]
    MatT sigma; // sized: [2 x nScales].  Rows are [psf_L, psf_y, psf_x] cols are the different scales 
                //CRITICAL: the order of sigma rows must match the order of dimension in imsize.
    FloatT sigma_ratio;
    Boxxer3D(const IVecT &size, const MatT &sigma);
    
    void setDoGSigmaRatio(FloatT sigma_ratio);

    void filterScaledLoG(const ImageT &im, ScaledImageT &fim);
    void filterScaledDoG(const ImageT &im, ScaledImageT &fim);
    IdxT scaleSpaceLoGMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals, IdxT neighborhood_size, IdxT scale_neighborhood_size);
    IdxT scaleSpaceDoGMaxima(const ImageStackT &im, IMatT &maxima, VecT &max_vals, IdxT neighborhood_size, IdxT scale_neighborhood_size);

    ImageT make_image() const { return ImageT(imsize(0),imsize(1),imsize(2)); }
    ImageStackT make_image_stack(IdxT nT) const { return ImageStackT(imsize(0),imsize(1),imsize(2),nT); }
    ScaledImageT make_scaled_image() const { return ScaledImageT(imsize(0),imsize(1),imsize(2),nScales); }

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
    void initialize_log_scale_filters();
    void initialize_dog_scale_filters();
};

} /* namespace boxxer */

#endif /* BOXXER_BOXXER3D_H */
