/** @file GaussFilter.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
* @date 2014-2019
* @brief The class declarations for Gaussian image filter classes.
 * 
 * These classes are meant to be a per-thread worker class or a direct interface
 * for single threaded processes.  Each object has its own local storage of which is only 1 or
 * 2 frames in size.
 */
#ifndef BOXXER_GAUSSFILTER_H
#define BOXXER_GAUSSFILTER_H

#include <cstdint>
#include <ostream>
#include <armadillo>

namespace boxxer {

/** Base filters */
template<class FloatT=float, class IdxT=uint32_t>
class GaussFIRFilter
{
public:
    using IVecT = arma::Col<IdxT>;
    using VecT = arma::Col<FloatT>;
    using MatT = arma::Mat<FloatT>;
    
    IdxT dim;
    IVecT size; //[nrows, ncols]
    VecT sigma; //sigma to filter at [sigma_rows, sigma_cols]
    IVecT hw; //Half width Full kernel width is 2*hw+1. Recommended 3sigma

    GaussFIRFilter(IdxT dim, const IVecT &size, const VecT &sigma);

    virtual void set_kernel_hw(const IVecT &kernel_half_width)=0;

    static VecT compute_Gauss_FIR_kernel(FloatT sigma, IdxT hw);
    static VecT compute_LoG_FIR_kernel(FloatT sigma, IdxT hw);
protected:
    static const IdxT max_kernel_hw;
    static const FloatT default_sigma_hw_ratio;
};

/**@{*/
/** 2D Filters */
template<class FloatT=float, class IdxT=uint32_t>
class GaussFilter2D : public GaussFIRFilter<FloatT,IdxT>
{
public:
    using IVecT = typename GaussFIRFilter<FloatT,IdxT>::IVecT;
    using VecT = typename GaussFIRFilter<FloatT,IdxT>::VecT;
    using ImageT = arma::Mat<FloatT>;

    GaussFilter2D(const IVecT &size, const VecT &sigma);
    GaussFilter2D(const IVecT &size, const VecT &sigma, const IVecT &kernel_hw);
    void set_kernel_hw(const IVecT &kernel_half_width);
    ImageT make_image() const { return ImageT(this->size(0),this->size(1)); }

    void filter(const ImageT &im, ImageT &out);
    void test_filter(const ImageT &im);

    template<class FloatT_, class IdxT_>
    friend std::ostream& operator<<(std::ostream &out, const GaussFilter2D<FloatT_,IdxT_> &filt);
private:
    ImageT temp_im;
    arma::field<VecT> kernels;
};

template<class FloatT=float, class IdxT=uint32_t>
class DoGFilter2D : public GaussFIRFilter<FloatT,IdxT>
{
public:
    using IVecT = typename GaussFIRFilter<FloatT,IdxT>::IVecT;
    using VecT = typename GaussFIRFilter<FloatT,IdxT>::VecT;
    using ImageT = arma::Mat<FloatT>;

    FloatT sigma_ratio;
    
    DoGFilter2D(const IVecT &size, const VecT &sigma, FloatT sigma_ratio);
    DoGFilter2D(const IVecT &size, const VecT &sigma, FloatT sigma_ratio, const IVecT &kernel_hw);
    void set_kernel_hw(const IVecT &kernel_half_width);
    void set_sigma_ratio(FloatT sigma_ratio);
    ImageT make_image() const { return ImageT(this->size(0),this->size(1)); }

    void filter(const ImageT &im, ImageT &out);
    void test_filter(const ImageT &im);

    template<class FloatT_, class IdxT_>
    friend std::ostream& operator<<(std::ostream &out, const DoGFilter2D<FloatT_,IdxT_> &filt);
private:
    ImageT temp_im0;
    ImageT temp_im1;
    arma::field<VecT> excite_kernels;
    arma::field<VecT> inhibit_kernels;
};

template<class FloatT=float, class IdxT=uint32_t>
class LoGFilter2D : public GaussFIRFilter<FloatT,IdxT>
{
public:
    using IVecT = typename GaussFIRFilter<FloatT,IdxT>::IVecT;
    using VecT = typename GaussFIRFilter<FloatT,IdxT>::VecT;
    using ImageT = arma::Mat<FloatT>;

    LoGFilter2D(const IVecT &size, const VecT &sigma);
    LoGFilter2D(const IVecT &size, const VecT &sigma, const IVecT &kernel_hw);
    void set_kernel_hw(const IVecT &kernel_half_width);
    ImageT make_image() const { return ImageT(this->size(0),this->size(1)); }

    void filter(const ImageT &im, ImageT &out);
    void test_filter(const ImageT &im);

    template<class FloatT_, class IdxT_>
    friend std::ostream& operator<<(std::ostream &out, const LoGFilter2D<FloatT_,IdxT_> &filt);
private:
    ImageT temp_im0;
    ImageT temp_im1;
    arma::field<VecT>  gauss_kernels;
    arma::field<VecT>  LoG_kernels;
};
/**@}*/

/**@{*/
/** 3D Filters */
template<class FloatT=float, class IdxT=uint32_t>
class GaussFilter3D : public GaussFIRFilter<FloatT,IdxT>
{
public:
    using IVecT = typename GaussFIRFilter<FloatT,IdxT>::IVecT;
    using VecT = typename GaussFIRFilter<FloatT,IdxT>::VecT;
    using ImageT = arma::Cube<FloatT>;

    GaussFilter3D(const IVecT &size, const VecT &sigma);
    GaussFilter3D(const IVecT &size, const VecT &sigma, const IVecT &kernel_hw);
    void set_kernel_hw(const IVecT &kernel_half_width);
    ImageT make_image() const { return ImageT(this->size(0),this->size(1),this->size(2)); }

    void filter(const ImageT &im, ImageT &out);
    void test_filter(const ImageT &im);

    template<class FloatT_, class IdxT_>
    friend std::ostream& operator<<(std::ostream &out, const GaussFilter3D<FloatT_,IdxT_> &filt);
private:
    ImageT temp_im0;
    ImageT temp_im1;
    arma::field<VecT> kernels;
};

template<class FloatT=float, class IdxT=uint32_t>
class DoGFilter3D : public GaussFIRFilter<FloatT,IdxT>
{
public:
    using IVecT = typename GaussFIRFilter<FloatT,IdxT>::IVecT;
    using VecT = typename GaussFIRFilter<FloatT,IdxT>::VecT;
    using ImageT = arma::Cube<FloatT>;

    FloatT sigma_ratio;

    DoGFilter3D(const IVecT &size, const VecT &sigma, FloatT sigma_ratio);
    DoGFilter3D(const IVecT &size, const VecT &sigma, FloatT sigma_ratio, const IVecT &kernel_hw);
    void set_kernel_hw(const IVecT &kernel_half_width);
    void set_sigma_ratio(FloatT sigma_ratio);
    ImageT make_image() const { return ImageT(this->size(0),this->size(1),this->size(2)); }

    void filter(const ImageT &im, ImageT &out);
    void test_filter(const ImageT &im);

    template<class FloatT_, class IdxT_>
    friend std::ostream& operator<<(std::ostream &out, const GaussFilter3D<FloatT_,IdxT_> &filt);
private:
    ImageT temp_im0;
    ImageT temp_im1;
    arma::field<VecT> excite_kernels;
    arma::field<VecT> inhibit_kernels;
};

template<class FloatT=float, class IdxT=uint32_t>
class LoGFilter3D : public GaussFIRFilter<FloatT,IdxT>
{
public:
    using IVecT = typename GaussFIRFilter<FloatT,IdxT>::IVecT;
    using VecT = typename GaussFIRFilter<FloatT,IdxT>::VecT;
    using ImageT = arma::Cube<FloatT>;

    LoGFilter3D(const IVecT &size, const VecT &sigma);
    LoGFilter3D(const IVecT &size, const VecT &sigma, const IVecT &kernel_hw);
    void set_kernel_hw(const IVecT &kernel_half_width);
    ImageT make_image() const { return ImageT(this->size(0),this->size(1),this->size(2)); }

    void filter(const ImageT &im, ImageT &out);
    void test_filter(const ImageT &im);

    template<class FloatT_, class IdxT_>
    friend std::ostream& operator<<(std::ostream &out, const LoGFilter3D<FloatT_,IdxT_> &filt);
private:
    ImageT temp_im0, temp_im1;
    arma::field<VecT> gauss_kernels;
    arma::field<VecT> LoG_kernels;
};
/**@}*/

} /* namespace boxxer */

#endif /* BOXXER_GAUSSFILTER_H */
