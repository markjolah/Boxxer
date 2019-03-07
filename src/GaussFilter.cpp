/** @file GaussFilter.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief Gaussian filter class member function definitions.
 *
 */

#include <limits>
#include <iomanip>

#include "Boxxer/GaussFilter.h"
#include "Boxxer/FilterKernels.h"

namespace boxxer {

/* GaussFIRFilter */

template<class FloatT, class IdxT>
GaussFIRFilter<FloatT,IdxT>::GaussFIRFilter(IdxT dim_, const IVecT &size_, const VecT &sigma_)
    : dim(dim_), size(size_), sigma(sigma_), hw(dim_)
{
    if(dim<1 || dim > 3) {
        std::ostringstream msg;
        msg<<"Got bad dim: "<<dim;
        throw ParameterValueError(msg.str());
    }
    if(size.n_elem != dim) {
        std::ostringstream msg;
        msg<<"Got bad size #elem: "<<size.n_elem<<" dim:"<<dim;
        throw ParameterValueError(msg.str());
    }
    if(!arma::all(size>0)) {
        std::ostringstream msg;
        msg<<"Got bad size: "<<size.t();
        throw ParameterValueError(msg.str());
    }
    if(sigma.n_elem != dim) {
        std::ostringstream msg;
        msg<<"Got bad sigma #elem: "<<sigma.n_elem<<" dim:"<<dim;
        throw ParameterValueError(msg.str());
    }
    if(!arma::all(sigma>0)) {
        std::ostringstream msg;
        msg<<"Got bad sigma: "<<sigma.t();
        throw ParameterValueError(msg.str());
    }
}

template<class FloatT, class IdxT>
typename GaussFIRFilter<FloatT,IdxT>::VecT
GaussFIRFilter<FloatT,IdxT>::compute_Gauss_FIR_kernel(FloatT sigma, IdxT hw)
{
    IdxT W=hw+1;  //full kernel size is hw*2+1, but we only compute the right half + center pixel
    arma::vec kernel(W);
    FloatT exp_norm=-0.5/(sigma*sigma); // -1/(2*sigma^2)
    kernel(0)=1.;
    FloatT sum=1.;
    for(IdxT r=1; r<W; r++) {
        FloatT val=exp(r*r*exp_norm);
        kernel(r)=val;
        sum+=2*val;
    }
    kernel/=sum;
    return arma::conv_to<VecT>::from(kernel);
}


template<class FloatT, class IdxT>
typename GaussFIRFilter<FloatT,IdxT>::VecT
GaussFIRFilter<FloatT,IdxT>::compute_LoG_FIR_kernel(FloatT sigma, IdxT hw)
{
    IdxT W=hw+1;  //full kernel size is hw*2+1, but we only compute the right half + center pixel
    arma::vec kernel(W);
    FloatT sigmanorm=1./(sigma*sigma);
    FloatT norm=sigmanorm/(sqrt(2*arma::Datum<FloatT>::pi)); //1/(sqrt(2*pi)*sigma^3) * sigma <-normalization term
    FloatT exp_norm=-0.5*sigmanorm;
    kernel(0)=norm;
//     FloatT sum=norm;
    for(IdxT r=1; r<W; r++) {
        FloatT rsq=r*r;
        FloatT val=norm*(1-rsq*sigmanorm)*exp(rsq*exp_norm);
        kernel(r)=val;
//         sum+=2*val;
    }
//     kernel(0)-=sum;
    return arma::conv_to<VecT>::from(kernel);
}




/* GaussFilter2D */

template<class FloatT, class IdxT>
GaussFilter2D<FloatT,IdxT>::GaussFilter2D(const IVecT &size, const VecT &sigma)
    : GaussFIRFilter<FloatT,IdxT>(2, size, sigma), kernels(2)
{
    auto hw=arma::conv_to<IVecT>::from(arma::ceil(this->default_sigma_hw_ratio * sigma));
    set_kernel_hw(hw);
    temp_im.set_size(size(0),size(1));
}

template<class FloatT, class IdxT>
GaussFilter2D<FloatT,IdxT>::GaussFilter2D(const IVecT &size, const VecT &sigma, const IVecT &kernel_hw)
    : GaussFIRFilter<FloatT,IdxT>(2, size, sigma), kernels(2)
{
    set_kernel_hw(kernel_hw);
    temp_im.set_size(size(0),size(1));
}


template<class FloatT, class IdxT>
void GaussFilter2D<FloatT,IdxT>::set_kernel_hw(const IVecT &kernel_half_width)
{
    if(!arma::all(kernel_half_width>0)){
        std::ostringstream msg;
        msg<<"Received bad kernel_half_width: "<<kernel_half_width.t();
        throw ParameterValueError(msg.str());
    }
    this->hw=kernel_half_width;
    for(IdxT d=0; d<this->dim; d++)
        kernels(d)=GaussFIRFilter<FloatT,IdxT>::compute_Gauss_FIR_kernel(this->sigma(d), this->hw(d));
}


template<class FloatT, class IdxT>
void GaussFilter2D<FloatT,IdxT>::filter(const ImageT &im, ImageT &out)
{
    gaussFIR_2Dx<FloatT>(im, temp_im, kernels(0));
    gaussFIR_2Dy<FloatT>(temp_im, out, kernels(1));
}

template<class FloatT, class IdxT>
void GaussFilter2D<FloatT,IdxT>::test_filter(const ImageT &im)
{
    ImageT fast_out=make_image();
    ImageT slow_out=make_image();
    filter(im, fast_out);
    gaussFIR_2Dx_small<FloatT>(im, temp_im, kernels(0));
    gaussFIR_2Dy_small<FloatT>(temp_im, slow_out, kernels(1));
    FloatT eps=4.*std::numeric_limits<FloatT>::epsilon();
    for(IdxT y=0; y<this->size(1); y++) for(IdxT x=0; x<this->size(0); x++)
        if( fabs(fast_out(x,y)-slow_out(x,y))>eps )
            printf("Fast (%i,%i):%.17f  != Slow (%i,%i):%.17f\n",x,y,fast_out(x,y),x,y,slow_out(x,y));
}

template<class FloatT, class IdxT>
std::ostream& operator<< (std::ostream &out, const GaussFilter2D<FloatT,IdxT> &filt)
{
    out<<std::setprecision(15);
    auto k0=filt.kernels(0);
    auto k1=filt.kernels(1);
    
    out<<"GaussFilter2D:[size=["<<filt.size(0)<<","<<filt.size(1)<<"]"
        <<" sigma=["<<filt.sigma(0)<<","<<filt.sigma(1)<<"]"
       <<" hw=["<<filt.hw(0)<<","<<filt.hw(1)<<"]"
       <<"\n >>KernelX:(sum:="<<2*arma::sum(k0)-k0(0)<<")\n"<<k0
       <<"\n >>KernelY:(sum:="<<2*arma::sum(k1)-k1(0)<<")\n"<<k1<<"\n";
    out<<std::setprecision(9);
    return out;
}


/* GaussFilter3D */

template<class FloatT, class IdxT>
GaussFilter3D<FloatT,IdxT>::GaussFilter3D(const IVecT &size, const VecT &sigma)
    : GaussFIRFilter<FloatT,IdxT>(3, size, sigma), kernels(3)
{
    auto hw=arma::conv_to<IVecT>::from(arma::ceil(this->default_sigma_hw_ratio * sigma));
    set_kernel_hw(hw);
    temp_im0.set_size(size(0),size(1),size(2));
    temp_im1.set_size(size(0),size(1),size(2));
}

template<class FloatT, class IdxT>
GaussFilter3D<FloatT,IdxT>::GaussFilter3D(const IVecT &size, const VecT &sigma, const IVecT &kernel_hw)
    : GaussFIRFilter<FloatT,IdxT>(3, size, sigma), kernels(3)
{
    set_kernel_hw(kernel_hw);
    temp_im0.set_size(size(0),size(1),size(2));
    temp_im1.set_size(size(0),size(1),size(2));
}


template<class FloatT, class IdxT>
void GaussFilter3D<FloatT,IdxT>::set_kernel_hw(const IVecT &kernel_half_width)
{
    if(!arma::all(kernel_half_width>0)){
        std::ostringstream msg;
        msg<<"Received bad kernel_half_width: "<<kernel_half_width.t();
        throw ParameterValueError(msg.str());
    }
    this->hw=kernel_half_width;
    for(IdxT d=0; d<this->dim; d++)
        kernels(d)=GaussFIRFilter<FloatT,IdxT>::compute_Gauss_FIR_kernel(this->sigma(d), this->hw(d));
}


template<class FloatT, class IdxT>
void GaussFilter3D<FloatT,IdxT>::filter(const ImageT &im, ImageT &out)
{
    kernels::gaussFIR_3Dx<FloatT>(im, temp_im0, kernels(0));
    kernels::gaussFIR_3Dy<FloatT>(temp_im0, temp_im1, kernels(1));
    kernels::gaussFIR_3Dz<FloatT>(temp_im1, out, kernels(2));
}

template<class FloatT, class IdxT>
void GaussFilter3D<FloatT,IdxT>::test_filter(const ImageT &im)
{
    ImageT fast_out=make_image();
    ImageT slow_out=make_image();
    filter(im, fast_out);
    kernels::gaussFIR_3Dx_small<FloatT>(im, temp_im0, kernels(0));
    kernels::gaussFIR_3Dy_small<FloatT>(temp_im0, temp_im1, kernels(1));
    kernels::gaussFIR_3Dz_small<FloatT>(temp_im1, slow_out, kernels(2));
    FloatT eps=4.*std::numeric_limits<FloatT>::epsilon();
    for(IdxT z=0; z<this->size(2); z++) for(IdxT y=0; y<this->size(1); y++) for(IdxT x=0; x<this->size(0); x++)
        if( fabs(fast_out(x,y,z)-slow_out(x,y,z))>eps )
            printf("Fast (%i,%i,%i):%.17f  != Slow (%i,%i,.%i):%.17f\n",x,y,z,fast_out(x,y,z),x,y,z,slow_out(x,y,z));
}

template<class FloatT, class IdxT>
std::ostream& operator<< (std::ostream &out, const GaussFilter3D<FloatT,IdxT> &filt)
{
    out<<std::setprecision(15);
    auto k0=filt.kernels(0);
    auto k1=filt.kernels(1);
    auto k2=filt.kernels(2);
    
    out<<"GaussFilter3D:[size=["<<filt.size(0)<<","<<filt.size(1)<<","<<filt.size(2)<<"]"
       <<" sigma=["<<filt.sigma(0)<<","<<filt.sigma(1)<<","<<filt.sigma(2)<<"]"
       <<" hw=["<<filt.hw(0)<<","<<filt.hw(1)<<","<<filt.hw(2)<<"]"
       <<"\n >>KernelX:(sum:="<<2*arma::sum(k0)-k0(0)<<")\n"<<k0
       <<"\n >>KernelY:(sum:="<<2*arma::sum(k1)-k1(0)<<")\n"<<k1<<"\n"
       <<"\n >>KernelZ:(sum:="<<2*arma::sum(k2)-k2(0)<<")\n"<<k2<<"\n";
    out<<std::setprecision(9);
    return out;
}

/* DoGFilter2D */
template<class FloatT, class IdxT>
DoGFilter2D<FloatT,IdxT>::DoGFilter2D(const IVecT &size, const VecT &sigma, FloatT sigma_ratio)
    : GaussFIRFilter<FloatT,IdxT>(2, size, sigma), sigma_ratio(sigma_ratio), excite_kernels(2), inhibit_kernels(2)
{
    if(!_sigma_ratio>1){
        std::ostringstream msg;
        msg<<"Received bad sigma_ratio: "<<sigma_ratio;
        throw ParameterValueError(msg.str());
    }
    auto hw=arma::conv_to<IVecT>::from(arma::ceil(this->default_sigma_hw_ratio * sigma));
    set_kernel_hw(hw);
    temp_im0.set_size(size(0),size(1));
    temp_im1.set_size(size(0),size(1));
}

template<class FloatT, class IdxT>
DoGFilter2D<FloatT,IdxT>::DoGFilter2D(const IVecT &size, const VecT &sigma, FloatT sigma_ratio, const IVecT &kernel_hw)
    : GaussFIRFilter<FloatT,IdxT>(2, size, sigma), sigma_ratio(sigma_ratio), excite_kernels(2), inhibit_kernels(2)
{
    if(!_sigma_ratio>1){
        std::ostringstream msg;
        msg<<"Received bad sigma_ratio: "<<sigma_ratio;
        throw ParameterValueError(msg.str());
    }
    set_kernel_hw(kernel_hw);
    temp_im0.set_size(size(0),size(1));
    temp_im1.set_size(size(0),size(1));
}


template<class FloatT, class IdxT>
void DoGFilter2D<FloatT,IdxT>::set_kernel_hw(const IVecT &kernel_half_width)
{
    if(!arma::all(kernel_half_width>0)){
        std::ostringstream msg;
        msg<<"Received bad kernel_half_width: "<<kernel_half_width.t();
        throw ParameterValueError(msg.str());
    }
    this->hw=kernel_half_width;
    for(IdxT d=0; d<this->dim; d++) {
        excite_kernels(d)=GaussFIRFilter<FloatT,IdxT>::compute_Gauss_FIR_kernel(this->sigma(d), this->hw(d));
        inhibit_kernels(d)=GaussFIRFilter<FloatT,IdxT>::compute_Gauss_FIR_kernel(this->sigma(d)*sigma_ratio, this->hw(d));
    }
}

template<class FloatT, class IdxT>
void DoGFilter2D<FloatT,IdxT>::set_sigma_ratio(FloatT _sigma_ratio)
{
    if(!_sigma_ratio>1){
        std::ostringstream msg;
        msg<<"Received bad sigma_ratio: "<<sigma_ratio;
        throw ParameterValueError(msg.str());
    }
    sigma_ratio = _sigma_ratio;
    set_kernel_hw(this->hw);
}

template<class FloatT, class IdxT>
void DoGFilter2D<FloatT,IdxT>::filter(const ImageT &im, ImageT &out)
{
    kernels::gaussFIR_2Dx<FloatT>(im, temp_im0, excite_kernels(0));
    kernels::gaussFIR_2Dy<FloatT>(temp_im0, out, excite_kernels(1));
    
    kernels::gaussFIR_2Dx<FloatT>(im, temp_im1, inhibit_kernels(0));
    kernels::gaussFIR_2Dy<FloatT>(temp_im1, temp_im0, inhibit_kernels(1));
    out-=temp_im0;
}

template<class FloatT, class IdxT>
void DoGFilter2D<FloatT,IdxT>::test_filter(const ImageT &im)
{
    ImageT fast_out=make_image();
    ImageT slow_out=make_image();
    filter(im, fast_out);
    kernels::gaussFIR_2Dx_small<FloatT>(im, temp_im0, excite_kernels(0));
    kernels::gaussFIR_2Dy_small<FloatT>(temp_im0, slow_out, excite_kernels(1));
    
    kernels::gaussFIR_2Dx_small<FloatT>(im, temp_im1, inhibit_kernels(0));
    kernels::gaussFIR_2Dy_small<FloatT>(temp_im1, temp_im0, inhibit_kernels(1));
    slow_out-=temp_im0;
    FloatT eps=4.*std::numeric_limits<FloatT>::epsilon();
    for(IdxT y=0; y<this->size(1); y++) for(IdxT x=0; x<this->size(0); x++)
        if( fabs(fast_out(x,y)-slow_out(x,y))>eps )
            printf("Fast (%i,%i):%.17f  != Slow (%i,%i):%.17f\n",x,y,fast_out(x,y),x,y,slow_out(x,y));
}

/* DoGFilter3D */
template<class FloatT, class IdxT>
DoGFilter3D<FloatT,IdxT>::DoGFilter3D(const IVecT &size, const VecT &sigma, FloatT sigma_ratio)
    : GaussFIRFilter<FloatT,IdxT>(3, size, sigma), sigma_ratio(sigma_ratio), excite_kernels(3), inhibit_kernels(3)
{
    if(!_sigma_ratio>1){
        std::ostringstream msg;
        msg<<"Received bad sigma_ratio: "<<sigma_ratio;
        throw ParameterValueError(msg.str());
    }
    auto hw=arma::conv_to<IVecT>::from(arma::ceil(this->default_sigma_hw_ratio * sigma));
    set_kernel_hw(hw);
    temp_im0.set_size(size(0),size(1),size(2));
    temp_im1.set_size(size(0),size(1),size(2));
}

template<class FloatT, class IdxT>
DoGFilter3D<FloatT,IdxT>::DoGFilter3D(const IVecT &size, const VecT &sigma, FloatT sigma_ratio, const IVecT &kernel_hw)
    : GaussFIRFilter<FloatT,IdxT>(3, size, sigma), sigma_ratio(sigma_ratio), excite_kernels(3), inhibit_kernels(3)
{
    if(!_sigma_ratio>1){
        std::ostringstream msg;
        msg<<"Received bad sigma_ratio: "<<sigma_ratio;
        throw ParameterValueError(msg.str());
    }
    set_kernel_hw(kernel_hw);
    temp_im0.set_size(size(0),size(1),size(2));
    temp_im1.set_size(size(0),size(1),size(2));
}


template<class FloatT, class IdxT>
void DoGFilter3D<FloatT,IdxT>::set_kernel_hw(const IVecT &kernel_half_width)
{
    if(!arma::all(kernel_half_width>0)){
        std::ostringstream msg;
        msg<<"Received bad kernel_half_width: "<<kernel_half_width.t();
        throw ParameterValueError(msg.str());
    }
    this->hw=kernel_half_width;
    for(IdxT d=0; d<this->dim; d++) {
        excite_kernels(d)=GaussFIRFilter<FloatT,IdxT>::compute_Gauss_FIR_kernel(this->sigma(d), this->hw(d));
        inhibit_kernels(d)=GaussFIRFilter<FloatT,IdxT>::compute_Gauss_FIR_kernel(this->sigma(d)*sigma_ratio, this->hw(d));
    }
}

template<class FloatT, class IdxT>
void DoGFilter3D<FloatT,IdxT>::set_sigma_ratio(FloatT _sigma_ratio)
{
    if(!_sigma_ratio>1){
        std::ostringstream msg;
        msg<<"Received bad sigma_ratio: "<<sigma_ratio;
        throw ParameterValueError(msg.str());
    }
    sigma_ratio = _sigma_ratio;
    set_kernel_hw(this->hw);
}

template<class FloatT, class IdxT>
void DoGFilter3D<FloatT,IdxT>::filter(const ImageT &im, ImageT &out)
{
    kernels::gaussFIR_3Dx<FloatT>(im, temp_im0, excite_kernels(0));
    kernels::gaussFIR_3Dy<FloatT>(temp_im0, temp_im1, excite_kernels(1));
    kernels::gaussFIR_3Dz<FloatT>(temp_im1, out, excite_kernels(2));
    
    kernels::gaussFIR_3Dx<FloatT>(im, temp_im0, inhibit_kernels(0));
    kernels::gaussFIR_3Dy<FloatT>(temp_im0, temp_im1, inhibit_kernels(1));
    kernels::gaussFIR_3Dz<FloatT>(temp_im1, temp_im0, inhibit_kernels(2));
    out-=temp_im0;
}

template<class FloatT, class IdxT>
void DoGFilter3D<FloatT,IdxT>::test_filter(const ImageT &im)
{
    ImageT fast_out=make_image();
    ImageT slow_out=make_image();
    filter(im, fast_out);    
    kernels::gaussFIR_3Dx_small<FloatT>(im, temp_im0, excite_kernels(0));
    kernels::gaussFIR_3Dy_small<FloatT>(temp_im0, temp_im1, excite_kernels(1));
    kernels::gaussFIR_3Dz_small<FloatT>(temp_im1, slow_out, excite_kernels(2));
    
    kernels::gaussFIR_3Dx_small<FloatT>(im, temp_im0, inhibit_kernels(0));
    kernels::gaussFIR_3Dy_small<FloatT>(temp_im0, temp_im1, inhibit_kernels(1));
    kernels::gaussFIR_3Dz_small<FloatT>(temp_im1, temp_im0, inhibit_kernels(2));
    slow_out-=temp_im0;
    FloatT eps=4.*std::numeric_limits<FloatT>::epsilon();
    for(IdxT z=0; z<this->size(2); z++) for(IdxT y=0; y<this->size(1); y++) for(IdxT x=0; x<this->size(0); x++)
        if( fabs(fast_out(x,y,z)-slow_out(x,y,z))>eps )
            printf("Fast (%i,%i,%i):%.17f  != Slow (%i,%i,%i):%.17f\n",x,y,z,fast_out(x,y,z),x,y,z,slow_out(x,y,z));
}


/* LoGFilter2D */

template<class FloatT, class IdxT>
LoGFilter2D<FloatT,IdxT>::LoGFilter2D(const IVecT &size, const VecT &sigma)
    : GaussFIRFilter<FloatT,IdxT>(2, size, sigma), gauss_kernels(2), LoG_kernels(2)
{
    auto hw=arma::conv_to<IVecT>::from(arma::ceil(this->default_sigma_hw_ratio * sigma));
    set_kernel_hw(hw);
    temp_im0.set_size(size(0),size(1));
    temp_im1.set_size(size(0),size(1));
}

template<class FloatT, class IdxT>
LoGFilter2D<FloatT,IdxT>::LoGFilter2D(const IVecT &size, const VecT &sigma, const IVecT &kernel_hw)
    : GaussFIRFilter<FloatT,IdxT>(2, size, sigma), gauss_kernels(2), LoG_kernels(2)
{
    set_kernel_hw(kernel_hw);
    temp_im0.set_size(size(0),size(1));
    temp_im1.set_size(size(0),size(1));
}


template<class FloatT, class IdxT>
void LoGFilter2D<FloatT,IdxT>::set_kernel_hw(const IVecT &kernel_half_width)
{
    if(!arma::all(kernel_half_width>0)){
        std::ostringstream msg;
        msg<<"Received bad kernel_half_width: "<<kernel_half_width.t();
        throw ParameterValueError(msg.str());
    }
    this->hw = kernel_half_width;
    for(IdxT d=0; d<this->dim; d++) {
        gauss_kernels(d)=GaussFIRFilter<FloatT,IdxT>::compute_Gauss_FIR_kernel(this->sigma(d), this->hw(d));
        LoG_kernels(d)=GaussFIRFilter<FloatT,IdxT>::compute_LoG_FIR_kernel(this->sigma(d), this->hw(d));
    }
}



template<class FloatT, class IdxT>
void LoGFilter2D<FloatT,IdxT>::filter(const ImageT &im, ImageT &out)
{
    kernels::gaussFIR_2Dy<FloatT>(im, temp_im0, LoG_kernels(1)); //G''(y)fc
    kernels::gaussFIR_2Dx<FloatT>(temp_im0, out, gauss_kernels(0)); //G(x)

    kernels::gaussFIR_2Dy<FloatT>(im, temp_im0, gauss_kernels(1)); //G(y)
    kernels::gaussFIR_2Dx<FloatT>(temp_im0, temp_im1, LoG_kernels(0)); //G''(x)
    out+=temp_im1;

//     kernels::gaussFIR_2Dy<FloatT>(im, temp_im0, LoG_kernels(1)); //G''(y)fc
//     kernels::gaussFIR_2Dx<FloatT>(im, temp_im1, gauss_kernels(0)); //G(x)
//     temp_im0 = temp_im0 % temp_im1; //straight product
//
//     kernels::gaussFIR_2Dy<FloatT>(im, temp_im1, gauss_kernels(1)); //G(y)
//     kernels::gaussFIR_2Dx<FloatT>(im, out, LoG_kernels(0)); //G''(x)
//     out = (out%temp_im1)+temp_im0;
}

template<class FloatT, class IdxT>
void LoGFilter2D<FloatT,IdxT>::test_filter(const ImageT &im)
{
    ImageT fast_out=make_image();
    ImageT slow_out=make_image();
    filter(im, fast_out);
    kernels::gaussFIR_2Dy_small<FloatT>(im, temp_im0, LoG_kernels(1)); //G''(y)fc
    kernels::gaussFIR_2Dx_small<FloatT>(temp_im0, slow_out, gauss_kernels(0)); //G(x)

    kernels::gaussFIR_2Dy_small<FloatT>(im, temp_im0, gauss_kernels(1)); //G(y)
    kernels::gaussFIR_2Dx_small<FloatT>(temp_im0, temp_im1, LoG_kernels(0)); //G''(x)
    slow_out+=temp_im1;

    FloatT eps=4.*std::numeric_limits<FloatT>::epsilon();
    for(IdxT y=0; y<this->size(1); y++) for(IdxT x=0; x<this->size(0); x++)
        if( fabs(fast_out(x,y)-slow_out(x,y))>eps )
            printf("Fast (%i,%i):%.17f  != Slow (%i,%i):%.17f\n",x,y,fast_out(x,y),x,y,slow_out(x,y));
}

template<class FloatT, class IdxT>
std::ostream& operator<< (std::ostream &out, const LoGFilter2D<FloatT,IdxT> &filt)
{
    out<<std::setprecision(15);
    auto gk0 = filt.gauss_kernels(0);
    auto gk1 = filt.gauss_kernels(1);
    auto logk0 = filt.LoG_kernels(0);
    auto logk1 = filt.LoG_kernels(1);
    
    out<<"LoGFilter2D:[size=["<<filt.size(0)<<","<<filt.size(1)<<"]"
        <<" sigma=["<<filt.sigma(0)<<","<<filt.sigma(1)<<"]"
       <<" hw=["<<filt.hw(0)<<","<<filt.hw(1)<<"]"
       <<"\n >>GaussKernelX:(sum:="<<2*arma::sum(gk0)-gk0(0)<<")\n"<<gk0
       <<"\n >>GaussKernelY:(sum:="<<2*arma::sum(gk1)-gk1(0)<<")\n"<<gk1
       <<"\n >>LoGKernelX:(sum:="<<2*arma::sum(logk0)-logk0(0)<<")\n"<<logk0
       <<"\n >>LoGKernelY:(sum:="<<2*arma::sum(logk1)-logk1(0)<<")\n"<<logk1<<"\n";
    out<<std::setprecision(9);
    return out;
}

/* LoGFilter3D */

template<class FloatT, class IdxT>
LoGFilter3D<FloatT,IdxT>::LoGFilter3D(const IVecT &size, const VecT &sigma)
    : GaussFIRFilter<FloatT,IdxT>(3, size, sigma), gauss_kernels(3), LoG_kernels(3)
{
    auto hw=arma::conv_to<IVecT>::from(arma::ceil(this->default_sigma_hw_ratio * sigma));
    set_kernel_hw(hw);
    temp_im0.set_size(size(0),size(1),size(2));
    temp_im1.set_size(size(0),size(1),size(2));
//     temp_im2.set_size(size(0),size(1),size(2));
}

template<class FloatT, class IdxT>
LoGFilter3D<FloatT,IdxT>::LoGFilter3D(const IVecT &size, const VecT &sigma, const IVecT &kernel_hw)
    : GaussFIRFilter<FloatT,IdxT>(3, size, sigma), gauss_kernels(3), LoG_kernels(3)
{
    set_kernel_hw(kernel_hw);
    temp_im0.set_size(size(0),size(1),size(2));
    temp_im1.set_size(size(0),size(1),size(2));
//     temp_im2.set_size(size(0),size(1),size(2));
}


template<class FloatT, class IdxT>
void LoGFilter3D<FloatT,IdxT>::set_kernel_hw(const IVecT &kernel_half_width)
{
    if(!arma::all(kernel_half_width>0)){
        std::ostringstream msg;
        msg<<"Received bad kernel_half_width: "<<kernel_half_width.t();
        throw ParameterValueError(msg.str());
    }
    this->hw=kernel_half_width;
    for(IdxT d=0; d<this->dim; d++) {
        gauss_kernels(d)=GaussFIRFilter<FloatT,IdxT>::compute_Gauss_FIR_kernel(this->sigma(d), this->hw(d));
        LoG_kernels(d)=GaussFIRFilter<FloatT,IdxT>::compute_LoG_FIR_kernel(this->sigma(d), this->hw(d));
    }
}


template<class FloatT, class IdxT>
void LoGFilter3D<FloatT,IdxT>::filter(const ImageT &im, ImageT &out)
{
    kernels::gaussFIR_3Dz<FloatT>(im, temp_im0, gauss_kernels(2));
    kernels::gaussFIR_3Dy<FloatT>(temp_im0, temp_im1, gauss_kernels(1));
    kernels::gaussFIR_3Dx<FloatT>(temp_im1, out, LoG_kernels(0));

    kernels::gaussFIR_3Dz<FloatT>(im, temp_im0, gauss_kernels(2));
    kernels::gaussFIR_3Dy<FloatT>(temp_im0, temp_im1, LoG_kernels(1));
    kernels::gaussFIR_3Dx<FloatT>(temp_im1, temp_im0, gauss_kernels(0));
    out+=temp_im0;

    kernels::gaussFIR_3Dz<FloatT>(im, temp_im0, LoG_kernels(2));
    kernels::gaussFIR_3Dy<FloatT>(temp_im0, temp_im1, gauss_kernels(1));
    kernels::gaussFIR_3Dx<FloatT>(temp_im1, temp_im0, gauss_kernels(0));
    out+=temp_im0;
//     kernels::gaussFIR_3Dz<FloatT>(im, temp_im0, gauss_kernels(2));
//     kernels::gaussFIR_3Dy<FloatT>(im, temp_im1, gauss_kernels(1));
//     kernels::gaussFIR_3Dx<FloatT>(im, out, LoG_kernels(0));
//     out = out%temp_im0%temp_im1;
//
//     kernels::gaussFIR_3Dz<FloatT>(im, temp_im0, gauss_kernels(2));
//     kernels::gaussFIR_3Dy<FloatT>(im, temp_im1, LoG_kernels(1));
//     temp_im0 = temp_im0%temp_im1;
//     kernels::gaussFIR_3Dx<FloatT>(im, temp_im1, gauss_kernels(0));
//     out += temp_im0%temp_im1;
//
//     kernels::gaussFIR_3Dz<FloatT>(im, temp_im0, LoG_kernels(2));
//     kernels::gaussFIR_3Dy<FloatT>(im, temp_im1, gauss_kernels(1));
//     temp_im0 = temp_im0%temp_im1;
//     kernels::gaussFIR_3Dx<FloatT>(im, temp_im0, gauss_kernels(0));
//     out += temp_im0%temp_im1;
}

template<class FloatT, class IdxT>
void LoGFilter3D<FloatT,IdxT>::test_filter(const ImageT &im)
{
    ImageT fast_out=make_image();
    ImageT slow_out=make_image();
    filter(im, fast_out);
    kernels::gaussFIR_3Dz_small<FloatT>(im, temp_im0, gauss_kernels(2));
    kernels::gaussFIR_3Dy_small<FloatT>(temp_im0, temp_im1, gauss_kernels(1));
    kernels::gaussFIR_3Dx_small<FloatT>(temp_im1, slow_out, LoG_kernels(0));

    kernels::gaussFIR_3Dz_small<FloatT>(im, temp_im0, gauss_kernels(2));
    kernels::gaussFIR_3Dy_small<FloatT>(temp_im0, temp_im1, LoG_kernels(1));
    kernels::gaussFIR_3Dx_small<FloatT>(temp_im1, temp_im0, gauss_kernels(0));
    slow_out+=temp_im0;

    kernels::gaussFIR_3Dz_small<FloatT>(im, temp_im0, LoG_kernels(2));
    kernels::gaussFIR_3Dy_small<FloatT>(temp_im0, temp_im1, gauss_kernels(1));
    kernels::gaussFIR_3Dx_small<FloatT>(temp_im1, temp_im0, gauss_kernels(0));
    slow_out+=temp_im0;
    FloatT eps=4.*std::numeric_limits<FloatT>::epsilon();
    for(IdxT z=0; z<this->size(2); z++) for(IdxT y=0; y<this->size(1); y++) for(IdxT x=0; x<this->size(0); x++)
        if( fabs(fast_out(x,y,z)-slow_out(x,y,z))>eps )
            printf("Fast (%i,%i,%i):%.17f  != Slow (%i,%i,.%i):%.17f\n",x,y,z,fast_out(x,y,z),x,y,z,slow_out(x,y,z));
}

template<class FloatT, class IdxT>
std::ostream& operator<< (std::ostream &out, const LoGFilter3D<FloatT,IdxT> &filt)
{
    out<<std::setprecision(15);
    auto gk0=filt.gauss_kernels(0);
    auto gk1=filt.gauss_kernels(1);
    auto gk2=filt.gauss_kernels(2);
    auto logk0=filt.LoG_kernels(0);
    auto logk1=filt.LoG_kernels(1);
    auto logk2=filt.LoG_kernels(2);
    
    out<<"LoGFilter3D:[size=["<<filt.size(0)<<","<<filt.size(1)<<","<<filt.size(2)<<"]"
       <<" sigma=["<<filt.sigma(0)<<","<<filt.sigma(1)<<","<<filt.sigma(2)<<"]"
       <<" hw=["<<filt.hw(0)<<","<<filt.hw(1)<<","<<filt.hw(2)<<"]"
       <<"\n >>Gauss KernelX:(sum:="<<2*arma::sum(gk0)-gk0(0)<<")\n"<<gk0
       <<"\n >>Gauss KernelY:(sum:="<<2*arma::sum(gk1)-gk1(0)<<")\n"<<gk1<<"\n"
       <<"\n >>Gauss KernelZ:(sum:="<<2*arma::sum(gk2)-gk2(0)<<")\n"<<gk2<<"\n"
       <<"\n >>LoG KernelX:(sum:="<<2*arma::sum(logk0)-logk0(0)<<")\n"<<logk0<<"\n"
       <<"\n >>LoG KernelY:(sum:="<<2*arma::sum(logk1)-logk1(0)<<")\n"<<logk1<<"\n"
       <<"\n >>LoG KernelZ:(sum:="<<2*arma::sum(logk2)-logk2(0)<<")\n"<<logk2<<"\n";
    out<<std::setprecision(9);
    return out;
}



/* Explicit Template Instantiation */
template class GaussFIRFilter<float>;
template class GaussFIRFilter<double>;

template class GaussFilter2D<float>;
template class GaussFilter2D<double>;

template class GaussFilter3D<float>;
template class GaussFilter3D<double>;

template class DoGFilter2D<float>;
template class DoGFilter2D<double>;

template class DoGFilter3D<float>;
template class DoGFilter3D<double>;

template class LoGFilter2D<float>;
template class LoGFilter2D<double>;

template class LoGFilter3D<float>;
template class LoGFilter3D<double>;


template std::ostream& operator<< <float>(std::ostream &out, const GaussFilter2D<float> &filt);
template std::ostream& operator<< <double>(std::ostream &out, const GaussFilter2D<double> &filt);

template std::ostream& operator<< <float>(std::ostream &out, const GaussFilter3D<float> &filt);
template std::ostream& operator<< <double>(std::ostream &out, const GaussFilter3D<double> &filt);

template std::ostream& operator<< <float>(std::ostream &out, const LoGFilter2D<float> &filt);
template std::ostream& operator<< <double>(std::ostream &out, const LoGFilter2D<double> &filt);

template std::ostream& operator<< <float>(std::ostream &out, const LoGFilter3D<float> &filt);
template std::ostream& operator<< <double>(std::ostream &out, const LoGFilter3D<double> &filt);

} /* namespace boxxer */
