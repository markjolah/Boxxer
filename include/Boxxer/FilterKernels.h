/** @file FilterKernels.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The boxxer::kernels namespace - low-level Gaussian finite-impulse response filters.
 *
 */
#ifndef BOXXER_FILTER_KERNELS_H
#define BOXXER_FILTER_KERNELS_H

#include <cstdint>
#include <armadillo>

namespace boxxer {

/** Gaussian finite-impulse response kernels: 1D, 2D, and 3D
 *
 * Template parameters for all namespace member functions.
 * FloatT - float (default) or double.
 * IntT - **signed** integer:  int32_t (default) or int64_t.
 *
 * All kernels are explicitly instantiated for:
 *  - FloatT = float, IntT = int32_t
 *  - FloatT = double, IntT = int32_t
 *
 * There is no testing for integer overflow. Probably int32_t will be sensible for most applications.
 *
 */
namespace kernels {

/**@{*/
/** 1D Gauss FIR Filters */
template <class FloatT=float, class IntT=int32_t>
void gaussFIR_1D(IntT size, const FloatT data[], FloatT fdata[], IntT hw, const FloatT kernel[]);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_1D(const arma::Col<FloatT> &data, arma::Col<FloatT> &fdata, const arma::Col<FloatT> &kernel)
{
    IntT hw = static_cast<IntT>(kernel.n_elem)-1;
    IntT size = static_cast<IntT>(data.n_elem);
    gaussFIR_1D<FloatT,IntT>(size, data.memptr(), fdata.memptr(), hw, kernel.memptr());
}

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_1D_small(IntT size, const FloatT data[], FloatT fdata[], IntT hw, const FloatT kernel[]);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_1D_arma(const arma::Col<FloatT> &data, arma::Col<FloatT> &fdata, const arma::Col<FloatT> &kernel);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_1D_inplace_arma(arma::Col<FloatT> &data, const arma::Col<FloatT> &kernel);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_1D_inplace(IntT size, FloatT data[], IntT hw, const FloatT kernel[]);
/**@}*/

/**@{*/
/** 2D Gauss FIR Filters */
template <class FloatT=float, class IntT=int32_t>
void gaussFIR_2Dx(const arma::Mat<FloatT> &data, arma::Mat<FloatT> &fdata, const arma::Col<FloatT> &kernel);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_2Dx_small(const arma::Mat<FloatT> &data, arma::Mat<FloatT> &fdata, const arma::Col<FloatT> &kernel);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_2Dx_arma(const arma::Mat<FloatT> &data, arma::Mat<FloatT> &fdata, const arma::Col<FloatT> &kernel);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_2Dy(const arma::Mat<FloatT> &data, arma::Mat<FloatT> &fdata, const arma::Col<FloatT> &kernel);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_2Dy_rowmajor(const arma::Mat<FloatT> &data, arma::Mat<FloatT> &fdata, const arma::Col<FloatT> &kernel);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_2Dy_colmajor(const arma::Mat<FloatT> &data, arma::Mat<FloatT> &fdata, const arma::Col<FloatT> &kernel);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_2Dy_small(const arma::Mat<FloatT> &data, arma::Mat<FloatT> &fdata, const arma::Col<FloatT> &kernel);
/**@}*/

/**@{*/
/** 3D Gauss FIR Filters */
template <class FloatT=float, class IntT=int32_t>
void gaussFIR_3Dx(const arma::Cube<FloatT> &data, arma::Cube<FloatT> &fdata, const arma::Col<FloatT> &kernel);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_3Dx_small(const arma::Cube<FloatT> &data, arma::Cube<FloatT> &fdata, const arma::Col<FloatT> &kernel);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_3Dy(const arma::Cube<FloatT> &data, arma::Cube<FloatT> &fdata, const arma::Col<FloatT> &kernel);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_3Dy_small(const arma::Cube<FloatT> &data, arma::Cube<FloatT> &fdata, const arma::Col<FloatT> &kernel);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_3Dz(const arma::Cube<FloatT> &data, arma::Cube<FloatT> &fdata, const arma::Col<FloatT> &kernel);

template <class FloatT=float, class IntT=int32_t>
void gaussFIR_3Dz_small(const arma::Cube<FloatT> &data, arma::Cube<FloatT> &fdata, const arma::Col<FloatT> &kernel);
/**@}*/

} /* namespace boxxer::kernels */

} /* namespace boxxer */

#endif /* BOXXER_FILTER_KERNELS_H */
