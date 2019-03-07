/**
 * @file Maxima.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class declaration for the local maxima finders Maxima2D and Maxima3D.
 */
#ifndef BOXXER_MAXIMA_H
#define BOXXER_MAXIMA_H

#include <armadillo>

namespace boxxer {

template <class FloatT=float, class IdxT=uint32_t>
class Maxima2D
{
public:
    using IVecT = arma::Col<IdxT>;
    using IMatT = arma::Mat<IdxT>;
    using VecT = arma::Col<FloatT>;
    using ImageT = arma::Mat<FloatT>;
    static const IdxT MinBoxsize;
    static const IdxT Ndim;
    IVecT size;
    IdxT boxsize;
    Maxima2D(const IVecT &sizeX, IdxT boxsize=3);
    IdxT find_maxima(const ImageT &im);
    IdxT find_maxima(const ImageT &im, IMatT &maxima_out, VecT &max_vals_out);
    void read_maxima(IdxT Nmaxima, IMatT &maxima_out, VecT &max_vals_out) const;
    void test_maxima(const ImageT &im);
private:
    IdxT max_maxima;//size of maxima and max_vals array
    IMatT maxima;// 2xN.
    VecT max_vals; //Nx1
    IMatT skip_buf;

    void detect_maxima(IdxT &Nmaxima, IdxT x, IdxT y, FloatT val);
    IdxT maxima_3x3(const ImageT &im);
    IdxT maxima_3x3_edges(const ImageT &im);
    IdxT maxima_3x3_slow(const ImageT &im);
    IdxT maxima_5x5(const ImageT &im);
    IdxT maxima_nxn(const ImageT &im, IdxT filter_size);
};


template <class FloatT=float, class IdxT=uint32_t>
class Maxima3D
{
public:
    using IVecT = arma::Col<IdxT>;
    using IMatT = arma::Mat<IdxT>;
    using ICubeT = arma::Cube<IdxT>;
    using VecT = arma::Col<FloatT>;
    using ImageT = arma::Cube<FloatT>;

    const IdxT dim=3;
    IVecT size;
    IdxT boxsize;
    Maxima3D(const IVecT &size, IdxT boxsize=3);
    IdxT find_maxima(const ImageT &im);
    IdxT find_maxima(const ImageT &im, IMatT &maxima_out, VecT &max_vals_out);
    void read_maxima(IdxT Nmaxima, IMatT &maxima_out, VecT &max_vals_out) const;
    void test_maxima(const ImageT &im);
private:
    IdxT max_maxima;//size of maxima and max_vals array
    IMatT maxima;// 2xN.
    VecT max_vals; //Nx1
    IMatT skip_buf;
    ICubeT skip_plane_buf;

    void detect_maxima(IdxT &Nmaxima, IdxT x, IdxT y, IdxT z, FloatT val);
    IdxT maxima_3x3(const ImageT &im);
    IdxT maxima_3x3_edges(const ImageT &im);
    IdxT maxima_3x3_slow(const ImageT &im);
    IdxT maxima_5x5(const ImageT &im);
    IdxT maxima_nxn(const ImageT &im, IdxT filter_size);
};

} /* namespace boxxer */

#endif /* BOXXER_MAXIMA_H */
