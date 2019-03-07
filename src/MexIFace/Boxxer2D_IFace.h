/** @file Boxxer2D_IFace.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief Boxxer2D MexIFace
 */

#ifndef BOXXER_BOXXER2D_IFACE
#define BOXXER_BOXXER2D_IFACE

#include<functional>

#include "MexIFace/MexIFace.h"
#include "Boxxer/Boxxer2D.h"

using namespace boxxer;

template<class FloatT, class IdxT>
class Boxxer2D_IFace : public mexiface::MexIFace,
                       public mexiface::MexIFaceHandler<Boxxer2D<FloatT,IdxT>>
{
public:
    Boxxer2D_IFace();

private:
    using BoxxerT = Boxxer2D<FloatT,IdxT>;
    using mexiface::MexIFaceHandler<BoxxerT>::obj;
    using IMatT = typename BoxxerT::IMatT;
    using VecT = typename BoxxerT::VecT;

    //Constructor
    void objConstruct() override;

    //Non-static member function calls
    void objSetDoGSigmaRatio();
    void objFilterScaledLoG();
    void objFilterScaledDoG();
    void objScaleSpaceLoGMaxima();
    void objScaleSpaceDoGMaxima();

    // Static member function wrappers
    void objFilterLoG();
    void objFilterDoG();
    void objFilterGauss();
    void objEnumerateImageMaxima();
};

template<class FloatT, class IdxT>
Boxxer2D_IFace<FloatT,IdxT>::Boxxer2D_IFace()
{
    methodmap["setDoGSigmaRatio"] = std::bind(&Boxxer2D_IFace::objSetDoGSigmaRatio, this);
    methodmap["filterScaledLoG"] = std::bind(&Boxxer2D_IFace::objFilterScaledLoG, this);
    methodmap["filterScaledDoG"] = std::bind(&Boxxer2D_IFace::objFilterScaledDoG, this);
    methodmap["scaleSpaceLoGMaxima"] = std::bind(&Boxxer2D_IFace::objScaleSpaceLoGMaxima, this);
    methodmap["scaleSpaceDoGMaxima"] = std::bind(&Boxxer2D_IFace::objScaleSpaceDoGMaxima, this);

    staticmethodmap["filterLoG"] = std::bind(&Boxxer2D_IFace::objFilterLoG, this);
    staticmethodmap["filterDoG"] = std::bind(&Boxxer2D_IFace::objFilterDoG, this);
    staticmethodmap["filterGauss"] = std::bind(&Boxxer2D_IFace::objFilterGauss, this);
    staticmethodmap["enumerateImageMaxima"] = std::bind(&Boxxer2D_IFace::objEnumerateImageMaxima, this);
}

template<class FloatT, class IdxT>
void Boxxer2D_IFace<FloatT,IdxT>::objConstruct()
{
    // [in] imsize - size:[2] type IdxT. Image size [X Y]
    // [in] sigma - size:[2, nScales] type FloatT.  scale-space sigmas Each column is a scale.
    // [out] handle - A new MexIFace object handle
    checkNumArgs(1,2);
    auto imsize = getVec<IdxT>();
    auto sigma = getMat<FloatT>();
    this->outputHandle(new BoxxerT(imsize,sigma));
}

template<class FloatT, class IdxT>
void Boxxer2D_IFace<FloatT,IdxT>::objSetDoGSigmaRatio()
{
    // [in] sigma_ratio: a new sigma_ratio>1
    checkNumArgs(0,1);
    obj->setDoGSigmaRatio(getAsFloat<FloatT>());
}

template<class FloatT, class IdxT>
void Boxxer2D_IFace<FloatT,IdxT>::objFilterScaledLoG()
{
    // [in] image: stack of imsize shaped frames, last dimension is time
    // [out] fimage: stack of imsize x nScales filtered frames. Size 4D: [x y S t]
    checkNumArgs(1,1);
    auto ims = getCube<FloatT>();
    auto fims = makeOutputArray<FloatT>(ims.n_rows, ims.n_cols, obj->nScales, ims.n_slices);
    obj->filterScaledLoG(ims,fims);
}

template<class FloatT, class IdxT>
void Boxxer2D_IFace<FloatT,IdxT>::objFilterScaledDoG()
{
    // [in] image: stack of imsize shaped frames, last dimension is time
    // [out] fimage: stack of imsize x nScales filtered frames. Size 4D: [x y S t]
    checkNumArgs(1,1);
    auto ims = getCube<FloatT>();
    auto fims = makeOutputArray<FloatT>(ims.n_rows, ims.n_cols, obj->nScales, ims.n_slices);
    obj->filterScaledDoG(ims,fims);
}

template<class FloatT, class IdxT>
void Boxxer2D_IFace<FloatT,IdxT>::objScaleSpaceLoGMaxima()
{
    // [in] image: Stack of imsize shaped frames, last dimension is time
    // [in] neighborhoodSize: Odd integer.  Acceptable values are in ValidMaximaNeighborhoodSizes.  (default=3)
    // [in] scaleNeighborhoodSize: Odd integer.  Acceptable values are in ValidMaximaNeighborhoodSizes.  (default=3)
    // [out] maxima: matrix type IdxT size:[dim+1, N]. List of maxima where rows are X, Y, ..., T and columns are different maxima detected.
    // [out] max_vals; type FloatT size:[N], vector of values at each maxima.
    checkNumArgs(2,3);
    auto ims = getCube<FloatT>();
    auto neighborhoodSize = getAsUnsigned<IdxT>();
    auto scaleNeighborhoodSize = getAsUnsigned<IdxT>();
    IMatT maxima;
    VecT max_vals;
    obj->scaleSpaceLoGMaxima(ims, maxima, max_vals, neighborhoodSize, scaleNeighborhoodSize);
    output(maxima);
    output(max_vals);
}

template<class FloatT, class IdxT>
void Boxxer2D_IFace<FloatT,IdxT>::objScaleSpaceDoGMaxima()
{
    // [in] image: Stack of imsize shaped frames, last dimension is time
    // [in] neighborhoodSize: Odd integer.  Acceptable values are in ValidMaximaNeighborhoodSizes.  (default=3)
    // [in] scaleNeighborhoodSize: Odd integer.  Acceptable values are in ValidMaximaNeighborhoodSizes.  (default=3)
    // [out] maxima: matrix type IdxT size:[dim+1, N]. List of maxima where rows are X, Y, ..., T and columns are different maxima detected.
    // [out] max_vals; type FloatT size:[N], vector of values at each maxima.
    checkNumArgs(2,3);
    auto ims = getCube<FloatT>();
    auto neighborhoodSize = getAsUnsigned<IdxT>();
    auto scaleNeighborhoodSize = getAsUnsigned<IdxT>();
    IMatT maxima;
    VecT max_vals;
    obj->scaleSpaceDoGMaxima(ims, maxima, max_vals, neighborhoodSize, scaleNeighborhoodSize);
    output(maxima);
    output(max_vals);
}

template<class FloatT, class IdxT>
void Boxxer2D_IFace<FloatT,IdxT>::objFilterLoG()
{
    // fimage = obj.filterLoG(image,sigma,sigmaRatio)
    // Image stack filter with single sigma, using LoG method
    //
    // [in] image: Stack of imsize shaped frames, last dimension is time
    // [in] sigma: Col vector size:[n,1] of sigma size to filter
    // [out] fimage: Stack of imsize shaped filtered frames
    checkNumArgs(1,2);
    auto ims = getCube<FloatT>();
    auto sigma = getVec<FloatT>();
    auto fims = makeOutputArray<FloatT>(ims.n_rows, ims.n_cols, ims.n_slices);
    BoxxerT::filterLoG(ims,fims,sigma);
}

template<class FloatT, class IdxT>
void Boxxer2D_IFace<FloatT,IdxT>::objFilterDoG()
{
    // fimage = obj.filterDoG(image,sigma,sigmaRatio)
    // Image stack filter with single sigma, using DoG method
    //
    // [in] image: Stack of imsize shaped frames, last dimension is time
    // [in] sigma: Col vector size:[n,1] of sigma size to filter
    // [in] sigmaRatio: scalar giving the ratio of sigmas in the DoG method:
    // [out] fimage: Stack of imsize shaped filtered frames
    checkNumArgs(1,3);
    auto ims = getCube<FloatT>();
    auto sigma = getVec<FloatT>();
    auto sigma_ratio = getAsFloat<FloatT>();
    auto fims = makeOutputArray<FloatT>(ims.n_rows, ims.n_cols, ims.n_slices);
    BoxxerT::filterDoG(ims, fims, sigma, sigma_ratio);
}

template<class FloatT, class IdxT>
void Boxxer2D_IFace<FloatT,IdxT>::objFilterGauss()
{
    // fimage = obj.filterGauss(image,sigma)
    // Image stack filter with single sigma, using single Gaussian method
    //
    // [in] image: Stack of imsize shaped frames, last dimension is time
    // [in] sigma: Col vector size:[n,1] of sigma size to filter
    // [out] fimage: Stack of imsize shaped filtered frames
    checkNumArgs(1,2);
    auto ims = getCube<FloatT>();
    auto sigma = getVec<FloatT>();
    auto fims = makeOutputArray<FloatT>(ims.n_rows, ims.n_cols, ims.n_slices);
    BoxxerT::filterGauss(ims,fims,sigma);
}

template<class FloatT, class IdxT>
void Boxxer2D_IFace<FloatT,IdxT>::objEnumerateImageMaxima()
{
    // [maxima,max_vals] = obj.enumerateImageMaxima(image,neighborhoodSize)
    // Enumerate the local maxima in the image where a maxima is the largest pixel value in it's neighborhoodSize neighborhood in each dimension.
    //
    // [in] image: Stack of imsize shaped frames, last dimension is time
    // [in] neighborhoodSize: Odd integer.  Acceptable values are in ValidMaximaNeighborhoodSizes.  (default=3)
    // [out] maxima: matrix type IdxT size:[dim+1, N]. List of maxima where rows are X, Y, ..., T and columns are different maxima detected.
    // [out] max_vals; type FloatT size:[N], vector of values at each maxima.
    checkNumArgs(2,2);
    auto ims = getCube<FloatT>();
    auto neighborhoodSize = getAsUnsigned<IdxT>();
    IMatT maxima;
    VecT max_vals;
    BoxxerT::enumerateImageMaxima(ims, maxima, max_vals, neighborhoodSize);
    output(maxima);
    output(max_vals);
}

#endif /* BOXXER_BOXXER2D_IFACE */
