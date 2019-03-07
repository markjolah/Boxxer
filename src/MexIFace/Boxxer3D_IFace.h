/** @file Boxxer3D_IFace.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief Boxxer3D MexIFace
 */

#ifndef BOXXER_BOXXER3D_IFACE
#define BOXXER_BOXXER3D_IFACE

#include<functional>

#include "MexIFace/MexIFace.h"
#include "Boxxer/Boxxer3D.h"

using namespace boxxer;

template<class FloatT, class IdxT>
class Boxxer3D_IFace : public mexiface::MexIFace,
                       public mexiface::MexIFaceHandler<Boxxer3D<FloatT,IdxT>>
{
public:
    Boxxer3D_IFace();

private:
    using BoxxerT = Boxxer3D<FloatT,IdxT>;
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
Boxxer3D_IFace<FloatT,IdxT>::Boxxer3D_IFace()
{
    methodmap["setDoGSigmaRatio"] = std::bind(&Boxxer3D_IFace::objSetDoGSigmaRatio, this);
    methodmap["filterScaledLoG"] = std::bind(&Boxxer3D_IFace::objFilterScaledLoG, this);
    methodmap["filterScaledDoG"] = std::bind(&Boxxer3D_IFace::objFilterScaledDoG, this);
    methodmap["scaleSpaceLoGMaxima"] = std::bind(&Boxxer3D_IFace::objScaleSpaceLoGMaxima, this);
    methodmap["scaleSpaceDoGMaxima"] = std::bind(&Boxxer3D_IFace::objScaleSpaceDoGMaxima, this);

    staticmethodmap["filterLoG"] = std::bind(&Boxxer3D_IFace::objFilterLoG, this);
    staticmethodmap["filterDoG"] = std::bind(&Boxxer3D_IFace::objFilterDoG, this);
    staticmethodmap["filterGauss"] = std::bind(&Boxxer3D_IFace::objFilterGauss, this);
    staticmethodmap["enumerateImageMaxima"] = std::bind(&Boxxer3D_IFace::objEnumerateImageMaxima, this);
}

template<class FloatT, class IdxT>
void Boxxer3D_IFace<FloatT,IdxT>::objConstruct()
{
    // [in] imsize - size:[3] type IdxT. Image size [X Y]
    // [in] sigma - size:[3, nScales] type FloatT.  scale-space sigmas Each column is a scale.
    // [out] handle - A new MexIFace object handle
    checkNumArgs(1,2);
    auto imsize = getVec<IdxT>();
    auto sigma = getMat<FloatT>();
    this->outputHandle(new BoxxerT(imsize,sigma));
}

template<class FloatT, class IdxT>
void Boxxer3D_IFace<FloatT,IdxT>::objSetDoGSigmaRatio()
{
    // [in] sigma_ratio: a new sigma_ratio>1
    checkNumArgs(0,1);
    obj->setDoGSigmaRatio(getAsFloat<FloatT>());
}

template<class FloatT, class IdxT>
void Boxxer3D_IFace<FloatT,IdxT>::objFilterScaledLoG()
{
    // [in] image: imsize shaped frame Size 3D: [x y z]
    // [out] fimage: Stack of imsize x nScales filtered frames. Size 4D: [x y z S]
    checkNumArgs(1,1);
    auto im = getCube<FloatT>();
    auto fims = makeOutputArray<FloatT>(ims.n_rows, ims.n_cols, ims.n_slices, obj->nScales);
    obj->filterScaledLoG(im,fims);
}

template<class FloatT, class IdxT>
void Boxxer3D_IFace<FloatT,IdxT>::objFilterScaledDoG()
{
    // [in] image: imsize shaped frame Size 3D: [x y z]
    // [out] fimage: Stack of imsize x nScales filtered frames. Size 4D: [x y z S]
    checkNumArgs(1,1);
    auto im = getCube<FloatT>();
    auto fims = makeOutputArray<FloatT>(ims.n_rows, ims.n_cols, ims.n_slices, obj->nScales);
    obj->filterScaledDoG(im,fims);
}

template<class FloatT, class IdxT>
void Boxxer3D_IFace<FloatT,IdxT>::objScaleSpaceLoGMaxima()
{
    // [in] image: Stack of imsize shaped frames, last dimension is time
    // [in] neighborhoodSize: Odd integer.  Acceptable values are in ValidMaximaNeighborhoodSizes.  (default=3)
    // [in] scaleNeighborhoodSize: Odd integer.  Acceptable values are in ValidMaximaNeighborhoodSizes.  (default=3)
    // [out] maxima: matrix type IdxT size:[dim+1, N]. List of maxima where rows are X, Y, ..., T and columns are different maxima detected.
    // [out] max_vals; type FloatT size:[N], vector of values at each maxima.
    checkNumArgs(2,3);
    auto ims = getHypercube<FloatT>();
    auto neighborhoodSize = getAsUnsigned<IdxT>();
    auto scaleNeighborhoodSize = getAsUnsigned<IdxT>();
    IMatT maxima;
    VecT max_vals;
    obj->scaleSpaceLoGMaxima(ims, maxima, max_vals, neighborhoodSize, scaleNeighborhoodSize);
    output(maxima);
    output(max_vals);
}

template<class FloatT, class IdxT>
void Boxxer3D_IFace<FloatT,IdxT>::objScaleSpaceDoGMaxima()
{
    // [in] image: Stack of imsize shaped frames, last dimension is time
    // [in] neighborhoodSize: Odd integer.  Acceptable values are in ValidMaximaNeighborhoodSizes.  (default=3)
    // [in] scaleNeighborhoodSize: Odd integer.  Acceptable values are in ValidMaximaNeighborhoodSizes.  (default=3)
    // [out] maxima: matrix type IdxT size:[dim+1, N]. List of maxima where rows are X, Y, ..., T and columns are different maxima detected.
    // [out] max_vals; type FloatT size:[N], vector of values at each maxima.
    checkNumArgs(2,3);
    auto ims = getHypercube<FloatT>();
    auto neighborhoodSize = getAsUnsigned<IdxT>();
    auto scaleNeighborhoodSize = getAsUnsigned<IdxT>();
    IMatT maxima;
    VecT max_vals;
    obj->scaleSpaceDoGMaxima(ims, maxima, max_vals, neighborhoodSize, scaleNeighborhoodSize);
    output(maxima);
    output(max_vals);
}

template<class FloatT, class IdxT>
void Boxxer3D_IFace<FloatT,IdxT>::objFilterLoG()
{
    // fimage = obj.filterLoG(image,sigma,sigmaRatio)
    // Image stack filter with single sigma, using LoG method
    //
    // [in] image: Stack of imsize shaped frames, last dimension is time
    // [in] sigma: Col vector size:[n,1] of sigma size to filter
    // [out] fimage: Stack of imsize shaped filtered frames
    checkNumArgs(1,2);
    auto ims = getHypercube<FloatT>();
    auto sigma = getVec<FloatT>();
    auto fims = makeOutputArray(ims.sX, ims.sY, ims.sZ, ims.sN);
    BoxxerT::filterLoG(ims,fims,sigma);
}

template<class FloatT, class IdxT>
void Boxxer3D_IFace<FloatT,IdxT>::objFilterDoG()
{
    // fimage = obj.filterDoG(image,sigma,sigmaRatio)
    // Image stack filter with single sigma, using DoG method
    //
    // [in] image: Stack of imsize shaped frames, last dimension is time
    // [in] sigma: Col vector size:[n,1] of sigma size to filter
    // [in] sigmaRatio: scalar giving the ratio of sigmas in the DoG method:
    // [out] fimage: Stack of imsize shaped filtered frames
    checkNumArgs(1,3);
    auto ims = getHypercube<FloatT>();
    auto sigma = getVec<FloatT>();
    auto sigma_ratio=getAsFloat<FloatT>();
    auto fims = makeOutputArray(ims.sX, ims.sY, ims.sZ, ims.sN);
    BoxxerT::filterDoG(ims, fims, sigma, sigma_ratio);
}

template<class FloatT, class IdxT>
void Boxxer3D_IFace<FloatT,IdxT>::objFilterGauss()
{
    // fimage = obj.filterGauss(image,sigma)
    // Image stack filter with single sigma, using single Gaussian method
    //
    // [in] image: Stack of imsize shaped frames, last dimension is time
    // [in] sigma: Col vector size:[n,1] of sigma size to filter
    // [out] fimage: Stack of imsize shaped filtered frames
    checkNumArgs(1,2);
    auto ims = getHypercube<FloatT>();
    auto sigma = getVec<FloatT>();
    auto fims = makeOutputArray(ims.sX, ims.sY, ims.sZ, ims.sN);
    BoxxerT::filterGauss(ims, fims, sigma);
}

template<class FloatT, class IdxT>
void Boxxer3D_IFace<FloatT,IdxT>::objEnumerateImageMaxima()
{
    // [maxima,max_vals] = obj.enumerateImageMaxima(image,neighborhoodSize)
    // Enumerate the local maxima in the image where a maxima is the largest pixel value in it's neighborhoodSize neighborhood in each dimension.
    //
    // [in] image: Stack of imsize shaped frames, last dimension is time
    // [in] neighborhoodSize: Odd integer.  Acceptable values are in ValidMaximaNeighborhoodSizes.  (default=3)
    // [out] maxima: matrix type IdxT size:[dim+1, N]. List of maxima where rows are X, Y, ..., T and columns are different maxima detected.
    // [out] max_vals; type FloatT size:[N], vector of values at each maxima.
    checkNumArgs(2,2);
    auto ims = getHypercube<FloatT>();
    auto neighborhoodSize = getAsUnsigned<IdxT>();
    IMatT maxima;
    VecT max_vals;
    BoxxerT::enumerateImageMaxima(ims, maxima, max_vals, neighborhoodSize);
    output(maxima);
    output(max_vals);
}

#endif /* BOXXER_BOXXER3D_IFACE */
