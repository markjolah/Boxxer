/** @file FilterKernels.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The boxxer::kernels namespace - low-level Gaussian finite-impulse response filters.
 *
 */

#include "Boxxer/BoxxerError.h"
#include "Boxxer/FilterKernels.h"

namespace boxxer {

namespace kernels {

template <class FloatT, class IntT>
void gaussFIR_1D(IntT size, const FloatT data[], FloatT fdata[], IntT hw, const FloatT kernel[])
{
    if(size<=2*hw+1) return gaussFIR_1D_small(size, data, fdata, hw, kernel);
    IntT x=0; //x will be guided along by several for loops
    for(; x<hw; x++) { //Mirroring boundary conditions
        FloatT val=kernel[0]*data[x];
        for(IntT r=1; r<=x; r++) val+=kernel[r]*(data[x-r]+data[x+r]);
        for(IntT r=x+1; r<=hw; r++) val+=kernel[r]*(data[x+r]+data[r-x-1]); //mirroring boundary conditions
        fdata[x]=val;
    }
//     for(; x+4<size-hw; x+=4) { //Main Loop unrolled 4x
//         FloatT val0=kernel[0]*data[x];
//         FloatT val1=kernel[0]*data[x+1];
//         FloatT val2=kernel[0]*data[x+2];
//         FloatT val3=kernel[0]*data[x+3];
//         for(IntT r=1; r<=hw; r++){
//             val0+=kernel[r]*(data[x-r]+data[x+r]);
//             val1+=kernel[r]*(data[x+1-r]+data[x+1+r]);
//             val2+=kernel[r]*(data[x+2-r]+data[x+2+r]);
//             val3+=kernel[r]*(data[x+3-r]+data[x+3+r]);
//         }
//         fdata[x]=val0;
//         fdata[x+1]=val1;
//         fdata[x+2]=val2;
//         fdata[x+3]=val3;
//     }
    for(; x<size-hw; x++) { //Main Loop
        FloatT val=kernel[0]*data[x];
        for(IntT r=1; r<=hw; r++) val+=kernel[r]*(data[x-r]+data[x+r]);
        fdata[x]=val;
    }
    for(; x<size; x++) { //column end pixels Mirroring boundary conditions
        FloatT val=kernel[0]*data[x];
        for(IntT r=1; r<=size-x-1; r++) val+=kernel[r]*(data[x-r]+data[x+r]);
        for(IntT r=size-x; r<=hw; r++) val+=kernel[r]*(data[x-r]+data[2*size-r-x-1]); //mirroring boundary conditions
        fdata[x]=val;
    }
}

template <class FloatT, class IntT>
void gaussFIR_1D_small(IntT size, const FloatT data[], FloatT fdata[], IntT hw, const FloatT kernel[])
{
    for(IntT x=0; x<size; x++) {
        FloatT val = 0.0;
        for(IntT r=-hw; r<=hw; r++) {
            if(x+r<-size || x+r>=2*size) continue; //This is beyond mirroring boundary conditions
            if(x+r<0) val += kernel[abs(r)]*data[-x-r-1];
            else if(x+r>=size) val += kernel[abs(r)]*data[2*size-r-x-1];
            else val += kernel[abs(r)]*data[x+r];
        }
        fdata[x] = val;
    }
}

template <class FloatT, class IntT>
void gaussFIR_1D_arma(const arma::Col<FloatT> &data, arma::Col<FloatT> &fdata, const arma::Col<FloatT> &kernel)
{
    IntT hw=static_cast<IntT>(kernel.n_elem)-1;
    IntT size=static_cast<IntT>(data.n_elem);
    if(size<=2*hw+1) return gaussFIR_1D_small(size, data.memptr(), fdata.memptr(), hw, kernel.memptr()); 
    for(IntT x=0; x<hw; x++) {
        FloatT val=kernel(0)*data(x);
        for(IntT r=1; r<=x; r++) val+=kernel(r)*(data(x-r)+data(x+r));
        for(IntT r=x+1; r<=hw; r++) val+=kernel(r)*(data(x+r)+data(r-x-1)); //mirroring boundary conditions
        fdata(x)=val;
    }
    for(IntT x=hw; x<size-hw; x++) { //Main Loop
        FloatT val=kernel(0)*data(x);
        for(IntT r=1; r<=hw; r++) val+=kernel(r)*(data(x-r)+data(x+r));
        fdata(x)=val;
    }
    for(IntT x=size-hw; x<size; x++) {
        FloatT val=kernel(0)*data(x);
        for(IntT r=1; r<=size-x-1; r++) val+=kernel(r)*(data(x-r)+data(x+r));
        for(IntT r=size-x; r<=hw; r++) val+=kernel(r)*(data(x-r)+data(2*size-r-x-1)); //mirroring boundary conditions
        fdata(x)=val;
    }
}

template <class FloatT, class IntT>
void gaussFIR_1D_inplace_arma(arma::Col<FloatT> &data, const arma::Col<FloatT> &kernel)
{
    IntT hw=static_cast<IntT>(kernel.n_elem)-1;
    IntT size=static_cast<IntT>(data.n_elem);
    if(!(size>=2*hw+1)){
        std::ostringstream msg;
        msg<<"Size: "<<size<<" is too small for hw: "<<hw;
        throw LogicalError(msg.str());
    }
    arma::Mat<FloatT> buf(hw+1,hw);
    buf.zeros();
    for(IntT x=0; x<hw; x++) for(IntT j=0; j<=hw; j++) buf(j,x)=kernel(j)*data(x); //Initialize buf
    for(IntT x=0; x<hw; x++) { //Compute initial hw elements using mirror boundary conditions
        FloatT val=kernel(0)*data(x);
        IntT r=1;
        for(; 0<=x-r && x+r<hw; r++) val+=buf(r,x+r)+buf(r,x-r); //x+r and x-r are in bounds and in buf
        for(; 0<=x-r; r++) val+=kernel(r)*data(x+r)+buf(r,x-r); //x-r in bounds and in buf, but data(x+r) unmodified and not in buf
        for(; x+r<hw; r++) val+=buf(r,x+r)+buf(r,r-x-1); //x-r OOB (mirroring boundary conditions), x+r in buf
        for(; r<=hw;  r++) val+=kernel(r)*data(x+r)+buf(r,r-x-1); //x-r OOB (mirroring boundary conditions), x+r unmodified and out of buf
        data(x)=val;
    }
    for(IntT x=hw; x<2*hw; x++){ //Modify x=hw..2*hw-1 to meet initial conditions of main loop
        FloatT val=0;
        for(IntT r=x-hw+1; r<=hw; r++) val+=buf(r,(x-r)%hw); //compute initial state of data(x)
        for(IntT r=0; r<=hw; r++) buf(r, x%hw)=kernel(r)*data(x); //Save new buf row for x
        data(x)=val; //Update data(x) with initial state.
    }
    for(IntT x=hw; x<size-hw; x++){ // Main loop
        //data(z) for z<x already computed
        //data(z) for z=x is current goal.  Already data(x)= sum_{j=-r}^{-1} orig(x+j)k(j)
        //data(z) for z=x+1..x+hw-1 are modified but need the current terms added.
        //data(x) for x=x+hw is unmodified and need to have its multiples with kernel stored in buf(x+hw%hw), and should itself be K(hw)*orig(x+hw).
        IntT x_idx=x%hw; //Where multiples of orig(x) are stored in buf
        FloatT x_hw_val=buf(hw,x_idx); //This will be value of data(x+hw) after iteration.  Save it now.
        FloatT x_val=buf(0,x_idx);
        for(IntT z=x+1; z<x+hw; z++) data(z)+=buf(z-x, x_idx); //update x=x+1..x+hw-1 w/ data(z)+=kernel(z-x)*orig(x);
        for(IntT j=0; j<=hw; j++) buf(j,x_idx)=kernel(j)*data(x+hw); //Update buf for x+hw (this will overwrite the saved data for orig(x)
        data(x+hw)=x_hw_val; //Overwrite data(x+hw) with kernel(r)*data(x)
        for(IntT j=1; j<=hw; j++) x_val+=buf(j,(x+j)%hw); //Finish computation of data(x)
        data(x)+=x_val;  //Add sum of rest of terms in x_val to data(x)
    }
    for(IntT x=size-hw; x<size; x++){ //Compute final hw entries.
        IntT x_idx=x%hw; //Where multiples of orig(x) are stored in buf
        FloatT x_val=buf(0,x_idx);
        IntT r=1;
        for(; x+r<size; r++) x_val+=buf(r,(x+r)%hw); //Finish computation of data(x), x+r is in bounds
        for(; r<=hw; r++) x_val+=buf(r,(2*size-r-x-1)%hw); //Finish computation of data(x), x+r is OOB
        for(IntT z=x+1;z<size;z++) data(z)+=buf(z-x, x_idx); //Update z=x+1..size-1 w/ data(z)+=kernel(z-x)*orig(x);
        data(x)+=x_val;  //Add sum of rest of terms in x_val to data(x)
    }
}

template <class FloatT, class IntT>
void gaussFIR_1D_inplace(IntT size, FloatT data[], IntT hw, const FloatT kernel[])
{
    //observed similar to safe version with -O2 -O3 makes it slower
    //about 3x slower than 2-vector version when optimized. Time is taken in central loop.
    if(!(size>=2*hw+1)){
        std::ostringstream msg;
        msg<<"Size: "<<size<<" is too small for hw: "<<hw;
        throw LogicalError(msg.str());
    }
    arma::Mat<FloatT> buf_vec(hw+1,hw);
    buf_vec.zeros();
    FloatT *buf=buf_vec.memptr();
    IntT nr=hw+1; // number of rows
    for(IntT x=0; x<hw; x++) for(IntT j=0; j<=hw; j++) buf[j+nr*x]=kernel[j]*data[x]; //Initialize buf
    for(IntT x=0; x<hw; x++) { //Compute initial hw elements using mirror boundary conditions
        FloatT val=kernel[0]*data[x];
        IntT r=1;
        for(; 0<=x-r && x+r<hw; r++) val+=buf[r+nr*(x+r)]+buf[r+nr*(x-r)]; //x+r and x-r are in bounds and in buf
        for(; 0<=x-r; r++) val+=kernel[r]*data[x+r]+buf[r+nr*(x-r)]; //x-r in bounds and in buf, but data(x+r) unmodified and not in buf
        for(; x+r<hw; r++) val+=buf[r+nr*(x+r)]+buf[r+nr*(r-x-1)]; //x-r OOB (mirroring boundary conditions), x+r in buf
        for(; r<=hw;  r++) val+=kernel[r]*data[x+r]+buf[r+nr*(r-x-1)]; //x-r OOB (mirroring boundary conditions), x+r unmodified and out of buf
        data[x]=val;
    }
    for(IntT x=hw; x<2*hw; x++){ //Modify x=hw..2*hw-1 to meet initial conditions of main loop
        FloatT val=0;
        for(IntT r=x-hw+1; r<=hw; r++) val+=buf[r+nr*((x-r)%hw)]; //compute initial state of data(x)
        for(IntT r=0; r<=hw; r++) buf[r+nr*(x%hw)]=kernel[r]*data[x]; //Save new buf row for x
        data[x]=val; //Update data(x) with initial state.
    }
    for(IntT x=hw; x<size-hw; x++){ // Main loop
        //data(z) for z<x already computed
        //data(z) for z=x is current goal.  Already data(x)= sum_{j=-r}^{-1} orig(x+j)k(j)
        //data(z) for z=x+1..x+hw-1 are modified but need the current terms added.
        //data(x) for x=x+hw is unmodified and need to have its multiples with kernel stored in buf(x+hw%hw), and should itself be K(hw)*orig(x+hw).
        IntT x_idx=x%hw; //Where multiples of orig(x) are stored in buf
        FloatT x_hw_val=buf[hw+nr*x_idx]; //This will be value of data(x+hw) after iteration.  Save it now.
        FloatT x_val=buf[0+nr*x_idx];
        for(IntT z=x+1; z<x+hw; z++) data[z]+=buf[(z-x) + nr*x_idx]; //update x=x+1..x+hw-1 w/ data(z)+=kernel(z-x)*orig(x);
        for(IntT j=0; j<=hw; j++) buf[j+nr*x_idx]=kernel[j]*data[x+hw]; //Update buf for x+hw (this will overwrite the saved data for orig(x)
        data[x+hw]=x_hw_val; //Overwrite data(x+hw) with kernel(r)*data(x)
        for(IntT j=1; j<=hw; j++) x_val+=buf[j+nr*((x+j)%hw)]; //Finish computation of data(x)
        data[x]+=x_val;  //Add sum of rest of terms in x_val to data(x)
    }
    for(IntT x=size-hw; x<size; x++){ //Compute final hw entries.
        IntT x_idx=x%hw; //Where multiples of orig(x) are stored in buf
        FloatT x_val=buf[0+nr*x_idx];
        IntT r=1;
        for(; x+r<size; r++) x_val+=buf[r+nr*((x+r)%hw)]; //Finish computation of data(x), x+r is in bounds
        for(; r<=hw; r++) x_val+=buf[r+nr*((2*size-r-x-1)%hw)]; //Finish computation of data(x), x+r is OOB
        for(IntT z=x+1;z<size;z++) data[z]+=buf[(z-x) + nr*x_idx]; //Update z=x+1..size-1 w/ data(z)+=kernel(z-x)*orig(x);
        data[x]+=x_val;  //Add sum of rest of terms in x_val to data(x)
    }
}

template <class FloatT, class IntT>
void gaussFIR_2Dx(const arma::Mat<FloatT> &data_vec, arma::Mat<FloatT> &fdata_vec, const arma::Col<FloatT> &kernel_vec)
{
    //Filters along x direction which is down columns for arma::Mat<FloatT>
    //Use mirroring boundary conditions as they will likely give the best approximation to
    //What would be off the edge of the images.  This gives data(0,0)==data(-1,0); d(1,0)==data(-2,0);
    //This seems to be what dip_image uses.
    IntT hw=static_cast<IntT>(kernel_vec.n_elem)-1;
    IntT sizeX=static_cast<IntT>(data_vec.n_rows);
    IntT sizeY=static_cast<IntT>(data_vec.n_cols);
    if(sizeX<=2*hw+1) return gaussFIR_2Dx_small(data_vec, fdata_vec, kernel_vec);
    const FloatT *data=data_vec.memptr();
    FloatT *fdata=fdata_vec.memptr();
    const FloatT *kernel=kernel_vec.memptr();
    
    for(IntT y=0; y<sizeY; y++){
        gaussFIR_1D(sizeX,&data[y*sizeX],&fdata[y*sizeX],hw,kernel);
    }
}

template <class FloatT, class IntT>
void gaussFIR_2Dx_arma(const arma::Mat<FloatT> &data, arma::Mat<FloatT> &fdata, const arma::Col<FloatT> &kernel)
{
    //Filters along x direction which is down columns for arma::Mat<FloatT>
    //Use mirroring boundary conditions as they will likely give the best approximation to
    //What would be off the edge of the images.  This gives data(0,0)==data(-1,0); d(1,0)==data(-2,0);
    //This seems to be what dip_image uses.
    IntT hw=static_cast<IntT>(kernel.n_elem)-1;
    IntT sizeX=static_cast<IntT>(data.n_rows);
    IntT sizeY=static_cast<IntT>(data.n_cols);
    if(sizeX<=2*hw+1) return gaussFIR_2Dx_small(data, fdata, kernel);

    for(IntT y=0; y<sizeY; y++){
        for(IntT x=0; x<hw; x++) {
            FloatT val=kernel(0)*data(x,y);
            for(IntT r=1; r<=x; r++) val+=kernel(r)*(data(x-r,y)+data(x+r,y));
            for(IntT r=x+1; r<=hw; r++) val+=kernel(r)*(data(x+r,y)+data(r-x-1,y)); //mirroring boundary conditions
            fdata(x,y)=val;
        }
        for(IntT x=hw; x<sizeX-hw; x++) { //Main Loop
            FloatT val=kernel(0)*data(x,y);
            for(IntT r=1; r<=hw; r++) val+=kernel(r)*(data(x-r,y)+data(x+r,y));
            fdata(x,y)=val;
        }
        for(IntT x=sizeX-hw; x<sizeX; x++) {
            FloatT val=kernel(0)*data(x,y);
            for(IntT r=1; r<sizeX-x; r++) val+=kernel(r)*(data(x-r,y)+data(x+r,y));
            for(IntT r=sizeX-x; r<=hw; r++) val+=kernel(r)*(data(x-r,y)+data(2*sizeX-r-x-1,y)); //mirroring boundary conditions
            fdata(x,y)=val;
        }
    }
}


template <class FloatT, class IntT>
void gaussFIR_2Dx_small(const arma::Mat<FloatT> &data, arma::Mat<FloatT> &fdata, const arma::Col<FloatT> &kernel)
{
    //6x slower
    //Filters along x direction which is down columns for arma::Mat<FloatT>
    //Use mirroring boundary conditions as they will likely give the best approximation to
    //What would be off the edge of the images.  This gives data(0,0)==data(-1,0); d(1,0)==data(-2,0);
    //This seems to be what dip_image uses.
    IntT hw=static_cast<IntT>(kernel.n_elem)-1;
    IntT sizeX=static_cast<IntT>(data.n_rows);
    IntT sizeY=static_cast<IntT>(data.n_cols);
    for(IntT y=0; y<sizeY; y++) for(IntT x=0; x<sizeX; x++) {
        FloatT val=0.0;
        for(IntT r=-hw; r<=hw; r++) {
            if(x+r<-sizeX || x+r>=2*sizeX) continue; //This is beyond mirroring boundary conditions
            if(x+r<0) val+=kernel(abs(r))*data(-x-r-1,y);
            else if(x+r>=sizeX) val+=kernel(abs(r))*data(2*sizeX-r-x-1,y);
            else val+=kernel(abs(r))*data(x+r,y);
        }
        fdata(x,y)=val;
    }
}

template <class FloatT, class IntT>
void gaussFIR_2Dy_small(const arma::Mat<FloatT> &data, arma::Mat<FloatT> &fdata, const arma::Col<FloatT> &kernel)
{
    //Filters along y direction which is across rows for arma::Mat<FloatT>
    //Use mirroring boundary conditions as they will likely give the best approximation to
    //What would be off the edge of the images.  This gives data(0,0)==data(-1,0); d(1,0)==data(-2,0);
    //This seems to be what dip_image uses.
    IntT hw=static_cast<IntT>(kernel.n_elem)-1;
    IntT sizeX=static_cast<IntT>(data.n_rows);
    IntT sizeY=static_cast<IntT>(data.n_cols);
    for(IntT y=0; y<sizeY; y++) for(IntT x=0; x<sizeX; x++) {
        FloatT val=0.0;
        for(IntT r=-hw; r<=hw; r++) {
            if(y+r<-sizeY || y+r>=2*sizeY) continue;  //This is beyond mirroring boundary conditions
            if(y+r<0) val+=kernel(abs(r))*data(x,-y-r-1);
            else if(y+r>=sizeY) val+=kernel(abs(r))*data(x,2*sizeY-r-y-1);
            else val+=kernel(abs(r))*data(x,y+r);
        }
        fdata(x,y)=val;
    }
}

template <class FloatT, class IntT>
void gaussFIR_2Dy_colmajor(const arma::Mat<FloatT> &data, arma::Mat<FloatT> &fdata, const arma::Col<FloatT> &kernel)
{
    //3% faster
    //Filters along y direction which is across rows for arma::Mat<FloatT>
    //Use mirroring boundary conditions as they will likely give the best approximation to
    //What would be off the edge of the images.  This gives data(0,0)==data(0,-1); d(1,0)==data(-2,0);
    //This seems to be what dip_image uses.
    IntT hw=static_cast<IntT>(kernel.n_elem)-1;
    IntT sizeX=static_cast<IntT>(data.n_rows);
    IntT sizeY=static_cast<IntT>(data.n_cols);
    if(sizeY<=2*hw+1) return gaussFIR_2Dy_small(data, fdata, kernel);

    for(IntT y=0; y<hw; y++) {
        for(IntT x=0; x<sizeX; x++){
            FloatT val=kernel(0)*data(x,y);
            for(IntT r=1; r<=y; r++) val+=kernel(r)*(data(x,y-r)+data(x,y+r));
            for(IntT r=y+1; r<=hw; r++) val+=kernel(r)*(data(x,y+r)+data(x,r-y-1)); //mirroring boundary conditions
            fdata(x,y)=val;
        }
    }
    for(IntT y=hw; y<sizeY-hw; y++) { //Main Loop
        for(IntT x=0; x<sizeX; x++){
            FloatT val=kernel(0)*data(x,y);
            for(IntT r=1; r<=hw; r++) val+=kernel(r)*(data(x,y-r)+data(x,y+r));
            fdata(x,y)=val;
        }
    }
    for(IntT y=sizeY-hw; y<sizeY; y++) {
        for(IntT x=0; x<sizeX; x++){
            FloatT val=kernel(0)*data(x,y);
            for(IntT r=1; r<sizeY-y; r++) val+=kernel(r)*(data(x,y-r)+data(x,y+r));
            for(IntT r=sizeY-y; r<=hw; r++) val+=kernel(r)*(data(x,y-r)+data(x,2*sizeY-r-y-1)); //mirroring boundary conditions
            fdata(x,y)=val;
        }
    }
}

template <class FloatT, class IntT>
void gaussFIR_2Dy_rowmajor(const arma::Mat<FloatT> &data, arma::Mat<FloatT> &fdata, const arma::Col<FloatT> &kernel)
{
    //Filters along y direction which is across rows for arma::Mat<FloatT>
    //Use mirroring boundary conditions as they will likely give the best approximation to
    //What would be off the edge of the images.  This gives data(0,0)==data(0,-1); d(1,0)==data(-2,0);
    //This seems to be what dip_image uses.
    IntT hw=static_cast<IntT>(kernel.n_elem)-1;
    IntT sizeX=static_cast<IntT>(data.n_rows);
    IntT sizeY=static_cast<IntT>(data.n_cols);
    if(sizeY<=2*hw+1) return gaussFIR_2Dy_small(data, fdata, kernel);

    for(IntT x=0; x<sizeX; x++){
        for(IntT y=0; y<hw; y++) {
            FloatT val=kernel(0)*data(x,y);
            for(IntT r=1; r<=y; r++) val+=kernel(r)*(data(x,y-r)+data(x,y+r));
            for(IntT r=y+1; r<=hw; r++) val+=kernel(r)*(data(x,y+r)+data(x,r-y-1)); //mirroring boundary conditions
            fdata(x,y)=val;
        }
        for(IntT y=hw; y<sizeY-hw; y++) { //Main Loop
            FloatT val=kernel(0)*data(x,y);
            for(IntT r=1; r<=hw; r++) val+=kernel(r)*(data(x,y-r)+data(x,y+r));
            fdata(x,y)=val;
        }
        for(IntT y=sizeY-hw; y<sizeY; y++) {
            FloatT val=kernel(0)*data(x,y);
            for(IntT r=1; r<sizeY-y; r++) val+=kernel(r)*(data(x,y-r)+data(x,y+r));
            for(IntT r=sizeY-y; r<=hw; r++) val+=kernel(r)*(data(x,y-r)+data(x,2*sizeY-r-y-1)); //mirroring boundary conditions
            fdata(x,y)=val;
        }
    }
}

template <class FloatT, class IntT>
void gaussFIR_2Dy(IntT sizeX, IntT sizeY, const FloatT data[], FloatT fdata[], IntT hw, const FloatT kernel[])
{
    for(IntT y=0; y<hw; y++) {
        const FloatT *datacol=&data[sizeX*y];
        for(IntT x=0; x<sizeX; x++){
            FloatT val=kernel[0]*datacol[x];
            for(IntT r=1;   r<=y;  r++) val+=kernel[r]*(datacol[x+sizeX*r]+datacol[x-sizeX*r]);
            for(IntT r=y+1; r<=hw; r++) val+=kernel[r]*(datacol[x+sizeX*r]+data[x+sizeX*(r-y-1)]); //mirroring boundary conditions
            fdata[x+sizeX*y]=val;
        }
    }
    for(IntT y=hw; y<sizeY-hw; y++) { //Main Loop
        const FloatT *datacol=&data[sizeX*y];
        for(IntT x=0; x<sizeX; x++){
            FloatT val=kernel[0]*datacol[x];
            for(IntT r=1; r<=hw; r++) val+=kernel[r]*(datacol[x-sizeX*r]+datacol[x+sizeX*r]);
            fdata[x+sizeX*y]=val;
        }
    }
    for(IntT y=sizeY-hw; y<sizeY; y++) {
        const FloatT *datacol=&data[sizeX*y];
        for(IntT x=0; x<sizeX; x++){
            FloatT val=kernel[0]*datacol[x];
            for(IntT r=1; r<=sizeY-y-1; r++) val+=kernel[r]*(datacol[x-sizeX*r]+datacol[x+sizeX*r]);
            for(IntT r=sizeY-y; r<=hw; r++) val+=kernel[r]*(datacol[x-sizeX*r]+data[x+sizeX*(2*sizeY-r-y-1)]); //mirroring boundary conditions
            fdata[x+sizeX*y]=val;
        }
    }
}

template <class FloatT, class IntT>
void gaussFIR_2Dy(const arma::Mat<FloatT> &data_vec, arma::Mat<FloatT> &fdata, const arma::Col<FloatT> &kernel_vec)
{
    //3% faster
    //Filters along y direction which is across rows for arma::Mat<FloatT>
    //Use mirroring boundary conditions as they will likely give the best approximation to
    //What would be off the edge of the images.  This gives data(0,0)==data(0,-1); d(1,0)==data(-2,0);
    //This seems to be what dip_image uses.
    IntT hw=static_cast<IntT>(kernel_vec.n_elem)-1;
    IntT sizeX=static_cast<IntT>(data_vec.n_rows);
    IntT sizeY=static_cast<IntT>(data_vec.n_cols);
    if(sizeY<=2*hw+1) return gaussFIR_2Dy_small(data_vec, fdata, kernel_vec);
    const FloatT *data=data_vec.memptr();
    const FloatT *kernel=kernel_vec.memptr();

    for(IntT y=0; y<hw; y++) {
        for(IntT x=0; x<sizeX; x++){
            FloatT val=kernel[0]*data_vec(x,y);
            for(IntT r=1; r<=y; r++) val+=kernel[r]*(data_vec(x,y-r)+data_vec(x,y+r));
            for(IntT r=y+1; r<=hw; r++) val+=kernel[r]*(data_vec(x,y+r)+data_vec(x,r-y-1)); //mirroring boundary conditions
            fdata(x,y)=val;
        }
    }
    for(IntT y=hw; y<sizeY-hw; y++) { //Main Loop
        const FloatT *datacol=&data[sizeX*y];
        for(IntT x=0; x<sizeX; x++){
            FloatT val=kernel[0]*datacol[x];
            for(IntT r=1; r<=hw; r++) val+=kernel[r]*(datacol[x-sizeX*r]+datacol[x+sizeX*r]);
            fdata(x,y)=val;
        }
    }
    for(IntT y=sizeY-hw; y<sizeY; y++) {
        for(IntT x=0; x<sizeX; x++){
            FloatT val=kernel[0]*data_vec(x,y);
            for(IntT r=1; r<sizeY-y; r++) val+=kernel[r]*(data_vec(x,y-r)+data_vec(x,y+r));
            for(IntT r=sizeY-y; r<=hw; r++) val+=kernel[r]*(data_vec(x,y-r)+data_vec(x,2*sizeY-r-y-1)); //mirroring boundary conditions
            fdata(x,y)=val;
        }
    }
}

//3D filters
template <class FloatT, class IntT>
void gaussFIR_3Dx_small(const arma::Cube<FloatT> &data, arma::Cube<FloatT> &fdata, const arma::Col<FloatT> &kernel)
{
    //Use mirroring boundary conditions.
    IntT hw=static_cast<IntT>(kernel.n_elem)-1;
    IntT sizeX=static_cast<IntT>(data.n_rows);
    IntT sizeY=static_cast<IntT>(data.n_cols);
    IntT sizeZ=static_cast<IntT>(data.n_slices);
    for(IntT z=0; z<sizeZ; z++) for(IntT y=0; y<sizeY; y++) for(IntT x=0; x<sizeX; x++) {
        FloatT val=0.0;
        for(IntT r=-hw; r<=hw; r++) {
            if(x+r<-sizeX || x+r>=2*sizeX) continue; //This is beyond mirroring boundary conditions
            if(x+r<0) val+=kernel(abs(r))*data(-x-r-1,y,z);
            else if(x+r>=sizeX) val+=kernel(abs(r))*data(2*sizeX-r-x-1,y,z);
            else val+=kernel(abs(r))*data(x+r,y,z);
        }
        fdata(x,y,z)=val;
    }
}

template <class FloatT, class IntT>
void gaussFIR_3Dx(const arma::Cube<FloatT> &data_vec, arma::Cube<FloatT> &fdata_vec, const arma::Col<FloatT> &kernel_vec)
{
    //Use mirroring boundary conditions
    IntT hw=static_cast<IntT>(kernel_vec.n_elem)-1;
    IntT sizeX=static_cast<IntT>(data_vec.n_rows);
    IntT sizeY=static_cast<IntT>(data_vec.n_cols);
    IntT sizeZ=static_cast<IntT>(data_vec.n_slices);
    if(sizeX<=2*hw+1) return gaussFIR_3Dx_small(data_vec, fdata_vec, kernel_vec);
    const FloatT *data=data_vec.memptr();
    FloatT *fdata=fdata_vec.memptr();
    const FloatT *kernel=kernel_vec.memptr();
    for(IntT z=0; z<sizeZ; z++)  for(IntT y=0; y<sizeY; y++) {
        gaussFIR_1D(sizeX,&data[sizeX*(y+z*sizeY)],&fdata[sizeX*(y+z*sizeY)],hw,kernel);
    }
}

template <class FloatT, class IntT>
void gaussFIR_3Dy_small(const arma::Cube<FloatT> &data, arma::Cube<FloatT> &fdata, const arma::Col<FloatT> &kernel)
{
    //Use mirroring boundary conditions.
    IntT hw=static_cast<IntT>(kernel.n_elem)-1;
    IntT sizeX=static_cast<IntT>(data.n_rows);
    IntT sizeY=static_cast<IntT>(data.n_cols);
    IntT sizeZ=static_cast<IntT>(data.n_slices);
    for(IntT z=0; z<sizeZ; z++) for(IntT y=0; y<sizeY; y++) for(IntT x=0; x<sizeX; x++) {
        FloatT val=0.0;
        for(IntT r=-hw; r<=hw; r++) {
            if(y+r<-sizeY || y+r>=2*sizeY) continue; //This is beyond mirroring boundary conditions
            if(y+r<0) val+=kernel(abs(r))*data(x,-y-r-1,z);
            else if(y+r>=sizeY) val+=kernel(abs(r))*data(x,2*sizeY-r-y-1,z);
            else val+=kernel(abs(r))*data(x,y+r,z);
        }
        fdata(x,y,z)=val;
    }
}

template <class FloatT, class IntT>
void gaussFIR_3Dy(const arma::Cube<FloatT> &data_vec, arma::Cube<FloatT> &fdata_vec, const arma::Col<FloatT> &kernel_vec)
{
    IntT hw=static_cast<IntT>(kernel_vec.n_elem)-1;
    IntT sizeX=static_cast<IntT>(data_vec.n_rows);
    IntT sizeY=static_cast<IntT>(data_vec.n_cols);
    IntT sizeZ=static_cast<IntT>(data_vec.n_slices);
    if(sizeY<=2*hw+1) return gaussFIR_3Dy_small(data_vec, fdata_vec, kernel_vec);
    const FloatT *data=data_vec.memptr();
    FloatT *fdata=fdata_vec.memptr();
    const FloatT *kernel=kernel_vec.memptr();
    
    IntT sizeXY=sizeX*sizeY;
    for(IntT z=0; z<sizeZ; z++){
        gaussFIR_2Dy(sizeX,sizeY,data,fdata,hw,kernel);
        data+=sizeXY;
        fdata+=sizeXY;
    }
}

template <class FloatT, class IntT>
void gaussFIR_3Dz_small(const arma::Cube<FloatT> &data, arma::Cube<FloatT> &fdata, const arma::Col<FloatT> &kernel)
{
    //Use mirroring boundary conditions.
    IntT hw=static_cast<IntT>(kernel.n_elem)-1;
    IntT sizeX=static_cast<IntT>(data.n_rows);
    IntT sizeY=static_cast<IntT>(data.n_cols);
    IntT sizeZ=static_cast<IntT>(data.n_slices);
    for(IntT z=0; z<sizeZ; z++) for(IntT y=0; y<sizeY; y++) for(IntT x=0; x<sizeX; x++) {
        FloatT val=0.0;
        for(IntT r=-hw; r<=hw; r++) {
            if(z+r<-sizeZ || z+r>=2*sizeZ) continue; //This is beyond mirroring boundary conditions
            if(z+r<0) val+=kernel(abs(r))*data(x,y,-z-r-1);
            else if(z+r>=sizeZ) val+=kernel(abs(r))*data(x,y,2*sizeZ-r-z-1);
            else val+=kernel(abs(r))*data(x,y,z+r);
        }
        fdata(x,y,z)=val;
    }
}

template <class FloatT, class IntT>
void gaussFIR_3Dz(const arma::Cube<FloatT> &data_vec, arma::Cube<FloatT> &fdata, const arma::Col<FloatT> &kernel_vec)
{
    IntT hw=static_cast<IntT>(kernel_vec.n_elem)-1;
    IntT sizeX=static_cast<IntT>(data_vec.n_rows);
    IntT sizeY=static_cast<IntT>(data_vec.n_cols);
    IntT sizeZ=static_cast<IntT>(data_vec.n_slices);
    if(sizeZ<=2*hw+1) return gaussFIR_3Dz_small(data_vec, fdata, kernel_vec);
    const FloatT *data=data_vec.memptr();
    const FloatT *kernel=kernel_vec.memptr();
    IntT sizeXY=sizeX*sizeY;
    for(IntT y=0; y<sizeY; y++) for(IntT x=0; x<sizeX; x++){
        const FloatT *dataslice=&data[x+sizeX*y];
        for(IntT z=0; z<hw; z++){
            FloatT val=kernel[0]*dataslice[sizeXY*z];
            for(IntT r=1;   r<=z;  r++) val+=kernel[r]*(dataslice[sizeXY*(z+r)]+dataslice[sizeXY*(z-r)]);
            for(IntT r=z+1; r<=hw; r++) val+=kernel[r]*(dataslice[sizeXY*(z+r)]+dataslice[sizeXY*(r-z-1)]); //mirroring boundary conditions
            fdata(x,y,z)=val;
        }
        for(IntT z=hw; z<sizeZ-hw; z++){
            FloatT val=kernel[0]*dataslice[sizeXY*z];
            for(IntT r=1; r<=hw; r++) val+=kernel[r]*(dataslice[sizeXY*(z-r)]+dataslice[sizeXY*(z+r)]);
            fdata(x,y,z)=val;
        }
        for(IntT z=sizeZ-hw; z<sizeZ; z++){
            FloatT val=kernel[0]*dataslice[sizeXY*z];
            for(IntT r=1; r<sizeZ-z; r++) val+=kernel[r]*(dataslice[sizeXY*(z-r)]+dataslice[sizeXY*(z+r)]);
            for(IntT r=sizeZ-z; r<=hw; r++) val+=kernel[r]*(dataslice[sizeXY*(z-r)]+dataslice[sizeXY*(2*sizeZ-r-z-1)]); //mirroring boundary conditions
            fdata(x,y,z)=val;
        }
    }
}


/* Explicit Template Instantiations */
/* 1D Gauss FIR Filters */
template void gaussFIR_1D<float>(const arma::Col<float> &data, arma::Col<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_1D<double>(const arma::Col<double> &data, arma::Col<double> &fdata, const arma::Col<double> &kernel);

template void gaussFIR_1D<float>(int32_t size, const float data[], float fdata[], int32_t hw, const float kernel[]);
template void gaussFIR_1D<double>(int32_t size, const double data[], double fdata[], int32_t hw, const double kernel[]);

template void gaussFIR_1D_small<float>(int32_t size, const float data[], float fdata[], int32_t hw, const float kernel[]);
template void gaussFIR_1D_small<double>(int32_t size, const double data[], double fdata[], int32_t hw, const double kernel[]);

template void gaussFIR_1D_arma<float>(const arma::Col<float> &data, arma::Col<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_1D_arma<double>(const arma::Col<double> &data, arma::Col<double> &fdata, const arma::Col<double> &kernel);

template void gaussFIR_1D_inplace_arma<float>(arma::Col<float> &data, const arma::Col<float> &kernel);
template void gaussFIR_1D_inplace_arma<double>(arma::Col<double> &data, const arma::Col<double> &kernel);

template void gaussFIR_1D_inplace<float>(int32_t size, float data[], int32_t hw, const float kernel[]);
template void gaussFIR_1D_inplace<double>(int32_t size, double data[], int32_t hw, const double kernel[]);

/* 2D Gauss FIR Filters */
template void gaussFIR_2Dx<float>(const arma::Mat<float> &data, arma::Mat<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_2Dx<double>(const arma::Mat<double> &data, arma::Mat<double> &fdata, const arma::Col<double> &kernel);

template void gaussFIR_2Dx_small<float>(const arma::Mat<float> &data, arma::Mat<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_2Dx_small<double>(const arma::Mat<double> &data, arma::Mat<double> &fdata, const arma::Col<double> &kernel);

template void gaussFIR_2Dx_arma<float>(const arma::Mat<float> &data, arma::Mat<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_2Dx_arma<double>(const arma::Mat<double> &data, arma::Mat<double> &fdata, const arma::Col<double> &kernel);

template void gaussFIR_2Dy<float>(const arma::Mat<float> &data, arma::Mat<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_2Dy<double>(const arma::Mat<double> &data, arma::Mat<double> &fdata, const arma::Col<double> &kernel);

template void gaussFIR_2Dy_rowmajor<float>(const arma::Mat<float> &data, arma::Mat<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_2Dy_rowmajor<double>(const arma::Mat<double> &data, arma::Mat<double> &fdata, const arma::Col<double> &kernel);

template void gaussFIR_2Dy_colmajor<float>(const arma::Mat<float> &data, arma::Mat<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_2Dy_colmajor<double>(const arma::Mat<double> &data, arma::Mat<double> &fdata, const arma::Col<double> &kernel);

template void gaussFIR_2Dy_small<float>(const arma::Mat<float> &data, arma::Mat<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_2Dy_small<double>(const arma::Mat<double> &data, arma::Mat<double> &fdata, const arma::Col<double> &kernel);

/* 3D Gauss FIR Filters */
template void gaussFIR_3Dx<float>(const arma::Cube<float> &data, arma::Cube<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_3Dx<double>(const arma::Cube<double> &data, arma::Cube<double> &fdata, const arma::Col<double> &kernel);

template void gaussFIR_3Dx_small<float>(const arma::Cube<float> &data, arma::Cube<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_3Dx_small<double>(const arma::Cube<double> &data, arma::Cube<double> &fdata, const arma::Col<double> &kernel);

template void gaussFIR_3Dy<float>(const arma::Cube<float> &data, arma::Cube<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_3Dy<double>(const arma::Cube<double> &data, arma::Cube<double> &fdata, const arma::Col<double> &kernel);

template void gaussFIR_3Dy_small<float>(const arma::Cube<float> &data, arma::Cube<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_3Dy_small<double>(const arma::Cube<double> &data, arma::Cube<double> &fdata, const arma::Col<double> &kernel);

template void gaussFIR_3Dz<float>(const arma::Cube<float> &data, arma::Cube<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_3Dz<double>(const arma::Cube<double> &data, arma::Cube<double> &fdata, const arma::Col<double> &kernel);

template void gaussFIR_3Dz_small<float>(const arma::Cube<float> &data, arma::Cube<float> &fdata, const arma::Col<float> &kernel);
template void gaussFIR_3Dz_small<double>(const arma::Cube<double> &data, arma::Cube<double> &fdata, const arma::Col<double> &kernel);

} /* namespace boxxer::kernels */

} /* namespace boxxer */
