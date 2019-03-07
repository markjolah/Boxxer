/**
 * @file Maxima.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief The class definitions for Maxima2D and Maxima3D, local maxima finders.
 */

#include "Boxxer/Maxima.h"

namespace boxxer {

template<class FloatT, class IdxT>
Maxima2D<FloatT, IdxT>::Maxima2D(const IVecT &size, IdxT boxsize)
    : size(size), boxsize(boxsize)
{
    assert(boxsize>=3 && boxsize%2==1);
    assert(arma::all(size>=boxsize));
    max_maxima=size(0)*size(1)/4;
    maxima.set_size(dim,max_maxima);
    max_vals.set_size(max_maxima);
    skip_buf.set_size(size(0),2);
}

template<class FloatT, class IdxT>
IdxT Maxima2D<FloatT, IdxT>::find_maxima(const ImageT &im)
{
    if(boxsize==3) return maxima_3x3(im);
    else return maxima_nxn(im,boxsize);
}

template<class FloatT, class IdxT>
inline
void Maxima2D<FloatT, IdxT>::detect_maxima(IdxT &Nmaxima, IdxT x, IdxT y, FloatT val)
{
    assert(Nmaxima<max_maxima);
    maxima(0,Nmaxima)=x;
    maxima(1,Nmaxima)=y;
    max_vals(Nmaxima)=val;
    Nmaxima++;
}

template<class FloatT, class IdxT>
IdxT Maxima2D<FloatT, IdxT>::find_maxima(const ImageT &im, IMatT &maxima_out, VecT &max_vals_out)
{
    IdxT Nmaxima=find_maxima(im);
    maxima_out.resize(dim,Nmaxima);
    max_vals_out.resize(Nmaxima);
    read_maxima(Nmaxima, maxima_out,max_vals_out);
    return Nmaxima;
}

template<class FloatT, class IdxT>
void Maxima2D<FloatT, IdxT>::read_maxima(IdxT Nmaxima, IMatT &maxima_out, VecT &max_vals_out) const
{
    assert(static_cast<IdxT>(maxima_out.n_rows)==dim && static_cast<IdxT>(maxima_out.n_cols)==Nmaxima);
    assert(static_cast<IdxT>(max_vals_out.n_elem)==Nmaxima);
    if(Nmaxima>0) {
        maxima_out=maxima.cols(0,Nmaxima-1);
        max_vals_out=max_vals.rows(0,Nmaxima-1);
    }
}

template<class FloatT, class IdxT>
void Maxima2D<FloatT, IdxT>::test_maxima(const ImageT &im)
{
    IdxT Nmaxima=maxima_3x3(im);
    IMatT maxima_out(dim,Nmaxima);
    VecT max_vals_out(Nmaxima);
    read_maxima(Nmaxima, maxima_out, max_vals_out);

    IdxT Nmaxima_slow=maxima_3x3_slow(im);
    IMatT maxima_out_slow(dim,Nmaxima_slow);
    VecT max_vals_out_slow(Nmaxima_slow);
    read_maxima(Nmaxima_slow, maxima_out_slow, max_vals_out_slow);

    if (Nmaxima!=Nmaxima_slow) prIdxTf("Nmaxima:%i  Nmaxima(Slow):%i\n",Nmaxima,Nmaxima_slow);
    for(IdxT n=0; n<std::min(Nmaxima,Nmaxima_slow); n++) {
        if(arma::any(maxima_out.col(n)!=maxima_out_slow.col(n)))
            prIdxTf("Maxima do not match: (%i, %i) != (%i, %i)\n",maxima_out(0,n), maxima_out(1,n), maxima_out_slow(0,n), maxima_out_slow(1,n));
    }
}

template<class FloatT, class IdxT>
IdxT Maxima2D<FloatT, IdxT>::maxima_3x3(const ImageT &im)
{
    IdxT Nmaxima=maxima_3x3_edges(im);
    skip_buf.zeros();
    IdxT sizeX=size(0);
    IdxT sizeY=size(1);
    IdxT *skip=skip_buf.memptr();
    IdxT *skip_next=skip+sizeX;
    for(IdxT y=1; y<sizeY-1; y++){
        for(IdxT x=1; x<sizeX-1; x++) {
            if(skip[x]) continue;
            double val=im(x,y);
            if (val<=im(x+1,y)) {//This is an increasing trend.  Follow until it ends.
                do {
                    x++; val=im(x,y);
                } while(x<sizeX-1 && val<=im(x+1,y)); //Increasing trend continues
                if(x>=sizeX-1) break;//Next pixel is bigger, so this pixel is non-max
            } else { //Next pixel is not bigger, check previous pixel
                if(val<=im(x-1,y)) continue;
            }
            assert(x+1<sizeX);
            skip[x+1]=1; //We are a 1D max so skip next pixel
            //Check next column and record any to skip
            if (val<=im(x-1,y+1)) { continue; } else { skip_next[x-1]=1; }
            if (val<=im(x,y+1))   { continue; } else { skip_next[x]=1;   }
            if (val<=im(x+1,y+1)) { continue; } else { skip_next[x+1]=1; }
            //Check previous column
            if (val<=im(x-1,y-1) || val<=im(x,y-1) || val<=im(x+1,y-1)) continue;
            //Detected maxima -- record it
            detect_maxima(Nmaxima, x, y, val);
        }
        memset(skip, 0, sizeof(IndexT)*sizeX);  //Reset skip
        std::swap(skip,skip_next);
    }
    return Nmaxima;
}

template<class FloatT, class IdxT>
IdxT Maxima2D<FloatT, IdxT>::maxima_3x3_slow(const ImageT &im)
{
    IdxT Nmaxima=maxima_3x3_edges(im);
    for(IdxT y=1; y<size(1)-1; y++) for(IdxT x=1; x<size(0)-1; x++) {
        double val=im(x,y);
        if(val<=im(x-1,y-1) || val<=im(x-1,y) || val<=im(x-1,y+1) ||
            val<=im(x,y-1) || val<=im(x,y+1) ||
            val<=im(x+1,y-1) || val<=im(x+1,y) || val<=im(x+1,y+1)) continue;
        detect_maxima(Nmaxima, x, y, val);
    }
    return Nmaxima;
}

template<class FloatT, class IdxT>
IdxT Maxima2D<FloatT, IdxT>::maxima_3x3_edges(const ImageT &im)
{
    IdxT x=0, y=0;
    IdxT Nmaxima=0;
    double val=im(x,y);
    //Top left corner x=0; y=0
    if(val>im(x,y+1) && val>im(x+1,y) && val>im(x+1,y+1)) detect_maxima(Nmaxima, x, y, val);
    //Left edge x=1 ... size(0)-2; y=0
    for(x=1; x<size(0)-1; x++) {
        val=im(x,y);
        if( val>im(x-1,y) && val>im(x+1,y) && val>im(x-1,y+1) && val>im(x,y+1) && val>im(x+1,y+1) )
            detect_maxima(Nmaxima, x, y, val);
    }
    //Bottom left corner x=size(0)-1; y=0
    val=im(x,y);
    if( val>im(x,y+1) && val>im(x-1,y) && val>im(x-1,y+1)) detect_maxima(Nmaxima, x, y, val);
    //Bottom edge x=size(0)-1; y=1 ... size(1)-2
    for(y=1; y<size(1)-1; y++) {
        val=im(x,y);
        if( val>im(x,y-1) && val>im(x,y+1) && val>im(x-1,y-1) && val>im(x-1,y) && val>im(x-1,y+1) )
            detect_maxima(Nmaxima, x, y, val);
    }
    //Bottom right corner x=size(0)-1; y=size(1)-1
    val=im(x,y);
    if(val>im(x,y-1) && val>im(x-1,y) && val>im(x-1,y-1)) detect_maxima(Nmaxima, x, y, val);
    //Right edge x=size(0)-2 ... 1; y=size(1)-1
    for(x=size(0)-2; x>=1; x--) {
        val=im(x,y);
        if( val>im(x-1,y) && val>im(x+1,y) && val>im(x-1,y-1) && val>im(x,y-1) && val>im(x+1,y-1) )
            detect_maxima(Nmaxima, x, y, val);
    }
    //Top right corner x=0; y=size(1)-1
    val=im(x,y);
    if(val>im(x,y-1) && val>im(x+1,y) && val>im(x+1,y-1)) detect_maxima(Nmaxima, x, y, val);
    //Top edge x=0; y=size(1)-2 ... 1
    for(y=size(1)-2; y>=1; y--) {
        val=im(x,y);
        if( val>im(x,y-1) && val>im(x,y+1) && val>im(x+1,y-1) && val>im(x+1,y) && val>im(x+1,y+1) )
            detect_maxima(Nmaxima, x, y, val);
    }
    return Nmaxima;
}

template<class FloatT, class IdxT>
IdxT Maxima2D<FloatT, IdxT>::maxima_5x5(const ImageT &im)
{
    IdxT Nmaxima=maxima_3x3(im);
    IMatT new_maxima(dim, Nmaxima);
    VecT new_max_vals(Nmaxima);
    IdxT new_Nmaxima=0;
    for(IdxT n=0; n<Nmaxima;n++){
        IdxT max_x=maxima(0,n);
        IdxT max_y=maxima(1,n);
        FloatT max_val=max_vals(n);
        IdxT x_upper=std::min(max_x+2,size(0));
        IdxT x_lower=std::max(max_x-2,0);
        IdxT y_upper=std::min(max_y+1,size(1));
        IdxT y_lower=std::max(max_y-1,0);
        bool ok=true;
        IdxT x,y;
        if(max_y>=2) { //check left column
            for(y=max_y-2, x=x_lower; x<x_upper; x++) if( im(x,y)>max_val) {ok=false; break;}
        } if(ok && max_x>=2) { //Check lower row
            for(x=max_x-2, y=y_lower; y<y_upper; y++) if( im(x,y)>max_val) {ok=false; break;}
        } if(ok && max_x+2<size(0)) { //Check upper row
            for(x=max_x+2, y=y_lower; y<y_upper; y++) if( im(x,y)>max_val) {ok=false; break;}
        } if(ok && max_y+2<size(1)) { //check right column
            for(y=max_y+2, x=x_lower; x<x_upper; x++) if( im(x,y)>max_val) {ok=false; break;}
        }
        if(ok){
            new_maxima.col(new_Nmaxima)=maxima.col(n);
            new_max_vals(new_Nmaxima)=max_vals(n);
            new_Nmaxima++;
        }
    }
    if(new_Nmaxima>0){
        maxima.cols(0,new_Nmaxima-1)=new_maxima.cols(0,new_Nmaxima-1);
        max_vals.rows(0,new_Nmaxima-1)=new_max_vals.rows(0,new_Nmaxima-1);
    }
    return new_Nmaxima;
}

template<class FloatT, class IdxT>
IdxT Maxima2D<FloatT, IdxT>::maxima_nxn(const ImageT &im, IdxT filter_size)
{
    IdxT Nmaxima = maxima_3x3(im);
    assert(filter_size%2==1); //filter(box) size is odd 
    assert(filter_size>3);
    IdxT k = (filter_size-1)/2;
    IMatT new_maxima(dim, Nmaxima);
    VecT new_max_vals(Nmaxima);
    IdxT new_Nmaxima = 0;
    for(IdxT n=0; n<Nmaxima;n++){
        FloatT max_val = max_vals(n);
        IdxT max_x = maxima(0,n);
        IdxT max_y = maxima(1,n);
        IdxT x_upper = std::min(max_x+k,size(0)-1);
        IdxT x_lower = std::max(max_x-k,0);
        IdxT y_upper = std::min(max_y+k,size(1)-1);
        IdxT y_lower = std::max(max_y-k,0);
        for(IdxT y=y_lower; y<=y_upper; y++) { //process each column to look for larger values
            if(max_y-1<=y && y<=max_y-1){ //middle column skip the portion already checked in the 3x3x3 core 
                for(IdxT x=x_lower; x<=max_x-2; x++) if(im(x,y)>max_val) goto maxima2D_nxn_reject;
                for(IdxT x=max_x+2; x<=x_upper; x++) if(im(x,y)>max_val) goto maxima2D_nxn_reject;
            } else { //left or right column. Process entire column
                for(IdxT x=x_lower; x<=x_upper; x++) if(im(x,y)>max_val) goto maxima2D_nxn_reject;
            } 
        }
        //OK if we made it here so record
        new_maxima.col(new_Nmaxima)=maxima.col(n);
        new_max_vals(new_Nmaxima)=max_vals(n);
        new_Nmaxima++;
maxima2D_nxn_reject: ;//Go here when local maxima is not valid
    }
    if(new_Nmaxima>0){ //Shrink down maxima, but keep maxima and max_vals the same size.
        maxima.cols(0,new_Nmaxima-1)=new_maxima.cols(0,new_Nmaxima-1);
        max_vals.rows(0,new_Nmaxima-1)=new_max_vals.rows(0,new_Nmaxima-1);
    }
    return new_Nmaxima;
}

/* Maxima3D */
template<class FloatT, class IdxT>
Maxima3D<FloatT, IdxT>::Maxima3D(const IVecT &size, IdxT boxsize)
: size(size), boxsize(boxsize)
{
    assert(static_cast<IdxT>(size.n_elem)==dim);
    assert(boxsize>=3 && boxsize%2==1);
    assert(arma::all(size>=boxsize));
    max_maxima=size(0)*size(1)*size(2)/8;
    maxima.set_size(dim,max_maxima);
    max_vals.set_size(max_maxima);
    skip_buf.set_size(size(0),2);
    skip_plane_buf.set_size(size(0),size(1),2);
}

template<class FloatT, class IdxT>
IdxT Maxima3D<FloatT, IdxT>::find_maxima(const ImageT &im)
{
    if(boxsize==3) return maxima_3x3(im);
    else return maxima_nxn(im,boxsize);
}

template<class FloatT, class IdxT>
inline
void Maxima3D<FloatT, IdxT>::detect_maxima(IdxT &Nmaxima, IdxT x, IdxT y, IdxT z, FloatT val)
{
    assert(Nmaxima<max_maxima);
    maxima(0,Nmaxima)=x;
    maxima(1,Nmaxima)=y;
    maxima(2,Nmaxima)=z;
    max_vals(Nmaxima)=val;
    Nmaxima++;
}

template<class FloatT, class IdxT>
IdxT Maxima3D<FloatT, IdxT>::find_maxima(const ImageT &im, IMatT &maxima_out, VecT &max_vals_out)
{
    IdxT Nmaxima=find_maxima(im);
    maxima_out.resize(dim,Nmaxima);
    max_vals_out.resize(Nmaxima);
    read_maxima(Nmaxima, maxima_out,max_vals_out);
    return Nmaxima;
}

template<class FloatT, class IdxT>
void Maxima3D<FloatT, IdxT>::read_maxima(IdxT Nmaxima, IMatT &maxima_out, VecT &max_vals_out) const
{
    assert(static_cast<IdxT>(maxima_out.n_rows)==dim && static_cast<IdxT>(maxima_out.n_cols)==Nmaxima);
    assert(static_cast<IdxT>(max_vals_out.n_elem)==Nmaxima);
    if(Nmaxima>0) {
        maxima_out=maxima.cols(0,Nmaxima-1);
        max_vals_out=max_vals.rows(0,Nmaxima-1);
    }
}

template<class FloatT, class IdxT>
void Maxima3D<FloatT, IdxT>::test_maxima(const ImageT &im)
{
    IdxT Nmaxima=maxima_3x3(im);
    IMatT maxima_out(dim,Nmaxima);
    VecT max_vals_out(Nmaxima);
    read_maxima(Nmaxima, maxima_out, max_vals_out);

    IdxT Nmaxima_slow=maxima_3x3_slow(im);
    IMatT maxima_out_slow(dim,Nmaxima_slow);
    VecT max_vals_out_slow(Nmaxima_slow);
    read_maxima(Nmaxima_slow, maxima_out_slow, max_vals_out_slow);

    if (Nmaxima!=Nmaxima_slow) prIdxTf("Nmaxima:%i  Nmaxima(Slow):%i\n",Nmaxima,Nmaxima_slow);
    for(IdxT n=0; n<std::min(Nmaxima,Nmaxima_slow); n++) {
        if(arma::any(maxima_out.col(n)!=maxima_out_slow.col(n)))
            prIdxTf("Maxima do not match: (%i, %i, %i) != (%i, %i, %i)\n",maxima_out(0,n), maxima_out(1,n), maxima_out(2,n),
                   maxima_out_slow(0,n), maxima_out_slow(1,n),  maxima_out_slow(2,n));
    }
}

template<class FloatT, class IdxT>
IdxT Maxima3D<FloatT, IdxT>::maxima_3x3(const ImageT &im)
{
    IdxT Nmaxima=maxima_3x3_edges(im);
    IdxT sizeX=size(0);
    IdxT sizeY=size(1);
    IdxT sizeZ=size(2);
    skip_buf.zeros();
    IdxT *skip=skip_buf.memptr();
    IdxT *skip_next=skip+sizeX;
    skip_plane_buf.zeros();
    IdxT *skip_plane=skip_plane_buf.memptr();
    IdxT *skip_plane_next=skip_plane+sizeX*sizeY;
    for(IdxT z=1; z<sizeZ-1; z++){
        for(IdxT y=1; y<sizeY-1; y++){
            for(IdxT x=1; x<sizeX-1; x++) {
                if(skip[x]) continue;
                if(skip_plane[y*sizeX+x]) continue;
                double val=im(x,y,z);
                //1D Max finding
                if (val<=im(x+1,y,z)) {//This is an increasing trend.  Follow until it ends.
                    do {
                        x++; val=im(x,y,z);
                    } while(x<sizeX-1 && val<=im(x+1,y,z)); //Increasing trend continues
                    if(x>=sizeX-1) break;//Next pixel is bigger, so this pixel is non-max
                } else { //Next pixel is not bigger, check previous pixel
                    if(val<=im(x-1,y,z)) continue;
                }
                assert(x+1<sizeX);
                skip[x+1]=1; //We are a 1D max so skip next pixel

                //Check next column and record any to skip
                if (val<=im(x-1,y+1,z)) { continue; } else { skip_next[x-1]=1; }
                if (val<=im(x,  y+1,z)) { continue; } else { skip_next[x]=1;   }
                if (val<=im(x+1,y+1,z)) { continue; } else { skip_next[x+1]=1; }
                //Check next plane and record any to skip
                if (val<=im(x-1,y-1,z+1)) { continue; } else { skip_plane_next[(y-1)*sizeX+x-1]=1; }
                if (val<=im(x,  y-1,z+1)) { continue; } else { skip_plane_next[(y-1)*sizeX+x  ]=1; }
                if (val<=im(x+1,y-1,z+1)) { continue; } else { skip_plane_next[(y-1)*sizeX+x+1]=1; }
                if (val<=im(x-1,y,  z+1)) { continue; } else { skip_plane_next[y*sizeX    +x-1]=1; }
                if (val<=im(x,  y,  z+1)) { continue; } else { skip_plane_next[y*sizeX    +x  ]=1; }
                if (val<=im(x+1,y,  z+1)) { continue; } else { skip_plane_next[y*sizeX    +x+1]=1; }
                if (val<=im(x-1,y+1,z+1)) { continue; } else { skip_plane_next[(y+1)*sizeX+x-1]=1; }
                if (val<=im(x,  y+1,z+1)) { continue; } else { skip_plane_next[(y+1)*sizeX+x  ]=1; }
                if (val<=im(x+1,y+1,z+1)) { continue; } else { skip_plane_next[(y+1)*sizeX+x+1]=1; }
                //Check previous column
                if (val<=im(x-1,y-1,z) || val<=im(x,y-1,z) || val<=im(x+1,y-1,z)) continue;
                //Check previous plane
                if (val<=im(x-1,y-1,z-1) || val<=im(x,y-1,z-1) || val<=im(x+1,y-1,z-1) ||
                    val<=im(x-1,y,  z-1) || val<=im(x,y,  z-1) || val<=im(x+1,y,  z-1) ||
                    val<=im(x-1,y+1,z-1) || val<=im(x,y+1,z-1) || val<=im(x+1,y+1,z-1)) continue;
                //Detected maxima -- record it
                detect_maxima(Nmaxima, x, y, z, val);
            }
            memset(skip, 0, sizeof(IdxT)*sizeX);  //Reset skip
            std::swap(skip,skip_next);
        }
        memset(skip_plane, 0, sizeof(IndexT)*sizeX*sizeY);  //Reset skip_plane
        std::swap(skip_plane,skip_plane_next);
    }
    return Nmaxima;
}

template<class FloatT, class IdxT>
IdxT Maxima3D<FloatT, IdxT>::maxima_3x3_slow(const ImageT &im)
{
    IdxT Nmaxima=maxima_3x3_edges(im);
    for(IdxT z=1; z<size(2)-1; z++) for(IdxT y=1; y<size(1)-1; y++) for(IdxT x=1; x<size(0)-1; x++) {
        double val=im(x,y,z);
        if(val>im(x-1,y-1,z-1) && val>im(x-1,y,z-1) && val>im(x-1,y+1,z-1) && val>im(x,y-1,z-1) && val>im(x,y,z-1) && val>im(x,y+1,z-1) && val>im(x+1,y-1,z-1) && val>im(x+1,y,z-1) && val>im(x+1,y+1,z-1) && //Plane z-1
           val>im(x-1,y-1,z) && val>im(x-1,y,z) && val>im(x-1,y+1,z) && val>im(x,y-1,z) && val>im(x,y+1,z) && val>im(x+1,y-1,z) && val>im(x+1,y,z) && val>im(x+1,y+1,z) && //Plane z
           val>im(x-1,y-1,z+1) && val>im(x-1,y,z+1) && val>im(x-1,y+1,z+1) && val>im(x,y-1,z+1) && val>im(x,y,z+1) && val>im(x,y+1,z+1) && val>im(x+1,y-1,z+1) && val>im(x+1,y,z+1) && val>im(x+1,y+1,z+1)){ //Plane z+1
            detect_maxima(Nmaxima, x, y, z, val);
        }
    }
    return Nmaxima;
}

template<class FloatT, class IdxT>
IdxT Maxima3D<FloatT, IdxT>::maxima_3x3_edges(const ImageT &im)
{
    IdxT x=0, y=0, z=0;
    IdxT Nmaxima=0;
    IdxT sizeX=size(0);
    IdxT sizeY=size(1);
    IdxT sizeZ=size(2);
//     std::cout<<"Size: "<<size<<std::endl;
//     std::cout<<"Size: ["<<sizeX<<","<<sizeY<<","<<sizeZ<<"]"<<std::endl;

    /* Forward Face (z=0) Edges and Corners */
    //Top Left Forward corner x=0; y=0; z=0
    double val=im(x,y,z);
    if(val>im(x,y+1,z) && val>im(x+1,y,z) && val>im(x+1,y+1,z) && //Plane z=0
       val>im(x,y,z+1) && val>im(x,y+1,z+1) && val>im(x+1,y,z+1) && val>im(x+1,y+1,z+1)){ //Plane z=1
        detect_maxima(Nmaxima, x, y, z, val);
    }
    //Left Forward edge x=1 ... sizeX-2; y=0; z=0
    for(x=1; x<sizeX-1; x++) {
        val=im(x,y,z);
        if( val>im(x-1,y,z) && val>im(x+1,y,z) && val>im(x-1,y+1,z) && val>im(x,y+1,z) && val>im(x+1,y+1,z) && //Plane z=0
            val>im(x,y,z+1) && val>im(x-1,y,z+1) && val>im(x+1,y,z+1) && val>im(x-1,y+1,z+1) && val>im(x,y+1,z+1) && val>im(x+1,y+1,z+1)){ //Plane z=1
            detect_maxima(Nmaxima, x, y, z, val);
        }
    }
    //Bottom left Forward corner x=sizeX-1; y=0; z=0
    val=im(x,y,z);
    if( val>im(x,y+1,z) && val>im(x-1,y,z) && val>im(x-1,y+1,z) && //Plane z=0
        val>im(x,y,z+1) && val>im(x,y+1,z+1) && val>im(x-1,y,z+1) && val>im(x-1,y+1,z+1)){//Plane z=1
        detect_maxima(Nmaxima, x, y, z, val);
    }
    //Bottom Forward edge x=sizeX-1; y=1 ... sizeY-2; z=0
    for(y=1; y<sizeY-1; y++) {
        val=im(x,y,z);
        if( val>im(x,y-1,z) && val>im(x,y+1,z) && val>im(x-1,y-1,z) && val>im(x-1,y,z) && val>im(x-1,y+1,z) &&//Plane z=0
            val>im(x,y,z+1) && val>im(x,y-1,z+1) && val>im(x,y+1,z+1) && val>im(x-1,y-1,z+1) && val>im(x-1,y,z+1) && val>im(x-1,y+1,z+1)){ //Plane z=1
            detect_maxima(Nmaxima, x, y, z, val);
        }
    }
    //Bottom Right Forward corner x=sizeX-1; y=sizeY-1; z=0
    val=im(x,y,z);
    if(val>im(x,y-1,z) && val>im(x-1,y,z) && val>im(x-1,y-1,z) &&//Plane z=0
        val>im(x,y,z+1) && val>im(x,y-1,z+1) && val>im(x-1,y,z+1) && val>im(x-1,y-1,z+1)){//Plane z=1
        detect_maxima(Nmaxima, x, y, z, val);
    }
    //Right Forward edge x=sizeX-2 ... 1; y=sizeY-1; z=0
    for(x=sizeX-2; x>=1; x--) {
        val=im(x,y,z);
        if( val>im(x-1,y,z) && val>im(x+1,y,z) && val>im(x-1,y-1,z) && val>im(x,y-1,z) && val>im(x+1,y-1,z) && //Plane z=0
            val>im(x,y,z+1) && val>im(x-1,y,z+1) && val>im(x+1,y,z+1) && val>im(x-1,y-1,z+1) && val>im(x,y-1,z+1) && val>im(x+1,y-1,z+1)){//Plane z=1
            detect_maxima(Nmaxima, x, y, z, val);
        }
    }
    //Top Right Forward corner x=0; y=sizeY-1; z=0
    val=im(x,y,z);
    if(val>im(x,y-1,z) && val>im(x+1,y,z) && val>im(x+1,y-1,z) &&//Plane z=0
        val>im(x,y,z+1) && val>im(x,y-1,z+1) && val>im(x+1,y,z+1) && val>im(x+1,y-1,z+1)){//Plane z=1
        detect_maxima(Nmaxima, x, y, z, val);
    }
    //Top Forward edge x=0; y=sizeY-2...1; z=0
    for(y=sizeY-2; y>=1; y--) {
        val=im(x,y,z);
        if( val>im(x,y-1,z) && val>im(x,y+1,z) && val>im(x+1,y-1,z) && val>im(x+1,y,z) && val>im(x+1,y+1,z) &&//Plane z=0
            val>im(x,y,z+1) && val>im(x,y-1,z+1) && val>im(x,y+1,z+1) && val>im(x+1,y-1,z+1) && val>im(x+1,y,z+1) && val>im(x+1,y+1,z+1)){//Plane z=1
            detect_maxima(Nmaxima, x, y,z, val);
        }
    }

    /* Receding Edges (z=1...sizeZ-2) Edges */
    //Top Left Receding Edge x=0; y=0; z=1 ... sizeZ-2;
    for(x=0,y=0,z=1; z<sizeZ-1; z++) {
        val=im(x,y,z);
        if( val>im(x,y,z-1) && val>im(x+1,y,z-1) && val>im(x+1,y+1,z-1) && val>im(x,y+1,z-1) && //Plane z-1
            val>im(x+1,y,z) && val>im(x+1,y+1,z) && val>im(x,y+1,z) && //Plane z
            val>im(x,y,z+1) && val>im(x+1,y,z+1) && val>im(x+1,y+1,z+1) && val>im(x,y+1,z+1)){ //Plane z+1
                detect_maxima(Nmaxima, x, y, z, val);
            }
    }
    //Bottom Left Receding Edge x=sizeX-1; y=0; z=1 ... sizeZ-2;
    for(x=sizeX-1,y=0,z=1; z<sizeZ-1; z++) {
        val=im(x,y,z);
        if( val>im(x,y,z-1) && val>im(x-1,y,z-1) && val>im(x-1,y+1,z-1) && val>im(x,y+1,z-1) && //Plane z-1
            val>im(x-1,y,z) && val>im(x-1,y+1,z) && val>im(x,y+1,z) && //Plane z
            val>im(x,y,z+1) && val>im(x-1,y,z+1) && val>im(x-1,y+1,z+1) && val>im(x,y+1,z+1)){ //Plane z+1
                detect_maxima(Nmaxima, x, y, z, val);
            }
    }
    //Bottom Right Receding Edge x=sizeX-1; y=sizeY-1; z=1 ... sizeZ-2;
    for(x=sizeX-1,y=sizeY-1,z=1; z<sizeZ-1; z++) {
        val=im(x,y,z);
        if( val>im(x,y,z-1) && val>im(x-1,y,z-1) && val>im(x-1,y-1,z-1) && val>im(x,y-1,z-1) && //Plane z-1
            val>im(x-1,y,z) && val>im(x-1,y-1,z) && val>im(x,y-1,z) && //Plane z
            val>im(x,y,z+1) && val>im(x-1,y,z+1) && val>im(x-1,y-1,z+1) && val>im(x,y-1,z+1)){ //Plane z+1
                detect_maxima(Nmaxima, x, y, z, val);
            }
    }
    //Top Right Receding Edge x=0; y=sizeY-1; z=1 ... sizeZ-2;
    for(x=0,y=sizeY-1,z=1; z<sizeZ-1; z++) {
        val=im(x,y,z);
        if( val>im(x,y,z-1) && val>im(x+1,y,z-1) && val>im(x+1,y-1,z-1) && val>im(x,y-1,z-1) && //Plane z-1
            val>im(x+1,y,z) && val>im(x+1,y-1,z) && val>im(x,y-1,z) && //Plane z
            val>im(x,y,z+1) && val>im(x+1,y,z+1) && val>im(x+1,y-1,z+1) && val>im(x,y-1,z+1)){ //Plane z+1
                detect_maxima(Nmaxima, x, y, z, val);
            }
    }


    /* Backward Face (z=sizeZ-1) Edges and Corners */
    //Top Left Backward corner x=0; y=0; z=sizeZ-1
    x=0; y=0; z=sizeZ-1;
    val=im(x,y,z);
    if(val>im(x,y+1,z) && val>im(x+1,y,z)   && val>im(x+1,y+1,z) && //Plane z=sizeZ-1
       val>im(x,y,z-1) && val>im(x,y+1,z-1) && val>im(x+1,y,z-1) && val>im(x+1,y+1,z-1)){ //Plane z=sizeZ-2
        detect_maxima(Nmaxima, x, y, z, val);
    }
    //Left Backward edge x=1 ... sizeX-2; y=0; z=sizeZ-1
    for(x=1; x<sizeX-1; x++) {
        val=im(x,y,z);
        if( val>im(x-1,y,z) && val>im(x+1,y,z) && val>im(x-1,y+1,z) && val>im(x,y+1,z) && val>im(x+1,y+1,z) && //Plane z=sizeZ-1
            val>im(x,y,z-1) && val>im(x-1,y,z-1) && val>im(x+1,y,z-1) && val>im(x-1,y+1,z-1) && val>im(x,y+1,z-1) && val>im(x+1,y+1,z-1)){ //Plane z=sizeZ-2
            detect_maxima(Nmaxima, x, y, z, val);
        }
    }
    //Bottom left Backward corner x=sizeX-1; y=0; z=sizeZ-1
    val=im(x,y,z);
    if( val>im(x,y+1,z) && val>im(x-1,y,z) && val>im(x-1,y+1,z) && //Plane z=sizeZ-1
        val>im(x,y,z-1) && val>im(x,y+1,z-1) && val>im(x-1,y,z-1) && val>im(x-1,y+1,z-1)){//Plane z=sizeZ-2
        detect_maxima(Nmaxima, x, y, z, val);
    }
    //Bottom Backward edge x=sizeX-1; y=1 ... sizeY-2; z=sizeZ-1
    for(y=1; y<sizeY-1; y++) {
        val=im(x,y,z);
        if( val>im(x,y-1,z) && val>im(x,y+1,z) && val>im(x-1,y-1,z) && val>im(x-1,y,z) && val>im(x-1,y+1,z) &&//Plane z=sizeZ-1
            val>im(x,y,z-1) && val>im(x,y-1,z-1) && val>im(x,y+1,z-1) && val>im(x-1,y-1,z-1) && val>im(x-1,y,z-1) && val>im(x-1,y+1,z-1)){ //Plane z=sizeZ-2
            detect_maxima(Nmaxima, x, y, z, val);
        }
    }
    //Bottom Right Backward corner x=sizeX-1; y=sizeY-1; z=sizeZ-1
    val=im(x,y,z);
    if(val>im(x,y-1,z) && val>im(x-1,y,z) && val>im(x-1,y-1,z) &&//Plane z=sizeZ-1
        val>im(x,y,z-1) && val>im(x,y-1,z-1) && val>im(x-1,y,z-1) && val>im(x-1,y-1,z-1)){//Plane z=sizeZ-2
        detect_maxima(Nmaxima, x, y, z, val);
    }
    //Right Backward edge x=sizeX-2 ... 1; y=sizeY-1; z=sizeZ-1
    for(x=sizeX-2; x>=1; x--) {
        val=im(x,y,z);
        if( val>im(x-1,y,z) && val>im(x+1,y,z) && val>im(x-1,y-1,z) && val>im(x,y-1,z) && val>im(x+1,y-1,z) && //Plane z=sizeZ-1
            val>im(x,y,z-1) && val>im(x-1,y,z-1) && val>im(x+1,y,z-1) && val>im(x-1,y-1,z-1) && val>im(x,y-1,z-1) && val>im(x+1,y-1,z-1)){//Plane z=sizeZ-2
            detect_maxima(Nmaxima, x, y, z, val);
        }
    }
    //Top Right Backward corner x=0; y=sizeY-1; z=sizeZ-1
    val=im(x,y,z);
    if(val>im(x,y-1,z) && val>im(x+1,y,z) && val>im(x+1,y-1,z) &&//Plane z=sizeZ-1
        val>im(x,y,z-1) && val>im(x,y-1,z-1) && val>im(x+1,y,z-1) && val>im(x+1,y-1,z-1)){//Plane z=sizeZ-2
        detect_maxima(Nmaxima, x, y, z, val);
    }
    //Top Backward edge x=0; y=sizeY-2...1; z=sizeZ-1
    for(y=sizeY-2; y>=1; y--) {
        val=im(x,y,z);
        if( val>im(x,y-1,z) && val>im(x,y+1,z) && val>im(x+1,y-1,z) && val>im(x+1,y,z) && val>im(x+1,y+1,z) &&//Plane z=sizeZ-1
            val>im(x,y,z-1) && val>im(x,y-1,z-1) && val>im(x,y+1,z-1) && val>im(x+1,y-1,z-1) && val>im(x+1,y,z-1) && val>im(x+1,y+1,z-1)){//Plane z=sizeZ-2
            detect_maxima(Nmaxima, x, y,z, val);
        }
    }

    /* Faces: The edges and corners have already been checked */
    //Top face x=0; y=1...sizeY-2; z=1...sizeZ-2;
    for(x=0, z=1; z<sizeZ-1; z++) for(y=1; y<sizeY-1; y++) {
        val=im(x,y,z);
        if(val>im(x,y-1,z-1)   && val>im(x,y,z-1)   && val>im(x,y+1,z-1)   && val>im(x,y-1,z)                      && val>im(x,y+1,z)   && val>im(x,y-1,z+1)   && val>im(x,y,z+1)   && val>im(x,y+1,z+1) &&//Plane x=0
            val>im(x+1,y-1,z-1) && val>im(x+1,y,z-1) && val>im(x+1,y+1,z-1) && val>im(x+1,y-1,z) && val>im(x+1,y,z) && val>im(x+1,y+1,z) && val>im(x+1,y-1,z+1) && val>im(x+1,y,z+1) && val>im(x+1,y+1,z+1)){//Plane x=1
                detect_maxima(Nmaxima, x, y, z, val);
        }
    }
    //Bottom face x=sizeX-1; y=1...sizeY-2; z=1...sizeZ-2;
    for(x=sizeX-1, z=1; z<sizeZ-1; z++) for(y=1; y<sizeY-1; y++) {
        val=im(x,y,z);
        if(val>im(x,y-1,z-1)   && val>im(x,y,z-1)   && val>im(x,y+1,z-1)   && val>im(x,y-1,z)                      && val>im(x,y+1,z)   && val>im(x,y-1,z+1)   && val>im(x,y,z+1)   && val>im(x,y+1,z+1) &&//Plane x=0
           val>im(x-1,y-1,z-1) && val>im(x-1,y,z-1) && val>im(x-1,y+1,z-1) && val>im(x-1,y-1,z) && val>im(x-1,y,z) && val>im(x-1,y+1,z) && val>im(x-1,y-1,z+1) && val>im(x-1,y,z+1) && val>im(x-1,y+1,z+1)){//Plane x=1
                detect_maxima(Nmaxima, x, y, z, val);
        }
    }

    //Left face x=1...sizeX-2; y=0; z=1...sizeZ-2;
    for(y=0, z=1; z<sizeZ-1; z++) for(x=1; x<sizeX-1; x++) {
        val=im(x,y,z);
        if(val>im(x-1,y,z-1)   && val>im(x,y,z-1)   && val>im(x+1,y,z-1)   && val>im(x-1,y,z)                      && val>im(x+1,y,z)   && val>im(x-1,y,z+1)   && val>im(x,y,z+1)   && val>im(x+1,y,z+1) &&//Plane y=0
           val>im(x-1,y+1,z-1) && val>im(x,y+1,z-1) && val>im(x+1,y+1,z-1) && val>im(x-1,y+1,z) && val>im(x,y+1,z) && val>im(x+1,y+1,z) && val>im(x-1,y+1,z+1) && val>im(x,y+1,z+1) && val>im(x+1,y+1,z+1)){//Plane y=1
                detect_maxima(Nmaxima, x, y, z, val);
        }
    }
    //Right face x=1...sizeX-2; y=sizeY-1; z=1...sizeZ-2;
    for(y=sizeY-1, z=1; z<sizeZ-1; z++) for(x=1; x<sizeX-1; x++) {
        val=im(x,y,z);
        if(val>im(x-1,y,z-1)   && val>im(x,y,z-1)   && val>im(x+1,y,z-1)   && val>im(x-1,y,z)                      && val>im(x+1,y,z)   && val>im(x-1,y,z+1)   && val>im(x,y,z+1)   && val>im(x+1,y,z+1) &&//Plane y=0
            val>im(x-1,y-1,z-1) && val>im(x,y-1,z-1) && val>im(x+1,y-1,z-1) && val>im(x-1,y-1,z) && val>im(x,y-1,z) && val>im(x+1,y-1,z) && val>im(x-1,y-1,z+1) && val>im(x,y-1,z+1) && val>im(x+1,y-1,z+1)){//Plane y=1
                detect_maxima(Nmaxima, x, y, z, val);
            }
    }

    //Front face x=1...sizeX-2; y=1...sizeY-2; z=0;
    for(z=0, y=1; y<sizeY-1; y++) for(x=1; x<sizeX-1; x++) {
        val=im(x,y,z);
        if(val>im(x-1,y-1,z)   && val>im(x,y-1,z)   && val>im(x+1,y-1,z)   && val>im(x-1,y,z)                      && val>im(x+1,y,z)   && val>im(x-1,y+1,z)   && val>im(x,y+1,z)   && val>im(x+1,y+1,z) &&//Plane z=0
            val>im(x-1,y-1,z+1) && val>im(x,y-1,z+1) && val>im(x+1,y-1,z+1) && val>im(x-1,y,z+1) && val>im(x,y,z+1) && val>im(x+1,y,z+1) && val>im(x-1,y+1,z+1) && val>im(x,y+1,z+1) && val>im(x+1,y+1,z+1)){//Plane z=1
                detect_maxima(Nmaxima, x, y, z, val);
            }
    }
    //Rear face x=1...sizeX-2; y=1...sizeY-2; z=sizeZ-1;
    for(z=sizeZ-1, y=1; y<sizeY-1; y++) for(x=1; x<sizeX-1; x++) {
        val=im(x,y,z);
        if(val>im(x-1,y-1,z)   && val>im(x,y-1,z)   && val>im(x+1,y-1,z)   && val>im(x-1,y,z)                      && val>im(x+1,y,z)   && val>im(x-1,y+1,z)   && val>im(x,y+1,z)   && val>im(x+1,y+1,z) &&//Plane z=sizeZ-1
           val>im(x-1,y-1,z-1) && val>im(x,y-1,z-1) && val>im(x+1,y-1,z-1) && val>im(x-1,y,z-1) && val>im(x,y,z-1) && val>im(x+1,y,z-1) && val>im(x-1,y+1,z-1) && val>im(x,y+1,z-1) && val>im(x+1,y+1,z-1)){//Plane z=sizeZ-2
                detect_maxima(Nmaxima, x, y, z, val);
        }
    }
    /* Why do cubes have so many damn corners, edges, and faces?! */
    return Nmaxima;
}

template<class FloatT, class IdxT>
IdxT Maxima3D<FloatT, IdxT>::maxima_5x5(const ImageT &im)
{
    IdxT Nmaxima=maxima_3x3(im);
    IMatT new_maxima(dim, Nmaxima);
    VecT new_max_vals(Nmaxima);
    IdxT new_Nmaxima=0;
    IdxT sizeX=size(0);
    IdxT sizeY=size(1);
    IdxT sizeZ=size(2);
    for(IdxT n=0; n<Nmaxima;n++){
        IdxT max_x=maxima(0,n);
        IdxT max_y=maxima(1,n);
        IdxT max_z=maxima(2,n);
        FloatT max_val=max_vals(n);
        IdxT x_upper=std::min(max_x+2,size(0));
        IdxT x_lower=std::max(max_x-2,0);
        //Bounds for checking an entire face.
        IdxT y_face_upper=std::min(max_y+2,size(1));
        IdxT y_face_lower=std::max(max_y-2,0);
        //Bounds for checking the edges of a face
        IdxT y_edge_upper=std::min(max_y+2,size(1));
        IdxT y_edge_lower=std::max(max_y-2,0);
        //Bounds for iterating over inner faces
        IdxT z_upper=std::min(max_z+1,size(2));
        IdxT z_lower=std::max(max_z-1,0);
        bool ok=true;
        IdxT x,y,z;

        if(max_z>=2) { //Check forward face max_z-2
            for(z=max_z-2, x=x_lower; x<x_upper; x++) for(y=y_face_lower; y<y_face_upper; y++)
                if(im(x,y,z)>max_val) {ok=false; break;}
        }
        for(z=z_lower; z<z_upper; z++){ //Check faces max_z-1...max_z+1.  The inner 3x3 of this has face has alrady been checked
            if(ok && max_y>=2) //Check Left Edge
                for(y=max_y-2, x=x_lower; x<x_upper; x++) if(im(x,y,z)>max_val) {ok=false; break;}
            if(ok && max_x+2<sizeX) //Check Bottom Edge
                for(x=max_x+2, y=y_edge_lower; y<y_edge_upper; y++) if(im(x,y,z)>max_val) {ok=false; break;}
            if(ok && max_y+2<sizeY) //Check Right Edge
                for(y=max_y+2, x=x_lower; x<x_upper; x++) if(im(x,y,z)>max_val) {ok=false; break;}
            if(ok && max_x>=2) //Check Top Edge
                for(x=max_x-2, y=y_edge_lower; y<y_edge_upper; y++) if(im(x,y,z)>max_val) {ok=false; break;}
        }
        if(ok && max_z+2<sizeZ) { //Check backward face max_z+2
            for(z=max_z+2, x=x_lower; x<x_upper; x++) for(y=y_face_lower; y<y_face_upper; y++)
                if(im(x,y,z)>max_val) {ok=false; break;}
        }
        if(ok){
            new_maxima.col(new_Nmaxima)=maxima.col(n);
            new_max_vals(new_Nmaxima)=max_vals(n);
            new_Nmaxima++;
        }
    }
    if(new_Nmaxima>0){
        maxima.cols(0,new_Nmaxima-1)=new_maxima.cols(0,new_Nmaxima-1);
        max_vals.rows(0,new_Nmaxima-1)=new_max_vals.rows(0,new_Nmaxima-1);
    }
    return new_Nmaxima;
}

template<class FloatT, class IdxT>
IdxT Maxima3D<FloatT, IdxT>::maxima_nxn(const ImageT &im, IdxT filter_size)
{
    IdxT Nmaxima=maxima_3x3(im);
    assert(filter_size%2==1);
    assert(filter_size>3);
    IdxT k = (filter_size-1)/2;
    IMatT new_maxima(dim, Nmaxima);
    VecT new_max_vals(Nmaxima);
    IdxT new_Nmaxima=0;
    for(IdxT n=0; n<Nmaxima;n++){
        FloatT max_val=max_vals(n);
        IdxT max_x=maxima(0,n);
        IdxT max_y=maxima(1,n);
        IdxT max_z=maxima(2,n);
        IdxT x_upper = std::min(max_x+k,size(0)-1);
        IdxT x_lower = std::max(max_x-k,0);
        IdxT y_upper = std::min(max_y+k,size(1)-1);
        IdxT y_lower = std::max(max_y-k,0);
        IdxT z_upper = std::min(max_z+k,size(2)-1);
        IdxT z_lower = std::max(max_z-k,0);

        for(IdxT z=z_lower; z<=z_upper; z++) { //process each face
            for(IdxT y=y_lower; y<=y_upper; y++) { //process each column
                if(max_z-1<=z && z<=max_z+1 && max_y-1<=y && y<=max_y-1){ //middle column skip the portion already checked in the 3x3x3 core 
                    for(IdxT x=x_lower; x<=max_x-2; x++) if(im(x,y,z)>max_val) goto maxima3D_nxn_reject;
                    for(IdxT x=max_x+2; x<=x_upper; x++) if(im(x,y,z)>max_val) goto maxima3D_nxn_reject;
                } else { //left or right column. Process entire column
                    for(IdxT x=x_lower; x<=x_upper; x++) if(im(x,y,z)>max_val) goto maxima3D_nxn_reject;
                } 
            }
        }
        //OK if we made it here so record
        new_maxima.col(new_Nmaxima)=maxima.col(n);
        new_max_vals(new_Nmaxima)=max_vals(n);
        new_Nmaxima++;
maxima3D_nxn_reject: ;//Go here when local maxima is not valid
    }
    if(new_Nmaxima>0){ //Shrink down maxima, but keep maxima and max_vals the same size.
        maxima.cols(0,new_Nmaxima-1)=new_maxima.cols(0,new_Nmaxima-1);
        max_vals.rows(0,new_Nmaxima-1)=new_max_vals.rows(0,new_Nmaxima-1);
    }
    return new_Nmaxima;
}

/* Explicit Template Instantiation */
template class Maxima2D<float>;
template class Maxima2D<double>;

template class Maxima3D<float>;
template class Maxima3D<double>;

} /* namespace boxxer */
