/** @file Boxxer3D_IFace.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief Boxxer3D_IFace mexFunction entry point
 */

#include "Boxxer3D_IFace.h"

Boxxer3D_IFace<float,uint32_t> iface; /**< Global iface object provides a iface.mexFunction */

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
