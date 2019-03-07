/** @file Boxxer2D_IFace.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2014-2019
 * @brief Boxxer2D_IFace mexFunction entry point
 */

#include "Boxxer/Boxxer2D_IFace.h"

Boxxer2D_IFace<float,int32_t> iface; /**< Global iface object provides a iface.mexFunction */

void mexFunction(int nlhs, mxArray *lhs[], int nrhs, const mxArray *rhs[])
{
    iface.mexFunction(nlhs, lhs, nrhs, rhs);
}
