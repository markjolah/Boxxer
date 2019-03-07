/**
* @file BoxxerError.h
* @author Mark J. Olah (mjo\@cs.unm DOT edu)
* @date 2014-2019
* @brief Error handling
*/
#ifndef BOXXER_BOXXER_ERROR_H
#define BOXXER_BOXXER_ERROR_H

#include "BacktraceException/BacktraceException.h"

namespace boxxer {

using BoxxerError = backtrace_exception::BacktraceException;

/** @brief Parameter value is not valid.
 */
struct ParameterValueError : public BoxxerError
{
    ParameterValueError(std::string message) : BoxxerError("ParameterValueError",message) {}
};

/** @brief Parameters are the incorrect shape, size or number of dimensions
 */
struct ParameterShapeError : public BoxxerError
{
    ParameterValueError(std::string message) : BoxxerError("ParameterShapeError",message) {}
};

/** @brief Internal logical error.  Bad logic or broken promises.
 */
struct LogicalError : public BoxxerError
{
    LogicalError(std::string message) : BoxxerError("LogicalError",message) {}
};

/** @brief Internal numerical error.
 */
struct NumericalError : public BoxxerError
{
    NumericalError(std::string message) : BoxxerError("NumericalError",message) {}
};

} /* namespace boxxer */

#endif /* BOXXER_BOXXER_ERROR_H */
