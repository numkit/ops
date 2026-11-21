// ops/include/numkit/ops/blas1.hpp
#pragma once

#include <keel/linalg.h>

namespace numkit::ops {

using keel::linalg::axpy;
using keel::linalg::dot;
using keel::linalg::dotc;
using keel::linalg::dotu;
using keel::linalg::nrm2;
using keel::linalg::scal;

} // namespace numkit::ops
