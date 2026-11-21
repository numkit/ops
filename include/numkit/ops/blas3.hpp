// ops/include/numkit/ops/blas3.hpp
#pragma once

#include <keel/linalg.h>

namespace numkit::ops {

using keel::MatrixSide;
using keel::MatrixUplo;
using keel::MatrixTranspose;
using keel::MatrixDiag;

using keel::gemm;
using keel::trsm;
using keel::syrk;

} // namespace numkit::ops
