// ops/include/numkit/ops/blas3.hpp
#pragma once

#include <keel/linalg.h>

namespace numkit::ops {

using keel::linalg::MatrixSide;
using keel::linalg::MatrixUplo;
using keel::linalg::MatrixTranspose;
using keel::linalg::MatrixDiag;

using keel::linalg::gemm;
using keel::linalg::trsm;
using keel::linalg::syrk;

} // namespace numkit::ops
