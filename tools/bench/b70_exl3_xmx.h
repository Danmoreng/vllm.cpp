#pragma once
#include "vt/ops.h"

namespace vt::xpu {
// Benchmark-owned, contiguous F16 -> F32 buffers, real codebook-2 weights.
// Experimental accuracy is checked against CPU panels by the benchmark.
void Exl3MatrixProbe(Queue& q, Tensor& out, const Tensor& in, const Tensor& trellis,
    const Tensor& suh, const Tensor& svh, Tensor& in_had, int bits);
}  // namespace vt::xpu
