#pragma once
#include "vt/ops.h"

namespace vt {
// Pinned EXL3 mul1 SmallM launch geometry. M is the physical row count,
// including graph padding; M>128 needs the separate W8A8 implementation.
struct Exl3SmallMPlan {
  bool vector;
  int row_block, padded_rows, tiles_per_thread, splits, tile_rows_per_split;
};
Exl3SmallMPlan PlanExl3SmallM(int64_t m, int64_t k, int64_t n, int bits);

struct Exl3GroupedLinearArgs {
  int bits = 4;
  int codebook = 2;
  const char* debug_name = nullptr;
};
using Exl3GroupedLinearFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
    const Exl3GroupedLinearArgs&);

// F16[M,K] -> F16[M,N], opaque trellis bytes [K/16,N/16,32*bits].
// suh is F16[S,K]; shard_of_nb is I32[N/128], selecting an independent input
// transform for each whole 128-column output block; svh is F16[N].
// Caller-owned scratch must remain alive until the queue completes:
//   in_had: F16[S,K/16,plan.padded_rows,16], blocked by input tile/row;
//   partials: F32[plan.splits,M,N], ordered split-K accumulation.
// Scratch/output may not overlap any operand or each other. This initial
// native XPU implementation supports mul1 4/6bpw and physical M<=128 only;
// unsupported arithmetic is refused rather than replaced by Packed/BF16.
void Exl3GroupedLinear(Queue&, Tensor& out, const Tensor& in, const Tensor& trellis,
    const Tensor& suh, const Tensor& svh, const Tensor& shard_of_nb,
    Tensor& in_had, Tensor& partials, const Exl3GroupedLinearArgs&);
}  // namespace vt
