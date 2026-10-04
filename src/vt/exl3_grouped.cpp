#include "vt/exl3_grouped.h"
#include "vt/exl3_w8a8_panel_plan.h"
#include <algorithm>
#include <limits>

namespace vt {
Exl3W8A8Plan PlanExl3W8A8(int64_t m, int64_t k, int64_t n,
                         int64_t groups, int bits) {
  VT_CHECK(m > 128 && m <= 4096, "EXL3 W8A8 requires physical M in [129,4096]");
  VT_CHECK(k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 &&
               k <= std::numeric_limits<int>::max() &&
               n <= std::numeric_limits<int>::max() && groups > 0 &&
               groups <= std::numeric_limits<int>::max() / m / 16,
           "EXL3 W8A8 invalid geometry or I32 input launch overflow");
  VT_CHECK(bits == 4 || bits == 6, "EXL3 W8A8 supports 4/6bpw only");
  VT_CHECK(k <= std::numeric_limits<int32_t>::max() / (127 * 127),
           "EXL3 W8A8 K exceeds bounded signed INT32 accumulation");
  VT_CHECK(m * (n / 128) <= std::numeric_limits<int>::max(),
           "EXL3 W8A8 output launch exceeds I32 indexing");
  const int ms = int((m + 255) / 256 * 256);
  auto multiply = [](size_t a, size_t b) {
    VT_CHECK(b == 0 || a <= (std::numeric_limits<size_t>::max() - 63) / b,
             "EXL3 W8A8 workspace size overflow");
    return a * b;
  };
  size_t cursor = 0;
  auto region = [&](size_t bytes) {
    const size_t start = cursor;
    VT_CHECK(bytes <= std::numeric_limits<size_t>::max() - cursor - 63,
             "EXL3 W8A8 workspace size overflow");
    cursor += (bytes + 63) / 64 * 64;
    return start;
  };
  const size_t rows = multiply(size_t(groups), size_t(ms));
  const size_t act = region(multiply(rows, size_t(k)));
  const size_t sx = region(multiply(rows, sizeof(float)));
  const size_t y = region(multiply(multiply(size_t(ms), size_t(n)), size_t{2}));
  const size_t sw = region(sizeof(float));
  const auto panel = PlanExl3W8A8PanelCapacity(k, n, 128);
  return {ms, act, sx, y, sw, cursor, panel.bytes};
}

void Exl3GroupedW8A8(Queue& q, Tensor& out, const Tensor& in, const Tensor& tr,
    const Tensor& suh, const Tensor& svh, const Tensor& shard,
    Tensor& workspace, Tensor& panel, const Exl3GroupedLinearArgs& args) {
  VT_CHECK(in.rank == 2 && out.rank == 2 && suh.rank == 2,
           "EXL3 W8A8 requires rank-2 input/output/suh");
  const int64_t m = in.shape[0], k = in.shape[1], n = out.shape[1];
  const auto plan = PlanExl3W8A8(m, k, n, suh.shape[0], args.bits);
  VT_CHECK(args.codebook == 2 && out.shape[0] == m &&
               in.dtype == DType::kF16 && out.dtype == DType::kF16,
           "EXL3 W8A8 requires mul1 and matching F16 input/output");
  VT_CHECK(tr.dtype == DType::kI8 && tr.rank == 3 && tr.shape[0] == k / 16 &&
               tr.shape[1] == n / 16 && tr.shape[2] == 32 * args.bits,
           "EXL3 W8A8 packed trellis layout mismatch");
  VT_CHECK(suh.dtype == DType::kF16 && suh.shape[1] == k &&
               svh.dtype == DType::kF16 && svh.rank == 1 && svh.shape[0] == n,
           "EXL3 W8A8 requires F16 suh[S,K] and svh[N]");
  VT_CHECK(shard.dtype == DType::kI32 && shard.rank == 1 && shard.shape[0] == n / 128,
           "EXL3 W8A8 requires I32 shard_of_nb[N/128]");
  VT_CHECK(workspace.dtype == DType::kI8 && workspace.rank == 1 &&
               workspace.shape[0] >= 0 && size_t(workspace.shape[0]) >= plan.workspace_bytes,
           "EXL3 W8A8 byte workspace too small or wrong layout");
  VT_CHECK(panel.dtype == DType::kI8 && panel.rank == 2 &&
               panel.shape[0] == k && panel.shape[1] == 128,
           "EXL3 W8A8 requires bounded I8[K,128] weight panel");
  for (const Tensor* t : std::initializer_list<const Tensor*>{
           &out, &in, &tr, &suh, &svh, &shard, &workspace, &panel}) {
    VT_CHECK(t->IsContiguous(), "EXL3 W8A8 requires contiguous tensors");
    VT_CHECK(t->device == q.device, "EXL3 W8A8 device mismatch");
  }
  reinterpret_cast<Exl3GroupedLinearFn>(GetOp(OpId::kExl3GroupedW8A8, q.device.type))(
      q, out, in, tr, suh, svh, shard, workspace, panel, args);
}

Exl3SmallMPlan PlanExl3SmallM(int64_t m, int64_t k, int64_t n, int bits) {
  VT_CHECK(m > 0 && m <= 128, "EXL3 SmallM requires physical M in [1,128]");
  VT_CHECK(k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 &&
               k <= std::numeric_limits<int>::max() &&
               n <= std::numeric_limits<int>::max(),
           "EXL3 SmallM requires positive I32 K/N multiples of 128");
  VT_CHECK(bits == 4 || bits == 6, "EXL3 SmallM supports 4/6bpw only");
  // c59d944 exl3_ops.sycl defaults: vector M<=2, DPAS MB24/40/48 enabled,
  // max MB64, NT2 at MB>=24, and thread targets 1024/1408/2048.
  const bool vector = m <= 2;
  const int mb = vector ? int(m) : m <= 8 ? 8 : m <= 16 ? 16 :
                 m <= 24 ? 24 : m <= 32 ? 32 : m <= 40 ? 40 : m <= 48 ? 48 : 64;
  const int blocks = vector ? 1 : int((m + mb - 1) / mb);
  const int nt = vector ? (bits == 4 && m == 1 ? 8 : 4) : mb <= 16 ? 4 : 2;
  const int64_t units = (n / 16 / nt) * blocks;
  const int target = vector ? 1024 : mb <= 16 ? 1408 : mb >= 40 ? 2048 : 1024;
  const int tk = int(k / 16);
  const int requested = int(std::max(int64_t{1}, std::min(int64_t(tk),
                                         (target + units - 1) / units)));
  const int rps = (tk + requested - 1) / requested;
  return {vector, mb, blocks * mb, nt, (tk + rps - 1) / rps, rps};
}

void Exl3GroupedLinear(Queue& q, Tensor& out, const Tensor& in, const Tensor& trellis,
    const Tensor& suh, const Tensor& svh, const Tensor& shard,
    Tensor& in_had, Tensor& partials, const Exl3GroupedLinearArgs& args) {
  VT_CHECK(in.rank == 2 && out.rank == 2, "EXL3 grouped linear requires rank-2 input/output");
  const int64_t m = in.shape[0], k = in.shape[1], n = out.shape[1];
  const auto plan = PlanExl3SmallM(m, k, n, args.bits);
  VT_CHECK(args.codebook == 2, "EXL3 grouped linear requires mul1 codebook");
  VT_CHECK(out.shape[0] == m, "EXL3 grouped linear output row mismatch");
  VT_CHECK(in.dtype == DType::kF16 && out.dtype == DType::kF16,
           "EXL3 grouped linear requires F16 model input/output");
  VT_CHECK(trellis.dtype == DType::kI8 && trellis.rank == 3 &&
               trellis.shape[0] == k / 16 && trellis.shape[1] == n / 16 &&
               trellis.shape[2] == 32 * args.bits,
           "EXL3 grouped linear requires opaque packed trellis [K/16,N/16,32*bits]");
  VT_CHECK(suh.dtype == DType::kF16 && suh.rank == 2 && suh.shape[0] > 0 &&
               suh.shape[0] <= std::numeric_limits<int>::max() && suh.shape[1] == k,
           "EXL3 grouped linear requires F16 suh[S,K]");
  VT_CHECK(suh.shape[0] * m * (k / 128) <= std::numeric_limits<int>::max(),
           "EXL3 grouped linear input launch exceeds I32 indexing");
  VT_CHECK(svh.dtype == DType::kF16 && svh.rank == 1 && svh.shape[0] == n,
           "EXL3 grouped linear requires F16 svh[N]");
  VT_CHECK(shard.dtype == DType::kI32 && shard.rank == 1 && shard.shape[0] == n / 128,
           "EXL3 grouped linear requires I32 shard_of_nb[N/128]");
  VT_CHECK(in_had.dtype == DType::kF16 && in_had.rank == 4 &&
               in_had.shape[0] == suh.shape[0] && in_had.shape[1] == k / 16 &&
               in_had.shape[2] == plan.padded_rows && in_had.shape[3] == 16,
           "EXL3 grouped linear input scratch layout mismatch");
  VT_CHECK(partials.dtype == DType::kF32 && partials.rank == 3 &&
               partials.shape[0] == plan.splits && partials.shape[1] == m &&
               partials.shape[2] == n, "EXL3 grouped linear partial scratch layout mismatch");
  for (const Tensor* t : std::initializer_list<const Tensor*>{
           &out, &in, &trellis, &suh, &svh, &shard, &in_had, &partials}) {
    VT_CHECK(t->IsContiguous(), "EXL3 grouped linear requires contiguous tensors");
    VT_CHECK(t->device == q.device, "EXL3 grouped linear device mismatch");
  }
  reinterpret_cast<Exl3GroupedLinearFn>(GetOp(OpId::kExl3GroupedLinear, q.device.type))(
      q, out, in, trellis, suh, svh, shard, in_had, partials, args);
}
}  // namespace vt
