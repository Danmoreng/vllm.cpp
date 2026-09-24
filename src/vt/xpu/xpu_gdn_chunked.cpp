#include "xpu_common.h"
#include "xpu_kernels.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace vt::xpu {
namespace {
constexpr int C = 64, D = 128, MaxHeads = 48;
constexpr size_t WorkspaceBytes = 16 * 1024 * 1024;
using BF = sycl::ext::oneapi::bfloat16;

// One chunk for one sequence at a time, irrespective of total sequence length.
// K/Q are BF16 XMX operands. The inverse, W/U, deltas, old state and all state
// arithmetic stay F32, unlike the donor's optional BF16 intermediate stores.
struct ChunkScratch {
  BF *q, *k, *kt;
  float *g, *exp_g, *tail_decay, *lower, *qk, *inverse, *w, *u, *delta, *state_t;
  ChunkScratch(void* storage, int heads) {
    auto* cursor = static_cast<char*>(storage);
    auto bf = [&](size_t count) { auto* p = reinterpret_cast<BF*>(cursor); cursor += count * sizeof(BF); return p; };
    auto fp = [&](size_t count) { auto* p = reinterpret_cast<float*>(cursor); cursor += count * sizeof(float); return p; };
    q = bf(heads * C * D); k = bf(heads * C * D); kt = bf(heads * D * C);
    g = fp(heads * C); exp_g = fp(heads * C); tail_decay = fp(heads * C);
    lower = fp(heads * C * C); qk = fp(heads * C * C); inverse = fp(heads * C * C);
    w = fp(heads * C * D); u = fp(heads * C * D); delta = fp(heads * C * D);
    state_t = fp(heads * D * D);
    VT_CHECK(size_t(cursor - static_cast<char*>(storage)) <= WorkspaceBytes, "GDN workspace overflow");
  }
};
static_assert(MaxHeads * (3 * C * D * sizeof(BF) +
    (3 * C + 3 * C * C + 3 * C * D + D * D) * sizeof(float)) <= WorkspaceBytes);

void Prepare(Queue& queue, ChunkScratch s, View qi, View ki, View gates, const float* state,
             const int32_t* offsets, int sequence, int base, int heads, int key_heads) {
  auto& q = NativeQueue(queue);
  // VT has already normalized q/k and transformed g/beta. Only a chunk-local
  // prefix sum belongs here; do not repeat donor softplus, sigmoid or L2Norm.
  q.parallel_for(sycl::range<1>(heads), [=](sycl::id<1> item) {
    const int h = item[0], first = offsets[sequence] + base, end = offsets[sequence + 1];
    float sum = 0;
    for (int i = 0; i < C; ++i) {
      if (first + i < end) sum += Load(gates, (first + i) * heads + h);
      s.g[h * C + i] = sum;
    }
    for (int i = 0; i < C; ++i) {
      s.exp_g[h * C + i] = sycl::exp(s.g[h * C + i]);
      s.tail_decay[h * C + i] = sycl::exp(sum - s.g[h * C + i]);
    }
  });
  q.parallel_for(sycl::range<1>(heads * D * D), [=](sycl::id<1> item) {
    const int h = item[0] / (D * D), inner = item[0] % (D * D), v = inner / D, k = inner % D;
    s.state_t[(h * D + k) * D + v] = state[(sequence * heads + h) * D * D + inner];
    if (inner < C * D) {
      const int row = inner / D, d = inner % D, token = offsets[sequence] + base + row;
      const bool valid = token < offsets[sequence + 1];
      const int kh = h / (heads / key_heads);
      const BF qv(valid ? Load(qi, (token * key_heads + kh) * D + d) : 0.0f);
      const BF kv(valid ? Load(ki, (token * key_heads + kh) * D + d) : 0.0f);
      s.q[h * C * D + inner] = qv; s.k[h * C * D + inner] = kv;
      s.kt[(h * D + d) * C + row] = kv;
    }
  });
}

// Compute K K^T using native BF16 XMX, F32 accumulation. Each SG16
// owns a 16x16 tile; no state operand is converted to BF16 for this operation.
void Dots(Queue& queue, ChunkScratch s, int heads) {
  namespace mx = sycl::ext::oneapi::experimental::matrix;
  NativeQueue(queue).parallel_for(sycl::nd_range<1>(heads * 16 * 16, 16),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
    auto sg = item.get_sub_group();
    const int tile = item.get_group(0);
    const int h = (tile / 16) % heads, row = (tile % 16) / 4 * 16, col = tile % 4 * 16;
    const BF* a = s.k + (h * C + row) * D;
    const BF* b = s.kt + h * D * C + col;
    float* out = s.lower + (h * C + row) * C + col;
    mx::joint_matrix<sycl::sub_group, BF, mx::use::a, 16, 16, mx::layout::row_major> ja;
    mx::joint_matrix<sycl::sub_group, BF, mx::use::b, 16, 16, mx::layout::row_major> jb;
    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> acc;
    mx::joint_matrix_fill(sg, acc, 0.0f);
    for (int d = 0; d < D; d += 16) {
      mx::joint_matrix_load(sg, ja, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(a + d), D);
      mx::joint_matrix_load(sg, jb, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(b + d * C), C);
      mx::joint_matrix_mad(sg, acc, ja, jb, acc);
    }
    mx::joint_matrix_store(sg, acc, sycl::address_space_cast<sycl::access::address_space::global_space,
        sycl::access::decorated::no>(out), C, mx::layout::row_major);
  });
}

void System(Queue& queue, ChunkScratch s, View beta, const int32_t* offsets,
            int sequence, int base, int heads, float scale) {
  auto& q = NativeQueue(queue);
  q.parallel_for(sycl::range<1>(heads * C * C), [=](sycl::id<1> item) {
    const int index = item[0], h = index / (C * C), row = (index / C) % C, col = index % C;
    const int token = offsets[sequence] + base + row;
    const bool valid = token < offsets[sequence + 1];
    const float decay = valid && row >= col ? sycl::exp(s.g[h * C + row] - s.g[h * C + col]) : 0;
    s.lower[index] = valid && row > col ? s.lower[index] * Load(beta, token * heads + h) * decay : 0;
    // VT's recurrence rounds q*scale in F32 before the dot. Factoring scale
    // out of both dots is algebraically equivalent but adds a rounding change.
    // Keep this small dot SIMD/F32 rather than narrowing scaled q to BF16.
    float dot = 0;
    if (valid && row >= col) for (int d = 0; d < D; ++d)
      dot += static_cast<float>(s.k[(h * C + col) * D + d]) *
          (static_cast<float>(s.q[(h * C + row) * D + d]) * scale);
    s.qk[index] = dot * decay;
  });
  // Each work-item solves one column of (I + lower)^-1. Dependencies stay in
  // that column, so neither subgroup barriers nor cross-workgroup waits occur.
  q.parallel_for(sycl::range<1>(heads * C), [=](sycl::id<1> item) {
    const int h = item[0] / C, col = item[0] % C;
    auto* inverse = s.inverse + h * C * C;
    const auto* lower = s.lower + h * C * C;
    for (int row = 0; row < C; ++row) {
      float value = row == col ? 1.0f : 0.0f;
      if (row > col) {
        value = -lower[row * C + col];
        for (int j = col + 1; j < row; ++j) value -= lower[row * C + j] * inverse[j * C + col];
      }
      inverse[row * C + col] = value;
    }
  });
}

void ComputeWU(Queue& queue, ChunkScratch s, View vi, View beta, const int32_t* offsets,
               int sequence, int base, int heads) {
  NativeQueue(queue).parallel_for(sycl::range<1>(heads * C * D), [=](sycl::id<1> item) {
    const int index = item[0], h = index / (C * D), row = (index / D) % C, d = index % D;
    const int first = offsets[sequence] + base, n = sycl::min(C, sycl::max(0, offsets[sequence + 1] - first));
    float w = 0, u = 0;
    if (row < n) for (int j = 0; j <= row; ++j) {
      const float inv = s.inverse[(h * C + row) * C + j];
      const float b = Load(beta, (first + j) * heads + h);
      w += inv * (static_cast<float>(s.k[(h * C + j) * D + d]) * b * s.exp_g[h * C + j]);
      u += inv * (Load(vi, ((first + j) * heads + h) * D + d) * b);
    }
    s.w[index] = w; s.u[index] = u;
  });
}

void OutputState(Queue& queue, ChunkScratch s, View output, float* state, const int32_t* offsets,
                 int sequence, int base, int heads, float scale) {
  auto& q = NativeQueue(queue);
  q.parallel_for(sycl::range<1>(heads * C * D), [=](sycl::id<1> item) {
    const int index = item[0], h = index / (C * D), row = (index / D) % C, v = index % D;
    float prediction = 0;
    for (int d = 0; d < D; ++d) prediction += s.w[(h * C + row) * D + d] * s.state_t[(h * D + d) * D + v];
    s.delta[index] = s.u[index] - prediction;
  });
  q.parallel_for(sycl::range<1>(heads * C * D), [=](sycl::id<1> item) {
    const int index = item[0], h = index / (C * D), row = (index / D) % C, v = index % D;
    const int token = offsets[sequence] + base + row;
    if (token >= offsets[sequence + 1]) return;
    float cross = 0, intra = 0;
    for (int d = 0; d < D; ++d)
      cross += s.state_t[(h * D + d) * D + v] * (static_cast<float>(s.q[(h * C + row) * D + d]) * scale);
    for (int j = 0; j <= row; ++j)
      intra += s.qk[(h * C + row) * C + j] * s.delta[(h * C + j) * D + v];
    Store(output, (token * heads + h) * D + v, cross * s.exp_g[h * C + row] + intra);
  });
  q.parallel_for(sycl::range<1>(heads * D * D), [=](sycl::id<1> item) {
    const int h = item[0] / (D * D), v = (item[0] / D) % D, d = item[0] % D;
    const int n = sycl::min(C, sycl::max(0, offsets[sequence + 1] - offsets[sequence] - base));
    if (!n) return;
    float value = s.state_t[(h * D + d) * D + v] * s.exp_g[h * C + n - 1];
    for (int j = 0; j < n; ++j)
      value += s.delta[(h * C + j) * D + v] * s.tail_decay[h * C + j] *
          static_cast<float>(s.k[(h * C + j) * D + d]);
    state[(sequence * heads + h) * D * D + v * D + d] = value;
  });
}
}

bool GdnChunkedPrefillKernel(Queue& queue, Tensor& out, const Tensor& qi, const Tensor& ki,
                            const Tensor& vi, const Tensor& g, const Tensor& beta, Tensor& state,
                            const Tensor& qsl, const GdnArgs& args) {
  const auto device = NativeQueue(queue).get_device();
  const int heads = state.shape[1], sequences = state.shape[0], key_heads = qi.shape[1], tokens = qi.shape[0];
  if (state.shape[2] != D || state.shape[3] != D || heads < 1 || heads > MaxHeads ||
      sequences < 1 || sequences > 4 || tokens < 1 || tokens > 6656 ||
      qi.dtype != DType::kBF16 || ki.dtype != DType::kBF16 || vi.dtype != DType::kBF16 ||
      !device.has(sycl::aspect::ext_intel_device_id) ||
      device.get_info<sycl::ext::intel::info::device::device_id>() != 57891 ||
      !device.has(sycl::aspect::ext_intel_matrix)) return false;
  VT_CHECK(state.dtype == DType::kF32, "XPU chunked GDN requires F32 state");
  for (const auto* t : std::initializer_list<const Tensor*>{&out, &qi, &ki, &vi, &g, &beta})
    VT_CHECK(!Overlap(state, *t), "XPU GDN state must have separate storage");
  const auto* offsets = static_cast<const int32_t*>(qsl.data);
  CheckDeviceMetadata(queue, [=] {
    if (offsets[0] != 0 || offsets[sequences] != tokens) return false;
    for (int i = 0; i < sequences; ++i) if (offsets[i] < 0 || offsets[i] > offsets[i + 1]) return false;
    return true;
  }, "XPU chunked GDN invalid sequence offsets", {&qsl});
  return WithGdnWorkspace(queue, WorkspaceBytes, [&](void* storage) {
    ChunkScratch scratch(storage, heads);
    WithOutput(queue, out, {&qi, &ki, &vi, &g, &beta}, [&](Tensor& target) {
      for (int sequence = 0; sequence < sequences; ++sequence) for (int base = 0; base < tokens; base += C) {
        Prepare(queue, scratch, View(qi), View(ki), View(g), static_cast<float*>(state.data),
                offsets, sequence, base, heads, key_heads);
        Dots(queue, scratch, heads);
        System(queue, scratch, View(beta), offsets, sequence, base, heads, args.scale);
        ComputeWU(queue, scratch, View(vi), View(beta), offsets, sequence, base, heads);
        OutputState(queue, scratch, View(target), static_cast<float*>(state.data), offsets, sequence, base, heads, args.scale);
      }
    });
  });
}
}  // namespace vt::xpu
