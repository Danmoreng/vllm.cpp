#include "xpu_common.h"
#include "xpu_kernels.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdlib>
#include <string_view>

namespace vt::xpu {
namespace {
constexpr int C = 64, D = 128, MaxHeads = 48;
constexpr size_t WorkspaceBytes = 16 * 1024 * 1024;
using BF = sycl::ext::oneapi::bfloat16;
bool QkXmxEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VT_XPU_GDN_QK");
    const std::string_view mode = value ? value : "xmx";
    VT_CHECK(mode == "reference" || mode == "xmx", "Invalid VT_XPU_GDN_QK");
    return mode == "xmx";
  }();
  return enabled;
}
bool DeltaCrossEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VT_XPU_GDN_DELTA_CROSS");
    const std::string_view mode = value ? value : "tile4";
    VT_CHECK(mode == "reference" || mode == "tile4", "Invalid VT_XPU_GDN_DELTA_CROSS");
    return mode == "tile4";
  }();
  return enabled;
}
bool IntraStateTileEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VT_XPU_GDN_INTRA_STATE");
    const std::string_view mode = value ? value : "tile4";
    VT_CHECK(mode == "reference" || mode == "tile4", "Invalid VT_XPU_GDN_INTRA_STATE");
    return mode == "tile4";
  }();
  return enabled;
}

// One chunk for one sequence at a time, irrespective of total sequence length.
// K/Q retain their input precision as XMX operands. The inverse, W/U, deltas,
// old state and all state arithmetic stay F32.
template <typename Input>
struct ChunkScratch {
  Input *q, *k, *kt;
  float *g, *exp_g, *tail_decay, *lower, *qk, *inverse, *w, *u, *delta, *cross, *state_t;
  ChunkScratch(void* storage, int heads) {
    auto* cursor = static_cast<char*>(storage);
    auto fp16 = [&](size_t count) { auto* p = reinterpret_cast<Input*>(cursor); cursor += count * sizeof(Input); return p; };
    auto fp = [&](size_t count) { auto* p = reinterpret_cast<float*>(cursor); cursor += count * sizeof(float); return p; };
    q = fp16(heads * C * D); k = fp16(heads * C * D); kt = fp16(heads * D * C);
    g = fp(heads * C); exp_g = fp(heads * C); tail_decay = fp(heads * C);
    lower = fp(heads * C * C); qk = fp(heads * C * C); inverse = fp(heads * C * C);
    w = fp(heads * C * D); u = fp(heads * C * D); delta = fp(heads * C * D);
    cross = fp(heads * C * D);
    state_t = fp(heads * D * D);
    VT_CHECK(size_t(cursor - static_cast<char*>(storage)) <= WorkspaceBytes, "GDN workspace overflow");
  }
};
static_assert(MaxHeads * (3 * C * D * sizeof(BF) +
    (3 * C + 3 * C * C + 4 * C * D + D * D) * sizeof(float)) <= WorkspaceBytes);

template <typename Input>
void Prepare(Queue& queue, ChunkScratch<Input> s, View qi, View ki, View gates, const float* state,
             const int32_t* offsets, int sequence, int base, int heads, int key_heads) {
  auto& q = NativeQueue(queue);
  // VT has already normalized q/k and transformed g/beta. Only a chunk-local
  // prefix sum belongs here; do not repeat donor softplus, sigmoid or L2Norm.
  const auto gates_event = q.parallel_for(sycl::range<1>(heads), [=](sycl::id<1> item) {
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
  RecordProfileEvent(queue, "gdn_chunk_gates", gates_event);
  const auto inputs_event = q.parallel_for(sycl::range<1>(heads * D * D), [=](sycl::id<1> item) {
    const int h = item[0] / (D * D), inner = item[0] % (D * D), v = inner / D, k = inner % D;
    s.state_t[(h * D + k) * D + v] = state[(sequence * heads + h) * D * D + inner];
    if (inner < C * D) {
      const int row = inner / D, d = inner % D, token = offsets[sequence] + base + row;
      const bool valid = token < offsets[sequence + 1];
      const int kh = h / (heads / key_heads);
      const Input qv(valid ? Load(qi, (token * key_heads + kh) * D + d) : 0.0f);
      const Input kv(valid ? Load(ki, (token * key_heads + kh) * D + d) : 0.0f);
      s.q[h * C * D + inner] = qv; s.k[h * C * D + inner] = kv;
      s.kt[(h * D + d) * C + row] = kv;
    }
  });
  RecordProfileEvent(queue, "gdn_chunk_inputs", inputs_event);
}

// Compute K K^T using native BF16/F16 XMX, F32 accumulation. Each SG16
// owns a 16x16 tile; no state operand is narrowed for this operation.
template <typename Input>
void Dots(Queue& queue, ChunkScratch<Input> s, int heads) {
  namespace mx = sycl::ext::oneapi::experimental::matrix;
  const bool qk_xmx = QkXmxEnabled();
  const auto event = NativeQueue(queue).parallel_for(sycl::nd_range<1>(heads * 16 * 16, 16),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
    auto sg = item.get_sub_group();
    const int tile = item.get_group(0);
    const int h = (tile / 16) % heads, row = (tile % 16) / 4 * 16, col = tile % 4 * 16;
    const Input* a = s.k + (h * C + row) * D;
    const Input* q = s.q + (h * C + row) * D;
    const Input* b = s.kt + h * D * C + col;
    float* out = s.lower + (h * C + row) * C + col;
    float* qk = s.qk + (h * C + row) * C + col;
    mx::joint_matrix<sycl::sub_group, Input, mx::use::a, 16, 16, mx::layout::row_major> ja;
    mx::joint_matrix<sycl::sub_group, Input, mx::use::b, 16, 16, mx::layout::row_major> jb;
    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> acc;
    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> qacc;
    mx::joint_matrix_fill(sg, acc, 0.0f);
    if (qk_xmx) mx::joint_matrix_fill(sg, qacc, 0.0f);
    for (int d = 0; d < D; d += 16) {
      mx::joint_matrix_load(sg, ja, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(a + d), D);
      mx::joint_matrix_load(sg, jb, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(b + d * C), C);
      mx::joint_matrix_mad(sg, acc, ja, jb, acc);
      if (qk_xmx) {
        mx::joint_matrix_load(sg, ja, sycl::address_space_cast<sycl::access::address_space::global_space,
            sycl::access::decorated::no>(q + d), D);
        mx::joint_matrix_mad(sg, qacc, ja, jb, qacc);
      }
    }
    mx::joint_matrix_store(sg, acc, sycl::address_space_cast<sycl::access::address_space::global_space,
        sycl::access::decorated::no>(out), C, mx::layout::row_major);
    if (qk_xmx)
      mx::joint_matrix_store(sg, qacc, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(qk), C, mx::layout::row_major);
  });
  RecordProfileEvent(queue, qk_xmx ? "gdn_chunk_dots_qk" : "gdn_chunk_dots", event);
}

template <typename Input>
void System(Queue& queue, ChunkScratch<Input> s, View beta, const int32_t* offsets,
            int sequence, int base, int heads, float scale) {
  auto& q = NativeQueue(queue);
  const bool qk_xmx = QkXmxEnabled();
  const auto system_event = q.parallel_for(sycl::range<1>(heads * C * C), [=](sycl::id<1> item) {
    const int index = item[0];
    const int h = index / (C * C), row = (index / C) % C, col = index % C;
    const int token = offsets[sequence] + base + row;
    const bool valid = token < offsets[sequence + 1];
    const float decay = valid && row >= col ? sycl::exp(s.g[h * C + row] - s.g[h * C + col]) : 0;
    s.lower[index] = valid && row > col ? s.lower[index] * Load(beta, token * heads + h) * decay : 0;
    // The XMX path factors out scale after the FP32 dot. The reference path
    // preserves VT's per-element FP32 q*scale rounding for comparison.
    if (qk_xmx) {
      s.qk[index] = valid && row >= col ? s.qk[index] * scale * decay : 0;
    } else {
      float dot = 0;
      if (valid && row >= col) for (int d = 0; d < D; ++d)
        dot += static_cast<float>(s.k[(h * C + col) * D + d]) *
            (static_cast<float>(s.q[(h * C + row) * D + d]) * scale);
      s.qk[index] = dot * decay;
    }
  });
  RecordProfileEvent(queue, "gdn_chunk_system", system_event);
  // Each work-item solves one column of (I + lower)^-1. Dependencies stay in
  // that column, so neither subgroup barriers nor cross-workgroup waits occur.
  const auto inverse_event = q.parallel_for(sycl::range<1>(heads * C), [=](sycl::id<1> item) {
    const int index = item[0];
    const int h = index / C, col = index % C;
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
  RecordProfileEvent(queue, "gdn_chunk_inverse", inverse_event);
}

template <typename Input>
void ComputeWU(Queue& queue, ChunkScratch<Input> s, View vi, View beta, const int32_t* offsets,
               int sequence, int base, int heads) {
  static const bool tile4 = [] {
    const char* value = std::getenv("VT_XPU_GDN_WU");
    const std::string_view mode = value ? value : "auto";
    VT_CHECK(mode == "auto" || mode == "reference" || mode == "tile4", "Invalid VT_XPU_GDN_WU");
    return mode != "reference";
  }();
  if (tile4) {
    constexpr int width = 4;
    const auto event = NativeQueue(queue).parallel_for(
        sycl::range<1>(heads * C * (D / width)), [=](sycl::id<1> item) {
      const int index = item[0];
      const int h = index / (C * (D / width));
      const int row = (index / (D / width)) % C, d = (index % (D / width)) * width;
      const int first = offsets[sequence] + base;
      const int n = sycl::min(C, sycl::max(0, offsets[sequence + 1] - first));
      float w[width] = {}, u[width] = {};
      if (row < n) for (int j = 0; j <= row; ++j) {
        const float inv = s.inverse[(h * C + row) * C + j];
        const float b = Load(beta, (first + j) * heads + h);
        const float decay = s.exp_g[h * C + j];
        for (int col = 0; col < width; ++col) {
          w[col] += inv * (static_cast<float>(s.k[(h * C + j) * D + d + col]) * b * decay);
          u[col] += inv * (Load(vi, ((first + j) * heads + h) * D + d + col) * b);
        }
      }
      for (int col = 0; col < width; ++col) {
        s.w[(h * C + row) * D + d + col] = w[col];
        s.u[(h * C + row) * D + d + col] = u[col];
      }
    });
    RecordProfileEvent(queue, "gdn_chunk_wu_tile4", event);
    return;
  }
  const auto event = NativeQueue(queue).parallel_for(sycl::range<1>(heads * C * D), [=](sycl::id<1> item) {
    const int index = item[0];
    const int h = index / (C * D), row = (index / D) % C, d = index % D;
    const int first = offsets[sequence] + base;
    const int n = sycl::min(C, sycl::max(0, offsets[sequence + 1] - first));
    float w = 0, u = 0;
    if (row < n) for (int j = 0; j <= row; ++j) {
      const float inv = s.inverse[(h * C + row) * C + j];
      const float b = Load(beta, (first + j) * heads + h);
      w += inv * (static_cast<float>(s.k[(h * C + j) * D + d]) * b * s.exp_g[h * C + j]);
      u += inv * (Load(vi, ((first + j) * heads + h) * D + d) * b);
    }
    s.w[index] = w; s.u[index] = u;
  });
  RecordProfileEvent(queue, "gdn_chunk_wu", event);
}

template <typename Input>
void OutputState(Queue& queue, ChunkScratch<Input> s, View output, float* state, const int32_t* offsets,
                 int sequence, int base, int heads, float scale) {
  auto& q = NativeQueue(queue);
  const bool delta_cross = DeltaCrossEnabled();
  if (delta_cross) {
    constexpr int width = 4;
    const auto event = q.parallel_for(sycl::range<1>(heads * C * (D / width)), [=](sycl::id<1> item) {
      const int index = item[0], h = index / (C * (D / width));
      const int row = (index / (D / width)) % C, v = (index % (D / width)) * width;
      float prediction[width] = {}, cross[width] = {};
      for (int d = 0; d < D; ++d) {
        const float w = s.w[(h * C + row) * D + d];
        const float qscaled = static_cast<float>(s.q[(h * C + row) * D + d]) * scale;
        for (int col = 0; col < width; ++col) {
          const float old = s.state_t[(h * D + d) * D + v + col];
          prediction[col] += w * old;
          cross[col] += old * qscaled;
        }
      }
      for (int col = 0; col < width; ++col) {
        const int at = (h * C + row) * D + v + col;
        s.delta[at] = s.u[at] - prediction[col];
        s.cross[at] = cross[col];
      }
    });
    RecordProfileEvent(queue, "gdn_chunk_delta_cross_tile4", event);
  } else {
    const auto delta_event = q.parallel_for(sycl::range<1>(heads * C * D), [=](sycl::id<1> item) {
      const int index = item[0], h = index / (C * D), row = (index / D) % C, v = index % D;
      float prediction = 0;
      for (int d = 0; d < D; ++d)
        prediction += s.w[(h * C + row) * D + d] * s.state_t[(h * D + d) * D + v];
      s.delta[index] = s.u[index] - prediction;
    });
    RecordProfileEvent(queue, "gdn_chunk_delta", delta_event);
  }
  const bool intra_state_tile = IntraStateTileEnabled();
  if (intra_state_tile && delta_cross) {
    constexpr int width = 4;
    const auto event = q.parallel_for(sycl::range<1>(heads * C * (D / width)), [=](sycl::id<1> item) {
      const int index = item[0], h = index / (C * (D / width));
      const int row = (index / (D / width)) % C, v = (index % (D / width)) * width;
      const int token = offsets[sequence] + base + row;
      if (token >= offsets[sequence + 1]) return;
      float intra[width] = {};
      for (int j = 0; j <= row; ++j) {
        const float qk = s.qk[(h * C + row) * C + j];
        for (int col = 0; col < width; ++col)
          intra[col] += qk * s.delta[(h * C + j) * D + v + col];
      }
      for (int col = 0; col < width; ++col) {
        const int at = (h * C + row) * D + v + col;
        Store(output, (token * heads + h) * D + v + col,
              s.cross[at] * s.exp_g[h * C + row] + intra[col]);
      }
    });
    RecordProfileEvent(queue, "gdn_chunk_output_tile4", event);
  } else {
    const auto output_event = q.parallel_for(sycl::range<1>(heads * C * D), [=](sycl::id<1> item) {
      const int index = item[0], h = index / (C * D), row = (index / D) % C, v = index % D;
      const int token = offsets[sequence] + base + row;
      if (token >= offsets[sequence + 1]) return;
      float cross = 0, intra = 0;
      if (delta_cross) cross = s.cross[index];
      else for (int d = 0; d < D; ++d)
        cross += s.state_t[(h * D + d) * D + v] * (static_cast<float>(s.q[(h * C + row) * D + d]) * scale);
      for (int j = 0; j <= row; ++j)
        intra += s.qk[(h * C + row) * C + j] * s.delta[(h * C + j) * D + v];
      Store(output, (token * heads + h) * D + v, cross * s.exp_g[h * C + row] + intra);
    });
    RecordProfileEvent(queue, "gdn_chunk_output", output_event);
  }
  if (intra_state_tile) {
    constexpr int width = 4;
    const auto event = q.parallel_for(sycl::range<1>(heads * D * (D / width)), [=](sycl::id<1> item) {
      const int index = item[0], h = index / (D * (D / width));
      const int v = (index / (D / width)) % D, d = (index % (D / width)) * width;
      const int n = sycl::min(C, sycl::max(0, offsets[sequence + 1] - offsets[sequence] - base));
      if (!n) return;
      float value[width];
      for (int col = 0; col < width; ++col)
        value[col] = s.state_t[(h * D + d + col) * D + v] * s.exp_g[h * C + n - 1];
      for (int j = 0; j < n; ++j) {
        const float weighted_delta = s.delta[(h * C + j) * D + v] * s.tail_decay[h * C + j];
        for (int col = 0; col < width; ++col)
          value[col] += weighted_delta * static_cast<float>(s.k[(h * C + j) * D + d + col]);
      }
      for (int col = 0; col < width; ++col)
        state[(sequence * heads + h) * D * D + v * D + d + col] = value[col];
    });
    RecordProfileEvent(queue, "gdn_chunk_state_tile4", event);
    return;
  }
  const auto state_event = q.parallel_for(sycl::range<1>(heads * D * D), [=](sycl::id<1> item) {
    const int h = item[0] / (D * D), v = (item[0] / D) % D, d = item[0] % D;
    const int n = sycl::min(C, sycl::max(0, offsets[sequence + 1] - offsets[sequence] - base));
    if (!n) return;
    float value = s.state_t[(h * D + d) * D + v] * s.exp_g[h * C + n - 1];
    for (int j = 0; j < n; ++j)
      value += s.delta[(h * C + j) * D + v] * s.tail_decay[h * C + j] *
          static_cast<float>(s.k[(h * C + j) * D + d]);
    state[(sequence * heads + h) * D * D + v * D + d] = value;
  });
  RecordProfileEvent(queue, "gdn_chunk_state", state_event);
}
}

template <typename Input>
bool RunChunked(Queue& queue, Tensor& out, const Tensor& qi, const Tensor& ki,
                const Tensor& vi, const Tensor& g, const Tensor& beta, Tensor& state,
                const Tensor& qsl, const GdnArgs& args, int heads, int key_heads,
                int sequences, int tokens) {
  return WithGdnWorkspace(queue, WorkspaceBytes, [&](void* storage) {
    ChunkScratch<Input> scratch(storage, heads);
    WithOutput(queue, out, {&qi, &ki, &vi, &g, &beta}, [&](Tensor& target) {
      const auto* offsets = static_cast<const int32_t*>(qsl.data);
      for (int sequence = 0; sequence < sequences; ++sequence)
        for (int base = 0; base < tokens; base += C) {
          Prepare(queue, scratch, View(qi), View(ki), View(g), static_cast<float*>(state.data),
                  offsets, sequence, base, heads, key_heads);
          Dots(queue, scratch, heads);
          System(queue, scratch, View(beta), offsets, sequence, base, heads, args.scale);
          ComputeWU(queue, scratch, View(vi), View(beta), offsets, sequence, base, heads);
          OutputState(queue, scratch, View(target), static_cast<float*>(state.data), offsets,
                      sequence, base, heads, args.scale);
        }
    });
  });
}

bool GdnChunkedPrefillKernel(Queue& queue, Tensor& out, const Tensor& qi, const Tensor& ki,
                            const Tensor& vi, const Tensor& g, const Tensor& beta, Tensor& state,
                            const Tensor& qsl, const GdnArgs& args) {
  const auto device = NativeQueue(queue).get_device();
  const int heads = state.shape[1], sequences = state.shape[0], key_heads = qi.shape[1], tokens = qi.shape[0];
  if (state.shape[2] != D || state.shape[3] != D || heads < 1 || heads > MaxHeads ||
      sequences < 1 || sequences > 4 || tokens < 1 || tokens > 6656 ||
      (qi.dtype != DType::kBF16 && qi.dtype != DType::kF16) ||
      ki.dtype != qi.dtype || vi.dtype != qi.dtype ||
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
  if (qi.dtype == DType::kF16)
    return RunChunked<sycl::half>(queue, out, qi, ki, vi, g, beta, state, qsl,
                                  args, heads, key_heads, sequences, tokens);
  return RunChunked<BF>(queue, out, qi, ki, vi, g, beta, state, qsl,
                        args, heads, key_heads, sequences, tokens);
}
}  // namespace vt::xpu
