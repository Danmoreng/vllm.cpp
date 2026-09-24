#include "xpu_common.h"
#include "xpu_kernels.h"
#include <cstdlib>
#include <string_view>

namespace vt::xpu {
namespace {
void CheckOffsets(Queue& q, const Tensor& qsl, int64_t sequences, int64_t tokens) {
  const auto* offsets = static_cast<const int32_t*>(qsl.data);
  CheckDeviceMetadata(q, [=] {
    if (offsets[0] != 0 || offsets[sequences] != tokens) return false;
    for (int64_t i = 0; i < sequences; ++i)
      if (offsets[i] < 0 || offsets[i + 1] < offsets[i] || offsets[i + 1] > tokens) return false;
    return true;
  }, "XPU GDN/conv invalid sequence offsets", {&qsl});
}
void CheckSlots(Queue& q, const Tensor& indices, int64_t slots, bool allow_null, bool unique) {
  const auto* ids = static_cast<const int32_t*>(indices.data);
  const auto rows = indices.Numel();
  CheckDeviceMetadata(q, [=] {
    for (int64_t i = 0; i < rows; ++i) {
      if (ids[i] >= slots || (!allow_null && ids[i] < 0)) return false;
      if (unique && ids[i] >= 0)
        for (int64_t j = 0; j < i; ++j) if (ids[j] == ids[i]) return false;
    }
    return true;
  }, "XPU GDN/conv invalid or duplicate state slot", {&indices});
}
inline bool Initial(View flags, int64_t row) {
  return flags.dtype == DType::kI8 ? static_cast<const int8_t*>(flags.data)[row] != 0
                                 : static_cast<const int32_t*>(flags.data)[row] != 0;
}
inline float Silu(float x) { return x / (1.0f + sycl::exp(-x)); }
}

void CausalConv1dFwdKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                           const Tensor* bias, Tensor& state, const Tensor& qsl,
                           const Tensor& initial, const CausalConv1dArgs& args) {
  TraceXpuOp(OpId::kCausalConv1dFwd, q, {&out, &x, &weight, bias, &state, &qsl, &initial});
  const auto sequences = state.shape[0], channels = x.shape[1], taps = weight.shape[1];
  const auto state_width = state.shape[2], width = taps - 1;
  FloatTensor(state);
  VT_CHECK(!Overlap(state, x) && !Overlap(state, out) && !Overlap(state, weight)
               && (!bias || !Overlap(state, *bias)), "XPU conv state must have separate storage");
  CheckOffsets(q, qsl, sequences, x.shape[0]);
  WithOutput(q, out, {&x, &weight, bias}, [&](Tensor& target) {
    const View src(x), dst(target), w(weight), b(bias ? *bias : weight), flags(initial);
    const auto* offsets = static_cast<const int32_t*>(qsl.data);
    const View cache(state);
    const bool has_bias = bias != nullptr, activation = args.silu_activation;
    NativeQueue(q).parallel_for(sycl::range<1>(sequences * channels), [=](sycl::id<1> item) {
      const int64_t seq = item[0] / channels, channel = item[0] % channels;
      const int64_t begin = offsets[seq], length = offsets[seq + 1] - begin;
      const auto old = (seq * channels + channel) * state_width;
      const bool keep = Initial(flags, seq);
      for (int64_t t = 0; t < length; ++t) {
        float acc = has_bias ? Load(b, channel) : 0.0f;
        for (int64_t j = 0; j < taps; ++j) {
          const auto token = t - width + j;
          const float value = token >= 0 ? Load(src, (begin + token) * src.stride[0] + channel)
                                        : keep ? Load(cache, old + width + token) : 0.0f;
          acc += Load(w, channel * taps + j) * value;
        }
        Store(dst, (begin + t) * channels + channel, activation ? Silu(acc) : acc);
      }
      // For a short sequence, old entries are read above the write cursor.
      // History always contains raw activations, never the convolved output.
      for (int64_t j = 0; j < width; ++j) {
        const auto token = length - width + j;
        const float value = token >= 0 ? Load(src, (begin + token) * src.stride[0] + channel)
                                      : keep ? Load(cache, old + width + token) : 0.0f;
        Store(cache, old + j, value);
      }
    });
  });
}

void CausalConv1dUpdateKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                              const Tensor* bias, Tensor& state, const Tensor* indices,
                              const CausalConv1dArgs& args) {
  TraceXpuOp(OpId::kCausalConv1dUpdate, q, {&out, &x, &weight, bias, &state, indices});
  const auto batch = x.shape[0], channels = x.shape[1], taps = weight.shape[1];
  const auto state_width = state.shape[2], width = taps - 1;
  FloatTensor(state);
  VT_CHECK(!Overlap(state, x) && !Overlap(state, out) && !Overlap(state, weight)
               && (!bias || !Overlap(state, *bias)), "XPU conv state must have separate storage");
  if (indices) CheckSlots(q, *indices, state.shape[0], true, true);
  WithOutput(q, out, {&x, &weight, bias, indices}, [&](Tensor& target) {
    const View src(x), dst(target), w(weight), b(bias ? *bias : weight);
    const View cache(state);
    const auto* ids = indices ? static_cast<const int32_t*>(indices->data) : nullptr;
    const bool has_bias = bias != nullptr, activation = args.silu_activation;
    NativeQueue(q).parallel_for(sycl::range<1>(batch * channels), [=](sycl::id<1> item) {
      const int64_t row = item[0] / channels, channel = item[0] % channels;
      const int64_t slot = ids ? ids[row] : row;
      if (slot < 0) return;  // VT conv contract leaves null-slot output unchanged.
      const auto history = (slot * channels + channel) * state_width;
      const float current = Load(src, row * src.stride[0] + channel);
      float acc = has_bias ? Load(b, channel) : 0.0f;
      for (int64_t j = 0; j < width; ++j) acc += Load(w, channel * taps + j) * Load(cache, history + j);
      acc += Load(w, channel * taps + width) * current;
      Store(dst, row * channels + channel, activation ? Silu(acc) : acc);
      for (int64_t j = 0; j + 1 < width; ++j) Store(cache, history + j, Load(cache, history + j + 1));
      if (width) Store(cache, history + width - 1, current);
    });
  }, true);
}
void GdnPostConvKernel(Queue& q, Tensor& qo, Tensor& ko, Tensor& vo, Tensor& go, Tensor& bo,
                        const Tensor& conv, const Tensor& araw, const Tensor& braw,
                        const Tensor& alog, const Tensor& bias, const L2NormArgs& args) {
  TraceXpuOp(OpId::kGdnPostConv, q, {&qo, &ko, &vo, &go, &bo, &conv, &araw, &braw, &alog, &bias});
  const int64_t tokens = conv.shape[0], hk = qo.shape[1], dk = qo.shape[2];
  const int64_t hv = vo.shape[1], dv = vo.shape[2], keys = hk * dk, values = hv * dv;
  const View src(conv), qs(qo), ks(ko), vs(vo), gs(go), bs(bo), a(araw), b(braw), al(alog), dt(bias);
  const float eps = args.eps;
  NativeQueue(q).parallel_for(sycl::range<1>(tokens * (hk + hv)), [=](sycl::id<1> item) {
    const int64_t token = item[0] / (hk + hv), head = item[0] % (hk + hv);
    const int64_t base = token * (2 * keys + values);
    if (head < hk) {
      float qss = 0, kss = 0;
      for (int64_t i = 0; i < dk; ++i) {
        const float qv = Load(src, base + head * dk + i), kv = Load(src, base + keys + head * dk + i);
        qss += qv * qv; kss += kv * kv;
      }
      const float qi = 1.0f / sycl::sqrt(qss + eps), ki = 1.0f / sycl::sqrt(kss + eps);
      for (int64_t i = 0; i < dk; ++i) {
        Store(qs, (token * hk + head) * dk + i, Load(src, base + head * dk + i) * qi);
        Store(ks, (token * hk + head) * dk + i, Load(src, base + keys + head * dk + i) * ki);
      }
    } else {
      const auto h = head - hk;
      for (int64_t i = 0; i < dv; ++i)
        Store(vs, (token * hv + h) * dv + i, Load(src, base + 2 * keys + h * dv + i));
      const float x = Load(a, token * a.stride[0] + h) + Load(dt, h);
      const float softplus = x > 20.0f ? x : sycl::log1p(sycl::exp(x));
      Store(gs, token * hv + h, -sycl::exp(Load(al, h)) * softplus);
      Store(bs, token * hv + h, 1.0f / (1.0f + sycl::exp(-Load(b, token * b.stride[0] + h))));
    }
  });
}

namespace {
void Recurrence(Queue& q, Tensor& out, const Tensor& qi, const Tensor& ki, const Tensor& vi,
                  const Tensor& g, const Tensor& beta, Tensor& state, const Tensor* qsl,
                  const Tensor* indices, float scale) {
  VT_CHECK(state.dtype == DType::kF32, "XPU GDN recurrence requires F32 state");
  for (const auto* t : std::initializer_list<const Tensor*>{&out, &qi, &ki, &vi, &g, &beta})
    VT_CHECK(!Overlap(state, *t), "XPU GDN state must have separate storage");
  const bool decode = qsl == nullptr;
  const auto rows = decode ? qi.shape[0] : state.shape[0];
  const auto hv = state.shape[1], dv = state.shape[2], dk = state.shape[3], hk = qi.shape[1];
  if (qsl) CheckOffsets(q, *qsl, rows, qi.shape[0]);
  if (indices) CheckSlots(q, *indices, state.shape[0], true, true);
  WithOutput(q, out, {&qi, &ki, &vi, &g, &beta}, [&](Tensor& target) {
    const View dst(target), qs(qi), ks(ki), vs(vi), gs(g), bs(beta);
    const auto* offsets = qsl ? static_cast<const int32_t*>(qsl->data) : nullptr;
    const auto* ids = indices ? static_cast<const int32_t*>(indices->data) : nullptr;
    auto* cache = static_cast<float*>(state.data);
    // One work-item owns one value row of S. Tokens remain sequential; no
    // extra q/k normalization or gate transformation occurs inside recurrence.
    NativeQueue(q).parallel_for(sycl::range<1>(rows * hv * dv), [=](sycl::id<1> item) {
      const int64_t row = item[0] / (hv * dv), head = (item[0] / dv) % hv, value = item[0] % dv;
      const int64_t kh = head / (hv / hk), slot = ids ? ids[row] : row;
      if (slot < 0) { Store(dst, (row * hv + head) * dv + value, 0.0f); return; }
      auto* s = cache + ((slot * hv + head) * dv + value) * dk;
      const int64_t first = decode ? row : offsets[row], end = decode ? row + 1 : offsets[row + 1];
      for (int64_t token = first; token < end; ++token) {
        const int64_t kbase = (token * hk + kh) * dk;
        const float decay = sycl::exp(Load(gs, token * hv + head));
        float prediction = 0;
        for (int64_t j = 0; j < dk; ++j) {
          s[j] *= decay;
          prediction += s[j] * Load(ks, kbase + j);
        }
        const float delta = (Load(vs, (token * hv + head) * dv + value) - prediction) * Load(bs, token * hv + head);
        float output = 0;
        for (int64_t j = 0; j < dk; ++j) {
          s[j] += delta * Load(ks, kbase + j);
          output += s[j] * (Load(qs, kbase + j) * scale);
        }
        Store(dst, (token * hv + head) * dv + value, output);
      }
    });
  });
}
}
void GdnPrefillKernel(Queue& q, Tensor& out, const Tensor& qi, const Tensor& ki, const Tensor& vi,
                       const Tensor& g, const Tensor& beta, Tensor& state, const Tensor& qsl,
                       const GdnArgs& args) {
  TraceXpuOp(OpId::kGdnPrefill, q, {&out, &qi, &ki, &vi, &g, &beta, &state, &qsl});
  enum class Mode { kAuto, kReference, kChunked };
  static const Mode mode = [] {
    const char* value = std::getenv("VT_XPU_GDN_PREFILL");
    const std::string_view name = value ? value : "auto";
    VT_CHECK(name == "auto" || name == "reference" || name == "chunked", "Invalid VT_XPU_GDN_PREFILL");
    return name == "auto" ? Mode::kAuto : name == "chunked" ? Mode::kChunked : Mode::kReference;
  }();
  bool chunked = mode == Mode::kChunked;
  if (mode == Mode::kAuto && qi.shape[1] == 16 && state.shape[1] == 48) {
    const auto device = NativeQueue(q).get_device();
    chunked = std::string_view(__VERSION__) == "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)" &&
        device.get_info<sycl::info::device::driver_version>() == "1.17.39758+10" &&
        device.get_platform().get_info<sycl::info::platform::version>() == "1.17";
  }
  if (chunked && GdnChunkedPrefillEnabled() && qi.shape[0] >= 64 &&
      GdnChunkedPrefillKernel(q, out, qi, ki, vi, g, beta, state, qsl, args)) return;
  Recurrence(q, out, qi, ki, vi, g, beta, state, &qsl, nullptr, args.scale);
}
void GdnDecodeKernel(Queue& q, Tensor& out, const Tensor& qi, const Tensor& ki, const Tensor& vi,
                      const Tensor& g, const Tensor& beta, Tensor& state, const Tensor* indices,
                      const GdnArgs& args) {
  TraceXpuOp(OpId::kGdnDecode, q, {&out, &qi, &ki, &vi, &g, &beta, &state, indices});
  Recurrence(q, out, qi, ki, vi, g, beta, state, nullptr, indices, args.scale);
}
void RmsNormGatedKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                         const Tensor& weight, const RmsNormGatedArgs& args) {
  TraceXpuOp(OpId::kRmsNormGated, q, {&out, &x, &gate, &weight});
  const auto width = x.shape[x.rank - 1], rows = x.Numel() / width;
  const auto group = gate.rank == 3 ? gate.shape[1] : 1;
  const auto eps = args.eps; const bool sigmoid = args.sigmoid_gate;
  WithOutput(q, out, {&x, &gate, &weight}, [&](Tensor& target) {
    const View src(x), dst(target), z(gate), w(weight);
    NativeQueue(q).parallel_for(sycl::range<1>(rows), [=](sycl::id<1> item) {
      const auto row = item[0];
      float sum = 0;
      for (int64_t j = 0; j < width; ++j) { const float v = Load(src, row * width + j); sum += v * v; }
      const float inv = 1.0f / sycl::sqrt(sum / static_cast<float>(width) + eps);
      const auto gbase = (row / group) * z.stride[0] + (row % group) * width;
      for (int64_t j = 0; j < width; ++j) {
        const float value = Load(z, gbase + j);
        const float act = sigmoid ? 1.0f / (1.0f + sycl::exp(-value)) : Silu(value);
        Store(dst, row * width + j, Load(src, row * width + j) * inv * Load(w, j) * act);
      }
    });
  });
}
void GdnStateGatherKernel(Queue& q, Tensor& working, const Tensor& cache,
                           const Tensor& indices, const Tensor* initial) {
  TraceXpuOp(OpId::kGdnStateGather, q, {&working, &cache, &indices, initial});
  CheckSlots(q, indices, cache.shape[0], false, false);
  const auto inner = working.shape[working.rank - 1], physical = cache.shape[cache.rank - 1];
  const auto row_size = working.Numel() / indices.Numel(), mid = row_size / inner;
  WithOutput(q, working, {&cache, &indices, initial}, [&](Tensor& target) {
    const View dst(target), src(cache), flags(initial ? *initial : indices);
    const auto* ids = static_cast<const int32_t*>(indices.data); const bool has_flags = initial != nullptr;
    NativeQueue(q).parallel_for(sycl::range<1>(working.Numel()), [=](sycl::id<1> item) {
      const auto row = item[0] / row_size, col = item[0] % row_size;
      const auto from = (ids[row] * mid + col / inner) * physical + col % inner;
      Store(dst, item[0], !has_flags || Initial(flags, row) ? Load(src, from) : 0.0f);
    });
  });
}
void GdnStateScatterKernel(Queue& q, Tensor& cache, const Tensor& working, const Tensor& indices) {
  TraceXpuOp(OpId::kGdnStateScatter, q, {&cache, &working, &indices});
  CheckSlots(q, indices, cache.shape[0], false, true);
  const auto inner = working.shape[working.rank - 1], physical = cache.shape[cache.rank - 1];
  const auto row_size = working.Numel() / indices.Numel(), mid = row_size / inner;
  WithOutput(q, cache, {&working, &indices}, [&](Tensor& target) {
    const View dst(target), src(working); const auto* ids = static_cast<const int32_t*>(indices.data);
    NativeQueue(q).parallel_for(sycl::range<1>(working.Numel()), [=](sycl::id<1> item) {
      const auto row = item[0] / row_size, col = item[0] % row_size;
      const auto to = (ids[row] * mid + col / inner) * physical + col % inner;
      Store(dst, to, Load(src, item[0]));
    });
  }, true);
}
}  // namespace vt::xpu
