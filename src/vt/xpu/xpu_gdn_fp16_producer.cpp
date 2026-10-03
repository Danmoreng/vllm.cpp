#include "xpu_common.h"
#include "xpu_kernels.h"
#include "chunk_gated_delta_rule_fp16_producer.hpp"
#include "gated_delta_rule_fp16_decode.hpp"

#include <cmath>

namespace vt::xpu {
namespace {
constexpr int Tokens = 128, Capacity = Tokens + 63, Hk = 16, Hv = 48, D = 128;
using T = cutlass::half_t;
constexpr size_t QBytes = size_t(Capacity) * Hk * D * sizeof(T);
constexpr size_t VBytes = size_t(Capacity) * Hv * D * sizeof(T);
constexpr size_t ABytes = size_t(Hv) * Capacity * 64 * sizeof(T);
constexpr size_t WBytes = size_t(Hv) * Capacity * D * sizeof(T);
constexpr size_t GateBytes = size_t(Hv) * Capacity * sizeof(float);
constexpr size_t Bytes = 2 * QBytes + VBytes + ABytes + 2 * WBytes +
    2 * GateBytes + 128 + 128;
}  // namespace
void GdnPrefillRawGateKernel(Queue& queue, Tensor& out, const Tensor& qi,
    const Tensor& ki, const Tensor& vi, const Tensor& raw_a, const Tensor& beta,
    const Tensor& a_log, const Tensor& dt_bias, Tensor& state, const Tensor& qsl,
    const GdnArgs&) {
  auto& native = NativeQueue(queue);
  const auto device = native.get_device();
  VT_CHECK(device.has(sycl::aspect::ext_intel_device_id) &&
               device.get_info<sycl::ext::intel::info::device::device_id>() == 57891 &&
               device.has(sycl::aspect::ext_intel_matrix),
           "gdn_prefill_raw_gate: requires the B70 matrix device");
  for (const Tensor* input : {&qi, &ki, &vi, &raw_a, &beta, &a_log, &dt_bias, &qsl}) {
    VT_CHECK(!Overlap(out, *input) && !Overlap(state, *input),
             "gdn_prefill_raw_gate: output/state overlaps an input");
  }
  VT_CHECK(!Overlap(out, state), "gdn_prefill_raw_gate: output overlaps state");
  TraceXpuOp(OpId::kGdnPrefillRawGate, queue,
             {&out, &qi, &ki, &vi, &raw_a, &beta, &a_log, &dt_bias, &state, &qsl});
  RecordGraphWrite(queue, state.data, Span(state));
  const auto* offsets = static_cast<const int32_t*>(qsl.data);
  CheckDeviceMetadata(queue, [=] { return offsets[0] == 0 && offsets[1] == Tokens; },
                      "gdn_prefill_raw_gate: requires offsets [0,128]", {&qsl});

  const bool submitted = WithGdnNativeWorkspace(queue, Bytes, [&](void* storage) {
    auto* cursor = static_cast<char*>(storage);
    auto take = [&](size_t bytes) {
      void* p = cursor;
      cursor += (bytes + 63) & ~size_t(63);
      return p;
    };
    auto* q = static_cast<T*>(take(QBytes));
    auto* k = static_cast<T*>(take(QBytes));
    auto* v = static_cast<T*>(take(VBytes));
    auto* A = static_cast<T*>(take(ABytes));
    auto* w = static_cast<T*>(take(WBytes));
    auto* u = static_cast<T*>(take(WBytes));
    auto* a = static_cast<float*>(take(GateBytes));
    auto* b = static_cast<float*>(take(GateBytes));
    auto* bias = static_cast<sycl::half*>(take(Hv * sizeof(T)));
    auto* index = static_cast<int*>(take(sizeof(int)));
    auto* initial = reinterpret_cast<bool*>(cursor);
    // The producer reads its zero-padded physical capacity at tile tails.
    native.memset(storage, 0, Bytes);
    native.memcpy(q, qi.data, size_t(Tokens) * Hk * D * sizeof(T));
    native.memcpy(k, ki.data, size_t(Tokens) * Hk * D * sizeof(T));
    native.memcpy(v, vi.data, size_t(Tokens) * Hv * D * sizeof(T));
    const View av(raw_a), bv(beta), dv(dt_bias);
    native.parallel_for(sycl::range<1>(size_t(Hv) * Capacity), [=](sycl::id<1> id) {
      const int h = id[0] / Capacity, t = id[0] % Capacity;
      if (t < Tokens) {
        a[id] = Load(av, t * av.stride[0] + h);
        b[id] = Load(bv, t * bv.stride[0] + h);
      }
      if (t == 0) bias[h] = sycl::half(Load(dv, h));
    });
    native.single_task([=] { index[0] = 0; initial[0] = true; });
    gdn::fp16_producer::kernel_launcher<T, float>(native,
        static_cast<T*>(out.data), q, k, v, A, w, u, b, a,
        static_cast<const float*>(a_log.data), reinterpret_cast<const T*>(bias),
        static_cast<float*>(state.data), Hv * D * D, offsets, index, initial,
        nullptr, 1, Capacity, Hk, D, Hv, D);
  });
  VT_CHECK(submitted, "gdn_prefill_raw_gate: native workspace unavailable");
}
void GdnPackedDecodeKernel(Queue& queue, Tensor& out, const Tensor& mixed,
    const Tensor& raw_a, const Tensor& raw_b, const Tensor& a_log,
    const Tensor& dt_bias, Tensor& state, const Tensor& indices,
    const GdnArgs& args) {
  auto& native = NativeQueue(queue);
  const auto device = native.get_device();
  VT_CHECK(device.has(sycl::aspect::ext_intel_device_id) &&
               device.get_info<sycl::ext::intel::info::device::device_id>() == 57891,
           "gdn_packed_decode: FP16 producer requires the B70");
  VT_CHECK(mixed.dtype == DType::kF16 && mixed.shape[0] == 1 &&
               mixed.shape[1] == (2 * Hk + Hv) * D &&
               raw_a.dtype == DType::kF16 && raw_b.dtype == DType::kF16 &&
               out.dtype == DType::kF16 && state.dtype == DType::kF32 &&
               a_log.dtype == DType::kF32 && dt_bias.dtype == DType::kF32 &&
               state.shape[1] == Hv && state.shape[2] == D && state.shape[3] == D &&
               state.shape[0] > 0 &&
               std::abs(args.scale - 1.0f / std::sqrt(float(D))) < 1e-6f,
           "gdn_packed_decode: qualified FP16 producer geometry is C1/Hk16/Hv48/D128/F32 state");
  for (const Tensor* input : {&mixed, &raw_a, &raw_b, &a_log, &dt_bias, &indices}) {
    VT_CHECK(!Overlap(out, *input) && !Overlap(state, *input),
             "gdn_packed_decode: output/state overlaps an input");
  }
  VT_CHECK(!Overlap(out, state), "gdn_packed_decode: output overlaps state");
  TraceXpuOp(OpId::kGdnPackedDecode, queue,
             {&out, &mixed, &raw_a, &raw_b, &a_log, &dt_bias, &state, &indices});
  RecordGraphWrite(queue, state.data, Span(state));
  const auto* index = static_cast<const int32_t*>(indices.data);
  const int64_t slots = state.shape[0];
  CheckDeviceMetadata(queue, [=] { return index[0] >= 0 && index[0] < slots; },
                      "gdn_packed_decode: invalid active state slot", {&indices});
  // Reserve the same completion-owned capacity as P128, including when
  // decode runs first. This workspace cannot grow after first allocation.
  const bool submitted = WithGdnNativeWorkspace(queue, Bytes, [&](void* storage) {
    auto* bias = static_cast<sycl::half*>(storage);
    auto* offsets = reinterpret_cast<int*>(static_cast<char*>(storage) + 128);
    const View bv(dt_bias);
    native.parallel_for(sycl::range<1>(Hv), [=](sycl::id<1> h) {
      bias[h] = sycl::half(Load(bv, h));
    });
    native.single_task([=] { offsets[0] = 0; offsets[1] = 1; });
    const auto* input = static_cast<const sycl::half*>(mixed.data);
    using Kernel = gdn::fp16_decode_producer::gated_delta_rule_kernel<sycl::half, float, 4>;
    native.parallel_for(Kernel::get_nd_range(1, Hv, D),
        Kernel(static_cast<sycl::half*>(out.data), input, input + Hk * D,
            input + 2 * Hk * D, static_cast<const sycl::half*>(raw_b.data),
            static_cast<const sycl::half*>(raw_a.data),
            static_cast<const float*>(a_log.data), bias,
            static_cast<float*>(state.data), Hv * D * D, offsets, nullptr,
            index, nullptr, nullptr, 1, 1, Hk, D, Hv, D));
  });
  VT_CHECK(submitted, "gdn_packed_decode: native workspace unavailable");
}
}  // namespace vt::xpu
