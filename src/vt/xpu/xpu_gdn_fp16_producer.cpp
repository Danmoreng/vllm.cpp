#include "xpu_common.h"
#include "xpu_kernels.h"
#include "chunk_gated_delta_rule_fp16_producer.hpp"
#include "gated_delta_rule_fp16_decode.hpp"
#include "vt/gdn_fp16_plan.h"

#include <cmath>

namespace vt::xpu {
namespace {
constexpr int Hk = 16, Hv = 48, D = 128;
using T = cutlass::half_t;
}  // namespace
void GdnPrefillRawGateKernel(Queue& queue, Tensor& out, const Tensor& qi,
    const Tensor& ki, const Tensor& vi, const Tensor& raw_a, const Tensor& beta,
    const Tensor& a_log, const Tensor& dt_bias, Tensor& state, const Tensor& qsl,
    const GdnArgs&) {
  auto& native = NativeQueue(queue);
  const auto plan = PlanGdnFp16C1(qi.shape[0]);
  const int tokens = plan.tokens, capacity = plan.capacity;
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
  CheckDeviceMetadata(queue, [=] { return offsets[0] == 0 && offsets[1] == tokens; },
                      "gdn_prefill_raw_gate: offsets must cover the exact logical C1 length", {&qsl});

  const bool submitted = WithGdnNativeWorkspace(queue, plan.bytes, [&](void* storage) {
    auto* cursor = static_cast<char*>(storage);
    auto* q = reinterpret_cast<T*>(cursor + plan.q_offset);
    auto* k = reinterpret_cast<T*>(cursor + plan.k_offset);
    auto* v = reinterpret_cast<T*>(cursor + plan.v_offset);
    auto* A = reinterpret_cast<T*>(cursor + plan.a_matrix_offset);
    auto* w = reinterpret_cast<T*>(cursor + plan.w_offset);
    auto* u = reinterpret_cast<T*>(cursor + plan.u_offset);
    auto* a = reinterpret_cast<float*>(cursor + plan.raw_a_offset);
    auto* b = reinterpret_cast<float*>(cursor + plan.beta_offset);
    auto* bias = reinterpret_cast<sycl::half*>(cursor + plan.bias_offset);
    auto* index = reinterpret_cast<int*>(cursor + plan.index_offset);
    auto* initial = reinterpret_cast<bool*>(cursor + plan.initial_offset);
    // The producer reads its zero-padded physical capacity at tile tails.
    native.memset(storage, 0, plan.bytes);
    native.memcpy(q, qi.data, size_t(tokens) * Hk * D * sizeof(T));
    native.memcpy(k, ki.data, size_t(tokens) * Hk * D * sizeof(T));
    native.memcpy(v, vi.data, size_t(tokens) * Hv * D * sizeof(T));
    const View av(raw_a), bv(beta), dv(dt_bias);
    native.parallel_for(sycl::range<1>(size_t(Hv) * capacity), [=](sycl::id<1> id) {
      const int h = id[0] / capacity, t = id[0] % capacity;
      if (t < tokens) {
        a[id] = Load(av, t * av.stride[0] + h);
        b[id] = Load(bv, t * bv.stride[0] + h);
      }
      if (t == 0) bias[h] = sycl::half(Load(dv, h));
    });
    // The caller has prepared this working state: zero for a fresh request,
    // or gathered persistent FP32 values for continuation. Always consume it.
    native.single_task([=] { index[0] = 0; initial[0] = true; });
    gdn::fp16_producer::kernel_launcher<T, float>(native,
        static_cast<T*>(out.data), q, k, v, A, w, u, b, a,
        static_cast<const float*>(a_log.data), reinterpret_cast<const T*>(bias),
        static_cast<float*>(state.data), Hv * D * D, offsets, index, initial,
        nullptr, 1, capacity, Hk, D, Hv, D);
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
  // The backend reserves the maximum supported completion-owned capacity
  // even when decode runs first. These small metadata offsets remain fixed.
  const bool submitted = WithGdnNativeWorkspace(queue, 256, [&](void* storage) {
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
