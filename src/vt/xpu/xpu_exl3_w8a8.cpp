// Native wrappers adapted from exl3xpu c59d944 csrc/exl3_ops.sycl.
// Copyright (c) 2026 0xSero. MIT license: third_party/exl3xpu/LICENSE.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
#include "xpu_common.h"
#include "xpu_kernels.h"
#include "xpu_gptq4.h"
#include "vt/exl3_grouped.h"
#include "vt/exl3_w8a8_panel_plan.h"
#pragma clang diagnostic pop
#include <exl3xpu/exl3_esimd.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace vt::xpu {
namespace {
template<class Kernel>
void Launch(Queue& q, int64_t count, int local, Kernel kernel, const char* stage) {
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>((count + local - 1) / local * local),
                                    sycl::range<1>(local)), kernel);
  });
  RecordProfileEvent(q, stage, event);
}

// The donor's float->INT8 conversion has no defined NaN/Inf contract. Reject
// those inputs, including overflow at either FP16 Hadamard rounding boundary,
// before writing any output. Bit tests remain valid in this fast-math TU.
struct ValidateRows {
  ::exl3::HadInQ8Kernel<sycl::half> input;
  uint32_t* valid;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::esimd;
    const int kb_n = input.Kdim / 128;
    const size_t id = it.get_global_id(0);
    if (id >= size_t(input.S) * input.M * kb_n) return;
    const int kb = int(id % kb_n), m = int(id / kb_n % input.M);
    const int g = int(id / kb_n / input.M);
    const auto xi = block_load<uint16_t, 128>(
        reinterpret_cast<const uint16_t*>(input.x) + size_t(m) * input.Kdim + kb * 128);
    const auto su = block_load<uint16_t, 128>(
        reinterpret_cast<const uint16_t*>(input.suh) + size_t(g) * input.Kdim + kb * 128);
    bool finite = !((xi & 0x7c00u) == 0x7c00u).any() &&
                  !((su & 0x7c00u) == 0x7c00u).any();
    // Check the product rounding separately: a subsequent max reduction is
    // not a valid detector for a NaN that some lanes may ignore.
    auto product_f32 =
        convert<float>(block_load<sycl::half, 128>(input.x + size_t(m) * input.Kdim + kb * 128)) *
        convert<float>(block_load<sycl::half, 128>(input.suh + size_t(g) * input.Kdim + kb * 128));
    // 65520 is the round-to-nearest FP16 overflow boundary. In this fast-math
    // TU a half round-trip can be optimized away, so validate the F32 bits
    // before narrowing as well as checking the resulting half bits.
    finite &= !((product_f32.bit_cast_view<uint32_t>() & 0x7fffffffu) >= 0x477ff000u).any();
    simd<sycl::half, 128> product = convert<sycl::half>(product_f32);
    finite &= !((product.bit_cast_view<uint16_t>() & 0x7c00u) == 0x7c00u).any();
    // Mirror blk's transform, checking the second boundary before its half
    // conversion for the same reason. The producer itself remains unchanged.
    auto transformed = convert<float>(product);
    ::exl3::fwht128(transformed);
    transformed *= ::exl3::kRsqrt128;
    finite &= !((transformed.bit_cast_view<uint32_t>() & 0x7fffffffu) >= 0x477ff000u).any();
    transformed = convert<float>(convert<sycl::half>(transformed));
    finite &= !((transformed.bit_cast_view<uint32_t>() & 0x7f800000u) == 0x7f800000u).any();
    if (!finite)
      atomic_update<atomic_op::store, uint32_t, 1>(valid, simd<uint32_t, 1>(0u),
                                                  simd<uint32_t, 1>(0u));
  }
};

struct ValidateOutputScale {
  const uint16_t* bits;
  int n;
  uint32_t* valid;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::esimd;
    const size_t block = it.get_global_id(0);
    if (block >= size_t(n / 128)) return;
    const auto values = block_load<uint16_t, 128>(bits + block * 128);
    if (((values & 0x7c00u) == 0x7c00u).any())
      atomic_update<atomic_op::store, uint32_t, 1>(valid, simd<uint32_t, 1>(0u),
                                                  simd<uint32_t, 1>(0u));
  }
};

template<int Bits>
void Reconstruct(Queue& q, const Tensor& tr, Tensor& panel, int k, int n,
                 int first_column, int columns) {
  ::exl3::ReconstructKernel<Bits, 2, 8, int8_t> kernel{
      static_cast<const uint32_t*>(tr.data), static_cast<int8_t*>(panel.data),
      n / 16, first_column / 16, columns, columns / 128, k / 16, 127.0f / 3.453125f};
  Launch(q, int64_t(k / 16) * (columns / 128), 8, kernel,
         "exl3_w8a8_weight_reconstruct");
}
}  // namespace

void Exl3GroupedW8A8Kernel(Queue& q, Tensor& out, const Tensor& in, const Tensor& tr,
    const Tensor& suh, const Tensor& svh, const Tensor& shard,
    Tensor& workspace, Tensor& panel, const Exl3GroupedLinearArgs& args) {
#ifndef VLLM_CPP_XPU_GPTQ4
  VT_CHECK(false, "EXL3 W8A8 requires the pinned oneDNN 3.13 build");
#else
  TraceXpuOp(OpId::kExl3GroupedW8A8, q,
             {&out, &in, &tr, &suh, &svh, &shard, &workspace, &panel});
  const int m = int(in.shape[0]), k = int(in.shape[1]), n = int(out.shape[1]);
  const int groups = int(suh.shape[0]);
  const auto plan = PlanExl3W8A8(m, k, n, groups, args.bits, args.w8a8_panel_columns);
  for (const Tensor* t : std::initializer_list<const Tensor*>{
           &out, &in, &tr, &suh, &svh, &shard, &workspace, &panel})
    VT_CHECK(reinterpret_cast<uintptr_t>(t->data) % 16 == 0,
             "EXL3 W8A8 storage requires 16-byte alignment");
  for (const Tensor* dst : {&out, &workspace, &panel})
    for (const Tensor* src : {&in, &tr, &suh, &svh, &shard})
      VT_CHECK(!Overlap(*dst, *src), "EXL3 W8A8 writable storage may not overlap operands");
  VT_CHECK(!Overlap(out, workspace) && !Overlap(out, panel) && !Overlap(workspace, panel),
           "EXL3 W8A8 output and scratch may not overlap");

  // Eager initial implementation: mapping is small, and the completion wait
  // also ensures that the panel's previous use has retired before reuse.
  std::vector<int32_t> mapping(n / 128);
  auto& backend = GetBackend(q.device);
  backend.Copy(q, mapping.data(), shard.data, mapping.size() * sizeof(int32_t));
  backend.Synchronize(q);
  const auto panels = PlanExl3W8A8Panels(mapping, groups, args.w8a8_panel_columns);
  const auto* sv_bits = static_cast<const uint16_t*>(svh.data);

  auto* bytes = static_cast<uint8_t*>(workspace.data);
  auto* xq = reinterpret_cast<int8_t*>(bytes + plan.activation_offset);
  auto* sx = reinterpret_cast<float*>(bytes + plan.row_scale_offset);
  auto* y = reinterpret_cast<sycl::half*>(bytes + plan.intermediate_offset);
  auto* sw = reinterpret_cast<float*>(bytes + plan.weight_scale_offset);
  // Both GPU checks use spare words in the existing 64-byte scale region.
  // Read their results together before output/panel/activation writes. This
  // avoids a separate metadata allocation, readback and retirement drain.
  auto* valid = reinterpret_cast<uint32_t*>(sw + 1);
  NativeQueue(q).single_task([=] { valid[0] = 1; valid[1] = 1; });
  Launch(q, n / 128, 8, ValidateOutputScale{sv_bits, n, valid + 1},
         "exl3_w8a8_validate_svh");
  ::exl3::HadInQ8Kernel<sycl::half> input{
      static_cast<const sycl::half*>(in.data), static_cast<const sycl::half*>(suh.data),
      xq, sx, m, k, groups, k, plan.padded_rows};
  Launch(q, int64_t(groups) * m * (k / 128), 8, ValidateRows{input, valid},
         "exl3_w8a8_validate_rows");
  std::array<uint32_t, 2> finite{};
  backend.Copy(q, finite.data(), valid, sizeof(finite));
  backend.Synchronize(q);
  // Preserve the previous error priority if both checks fail.
  VT_CHECK(finite[1], "EXL3 W8A8 requires finite svh");
  VT_CHECK(finite[0], "EXL3 W8A8 requires finite inputs and finite FP16 transformed rows");

  for (const Tensor* t : {&out, &workspace, &panel}) RecordGraphWrite(q, t->data, Span(*t));
  // Poison/stale padding is never consumed by GEMM. The active producer writes
  // every byte of each real row. For zero rows its exact fallback is scale1.
  NativeQueue(q).memset(xq, 0, size_t(groups) * plan.padded_rows * k);
  NativeQueue(q).parallel_for(sycl::range<1>(size_t(groups) * plan.padded_rows),
                            [=](sycl::id<1> i) { sx[i[0]] = 1.0f; });
  NativeQueue(q).single_task([=] { *sw = 3.453125f / 127.0f; });
  if (k / 128 <= 48) {
    ::exl3::HadInQ8WgKernel<sycl::half, 8, 6> had{
        input.x, input.suh, xq, sx, m, k, groups, k, plan.padded_rows};
    Launch(q, int64_t(groups) * m * 8, 8, had, "exl3_w8a8_input_quantize");
  } else if (k / 128 <= 144) {
    ::exl3::HadInQ8WgKernel<sycl::half, 16, 9> had{
        input.x, input.suh, xq, sx, m, k, groups, k, plan.padded_rows};
    Launch(q, int64_t(groups) * m * 16, 16, had, "exl3_w8a8_input_quantize");
  } else {
    Launch(q, int64_t(groups) * m, 8, input, "exl3_w8a8_input_quantize");
  }

  const auto weight_scale = Tensor::Contiguous(sw, DType::kF32, q.device, {1});
  for (const auto& part : panels) {
    if (args.bits == 4) Reconstruct<4>(q, tr, panel, k, n, part.first_column, part.columns);
    else Reconstruct<6>(q, tr, panel, k, n, part.first_column, part.columns);
    const int group = part.source_group;
    const auto a = Tensor::Contiguous(xq + size_t(group) * plan.padded_rows * k,
        DType::kI8, q.device, {plan.padded_rows, k});
    const auto scales = Tensor::Contiguous(sx + size_t(group) * plan.padded_rows,
        DType::kF32, q.device, {plan.padded_rows});
    auto dst = Tensor::Contiguous(y + part.first_column, DType::kF16, q.device,
                                  {plan.padded_rows, part.columns});
    dst.stride[0] = n;
    // Reconstruction writes a compact K*width prefix, including short tails.
    const auto weight_view = Tensor::Contiguous(panel.data, DType::kI8, q.device,
                                                {k, part.columns});
    Exl3W8A8Matmul(q, dst, a, weight_view, scales, weight_scale);
  }
  // oneDNN rounds to F16 before the output Hadamard, as the production donor
  // does. The fallback I32 HadOutQ8 recipe is a different arithmetic route.
  ::exl3::HadOutKernel<sycl::half, sycl::half> tail{
      y, static_cast<const sycl::half*>(svh.data), static_cast<sycl::half*>(out.data),
      m, n, 1, n};
  Launch(q, int64_t(m) * (n / 128), 8, tail, "exl3_w8a8_output_hadamard");
  const char* setting = std::getenv("VT_XPU_EXL3_TRACE");
  if (setting && std::string_view(setting) == "1") {
    const nlohmann::json event = {{"event", "exl3_w8a8_dispatch"}, {"queue_id", q.id},
        {"matrix", args.debug_name ? args.debug_name : ""}, {"m", m}, {"padded_m", plan.padded_rows},
        {"k", k}, {"n", n}, {"groups", groups}, {"bits", args.bits},
        {"leaf", "signed_int8_onednn_f16_hadamard"}, {"intermediate_stride", n},
        {"weight_panel_bytes", plan.weight_panel_bytes}, {"workspace_bytes", plan.workspace_bytes},
        {"weight_panel_columns", plan.weight_panel_columns}, {"panel_submissions", panels.size()},
        {"input_dtype", Name(in.dtype)}, {"output_dtype", Name(out.dtype)}};
    std::fprintf(stderr, "EXL3_W8A8_DISPATCH %s\n", event.dump().c_str());
  }
#endif
}
}  // namespace vt::xpu
