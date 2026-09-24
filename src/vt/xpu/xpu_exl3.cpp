// Correctness path following cpu_exl3_kernels.cpp: Had128 -> compressed GEMM
// -> Had128. Weights remain packed throughout; only activations use scratch.
#include "xpu_common.h"
#include "xpu_exl3.h"
#include "xpu_kernels.h"

namespace vt::xpu {
namespace {
constexpr float kInvSqrt128 = 0.088388347648f;
float FlipSign(float x) { return sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(x) ^ 0x80000000u); }

void Had(Queue& q, Tensor& out, const Tensor& in, const Tensor* pre, const Tensor* post, float scale) {
  if (in.Numel() == 0) return;
  const View src(in), dst(out);
  const auto cols = in.shape[1];
  const auto* pre_values = pre ? static_cast<const sycl::half*>(pre->data) : nullptr;
  const auto* post_values = post ? static_cast<const sycl::half*>(post->data) : nullptr;
  const auto blocks = static_cast<size_t>(in.Numel() / 128);
  NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float> shared(128, h);
    h.parallel_for(sycl::nd_range<1>(blocks * 32, 32), [=](sycl::nd_item<1> item) {
      const int lane = item.get_local_id(0);
      const int64_t base = item.get_group(0) * 128;
      const int64_t col = base % cols;
      float v[4];
      for (int j = 0; j < 4; ++j) {
        v[j] = Load(src, base + lane * 4 + j);
        if (pre_values) v[j] = Round(src.dtype, v[j] * static_cast<float>(pre_values[col + lane * 4 + j]));
      }
      const float s0 = v[0] + v[1], d0 = v[0] - v[1];
      const float s1 = v[2] + v[3], d1 = v[2] - v[3];
      v[0] = s0 + s1; v[1] = d0 + d1; v[2] = s0 - s1; v[3] = d0 - d1;
      for (int j = 0; j < 4; ++j) shared[lane * 4 + j] = v[j];
      item.barrier(sycl::access::fence_space::local_space);
      for (int step = 1; step < 32; step *= 2) {
        for (int j = 0; j < 4; ++j)
          v[j] = ((lane & step) ? FlipSign(v[j]) : v[j]) + shared[(lane ^ step) * 4 + j];
        item.barrier(sycl::access::fence_space::local_space);
        for (int j = 0; j < 4; ++j) shared[lane * 4 + j] = v[j];
        item.barrier(sycl::access::fence_space::local_space);
      }
      for (int j = 0; j < 4; ++j) {
        // Half output rounds BEFORE multiplying svh, then again at the store.
        float value = Round(dst.dtype, v[j] * scale);
        if (post_values) value *= static_cast<float>(post_values[col + lane * 4 + j]);
        Store(dst, base + lane * 4 + j, value);
      }
    });
  });
}
}

void Exl3HadR128Kernel(Queue& q, Tensor& out, const Tensor& in, const Exl3HadArgs& args) {
  TraceOpTensors(OpId::kExl3HadR128, q, {&out, &in, args.pre_scale, args.post_scale});
  const auto* sc = args.pre_scale ? args.pre_scale : args.post_scale;
  VT_CHECK(!sc || sc->IsContiguous(), "XPU EXL3 Had128 requires contiguous scale");
  // Exact in-place is safe: each workgroup reads its whole block before stores.
  if (out.data == in.data && (!sc || !Overlap(out, *sc))) {
    Had(q, out, in, args.pre_scale, args.post_scale, args.scale * kInvSqrt128);
  } else {
    WithOutput(q, out, {&in, sc}, [&](Tensor& target) {
      Had(q, target, in, args.pre_scale, args.post_scale, args.scale * kInvSqrt128);
    });
  }
}

void Exl3GemmKernel(Queue& q, Tensor& out, const Tensor& in, const Tensor& trellis,
                    const Tensor& suh, const Tensor& svh, Tensor& in_had, const Exl3GemmArgs& args) {
  TraceOpTensors(OpId::kExl3Gemm, q, {&out, &in, &trellis, &suh, &svh, &in_had});
  const int64_t m = in.shape[0], k = in.shape[1], n = out.shape[1];
  if (m == 0 || k == 0 || n == 0) return;
  VT_CHECK(!Overlap(in_had, trellis) && !Overlap(in_had, suh) && !Overlap(in_had, svh),
           "XPU EXL3 input scratch may not overwrite weights");
  VT_CHECK(!Overlap(out, trellis) && !Overlap(out, suh) && !Overlap(out, svh),
           "XPU EXL3 output may not overwrite weights");
  const auto raw_tensor = Tensor::Contiguous(nullptr, DType::kF32, q.device, {m, n});
  Scratch storage(q.device, raw_tensor.Bytes());
  auto raw = raw_tensor; raw.data = storage.data;
  Exl3HadR128Kernel(q, in_had, in, Exl3HadArgs{&suh, nullptr, 1.0f});
  const auto* ah = static_cast<const sycl::half*>(in_had.data);
  const auto* packed = static_cast<const unsigned char*>(trellis.data);
  auto* result = static_cast<float*>(raw.data);
  const int bits = args.bits, cb = args.codebook;
  NativeQueue(q).parallel_for(sycl::range<1>(m * n), [=](sycl::id<1> item) {
    const int64_t row = item[0] / n, col = item[0] % n;
    float acc = 0;
    for (int64_t inner = 0; inner < k; ++inner) {
      const float value = static_cast<float>(ah[row * k + inner]);
      if (value == 0.0f) continue;  // same signed-zero behavior as CPU
      const auto* tile = packed + ((inner / 16) * (n / 16) + col / 16) * (32 * bits);
      const int t = exl3::Fragment(inner % 16, col % 16);
      acc += value * exl3::Decode(exl3::Codeword(tile, bits, t), cb);
    }
    result[item[0]] = acc;
  });
  Had(q, out, raw, nullptr, &svh, kInvSqrt128);
  // Free is conservative and waits all owned queues; surface errors explicitly.
  GetBackend(q.device).Synchronize(q);
}
}  // namespace vt::xpu
