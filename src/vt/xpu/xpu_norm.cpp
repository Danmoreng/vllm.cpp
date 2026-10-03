#include "xpu_common.h"
#include "xpu_kernels.h"
#include "xpu_qk_norm.h"
namespace vt::xpu {
void RmsNormKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                   const RmsNormArgs& args, Tensor* residual) {
  TraceXpuOp(OpId::kRmsNorm, q, {&out, &x, &weight, residual});
  FloatTensor(out); FloatTensor(x); FloatTensor(weight);
  VT_CHECK(x.shape[1] > 0, "XPU RMSNorm requires positive hidden width");
  if (residual) {
    FloatTensor(*residual);
    VT_CHECK(!Overlap(*residual, weight), "XPU RMSNorm residual may not alias weights");
    VT_CHECK(!Overlap(*residual, x) || residual->data == x.data, "XPU RMSNorm partial residual/input alias");
  }
  WithOutput(q, out, {&x, &weight, residual}, [&](Tensor& target) {
    const View dst(target), src(x), w(weight), res(residual ? *residual : x);
    const auto width = x.shape[1]; const bool has_res = residual != nullptr;
    const auto eps = args.eps; const bool gemma = args.gemma;
    if (args.qk_fp16) {
      const int64_t rows = x.shape[0];
      const auto sizes = NativeQueue(q).get_device().get_info<sycl::info::device::sub_group_sizes>();
      if (std::find(sizes.begin(), sizes.end(), 16) != sizes.end()) {
        constexpr int lanes = 16, workgroup = 128;
        const auto global = ((rows * lanes + workgroup - 1) / workgroup) * workgroup;
        const auto event = NativeQueue(q).parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(workgroup)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const int64_t row = item.get_global_linear_id() / lanes;
          if (row >= rows) return;
          const int lane = item.get_local_linear_id() % lanes;
          const int64_t base = row * src.stride[0];
          const float mean = ProducerQkMean256(src, base, item.get_sub_group(), lane, rows);
          const float inverse = sycl::rsqrt(mean + eps);
          for (int col = lane; col < 256; col += lanes)
            Store(dst, row * dst.stride[0] + col,
                  ProducerQkNormValue(Load(src, base + col), inverse, Load(w, col), true));
        });
        RecordProfileEvent(q, "rms_norm_qk_fp16_subgroup", event);
      } else {
        const auto event = NativeQueue(q).parallel_for(sycl::range<1>(rows), [=](sycl::id<1> item) {
          const int64_t row = item[0], base = row * src.stride[0];
          const float mean = ProducerQkMean256Scalar(src, base, rows);
          const float inverse = sycl::rsqrt(mean + eps);
          for (int col = 0; col < 256; ++col)
            Store(dst, row * dst.stride[0] + col,
                  ProducerQkNormValue(Load(src, base + col), inverse, Load(w, col), true));
        });
        RecordProfileEvent(q, "rms_norm_qk_fp16_scalar", event);
      }
      return;
    }
    // Pinned EXL3 FP16 GemmaRMSNorm uses the native producer IR:
    // normalize x.float()+res.float(), but return that sum narrowed as res.
    // Do not normalize a reread of the FP16 store. Other dtype/weight modes
    // retain their existing residual-rounding contract.
    const bool fp32_sum = has_res && gemma && src.dtype == DType::kF16 &&
                          dst.dtype == DType::kF16 && res.dtype == DType::kF16;
    // Wide decode rows must distribute the reduction over a work-group. A
    // single work-item reading 5120 elements twice dominates B70 token time.
    if (width >= 256) {
      constexpr size_t kWorkGroup = 256;
      const auto event = NativeQueue(q).parallel_for(
          sycl::nd_range<1>(sycl::range<1>(x.shape[0] * kWorkGroup),
                            sycl::range<1>(kWorkGroup)),
          [=](sycl::nd_item<1> item) {
        const auto row = item.get_group(0);
        const auto lane = item.get_local_id(0);
        float partial = 0.0f;
        for (int64_t col = lane; col < width; col += kWorkGroup) {
          float value = Load(src, row * src.stride[0] + col);
          if (has_res) {
            const auto off = row * res.stride[0] + col;
            value += Load(res, off);
            if (!fp32_sum) {
              value = Round(res.dtype, value);
              Store(res, off, value);
            }
          }
          partial += value * value;
        }
        const float sum = sycl::reduce_over_group(
            item.get_group(), partial, sycl::plus<float>());
        const float scale = 1.0f / sycl::sqrt(sum / static_cast<float>(width) + eps);
        sycl::group_barrier(item.get_group());
        for (int64_t col = lane; col < width; col += kWorkGroup) {
          float value = has_res ? Load(res, row * res.stride[0] + col)
                               : Load(src, row * src.stride[0] + col);
          if (fp32_sum) {
            value += Load(src, row * src.stride[0] + col);
            Store(res, row * res.stride[0] + col, value);
          }
          const float weight_value = gemma ? 1.0f + Load(w, col) : Load(w, col);
          Store(dst, row * dst.stride[0] + col, value * scale * weight_value);
        }
      });
      RecordProfileEvent(q, "rms_norm", event);
      return;
    }
    // Keep the sequential reduction for narrow rows and its exact FP32 sum.
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(x.shape[0]), [=](sycl::id<1> item) {
      const auto row = item[0];
      float sum = 0;
      for (int64_t col = 0; col < width; ++col) {
        float value = Load(src, row * src.stride[0] + col);
        if (has_res) {
          const auto off = row * res.stride[0] + col;
          value += Load(res, off);
          if (!fp32_sum) {
            value = Round(res.dtype, value);
            Store(res, off, value);
          }
        }
        sum += value * value;
      }
      const float scale = 1.0f / sycl::sqrt(sum / static_cast<float>(width) + eps);
      for (int64_t col = 0; col < width; ++col) {
        float value = has_res ? Load(res, row * res.stride[0] + col)
                             : Load(src, row * src.stride[0] + col);
        if (fp32_sum) {
          value += Load(src, row * src.stride[0] + col);
          Store(res, row * res.stride[0] + col, value);
        }
        const float weight_value = gemma ? 1.0f + Load(w, col) : Load(w, col);
        Store(dst, row * dst.stride[0] + col, value * scale * weight_value);
      }
    });
    RecordProfileEvent(q, "rms_norm", event);
  });
}
}  // namespace vt::xpu
