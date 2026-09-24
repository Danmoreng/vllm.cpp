#include "xpu_common.h"
#include "xpu_kernels.h"
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
    // One work-item per row preserves the CPU's sequential FP32 variance sum.
    // PR02 correctness reference; parallel reductions can be tuned separately.
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(x.shape[0]), [=](sycl::id<1> item) {
      const auto row = item[0];
      float sum = 0;
      for (int64_t col = 0; col < width; ++col) {
        float value = Load(src, row * src.stride[0] + col);
        if (has_res) {
          const auto off = row * res.stride[0] + col;
          value = Round(res.dtype, value + Load(res, off));
          Store(res, off, value);
        }
        sum += value * value;
      }
      const float scale = 1.0f / sycl::sqrt(sum / static_cast<float>(width) + eps);
      for (int64_t col = 0; col < width; ++col) {
        const float value = has_res ? Load(res, row * res.stride[0] + col)
                                    : Load(src, row * src.stride[0] + col);
        const float weight_value = gemma ? 1.0f + Load(w, col) : Load(w, col);
        Store(dst, row * dst.stride[0] + col, value * scale * weight_value);
      }
    });
    RecordProfileEvent(q, "rms_norm", event);
  });
}
}  // namespace vt::xpu
