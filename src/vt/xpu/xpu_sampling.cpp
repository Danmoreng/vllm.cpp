#include "xpu_common.h"
#include "xpu_kernels.h"
#include <limits>
namespace vt::xpu {
void GreedyArgmaxKernel(Queue& q, Tensor& token_ids, const Tensor& logits) {
  TraceOpTensors(OpId::kGreedyArgmax, q, {&token_ids, &logits});
  if (logits.shape[0] == 0) return;
  const auto rows = static_cast<size_t>(logits.shape[0]);
  const auto vocab = logits.shape[1];
  const auto* values = static_cast<const float*>(logits.data);
  auto* ids = static_cast<int64_t*>(token_ids.data);
  constexpr size_t lanes = 128;
  NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float> best_values(lanes, h);
    sycl::local_accessor<int64_t> best_ids(lanes, h);
    h.parallel_for(sycl::nd_range<1>(rows * lanes, lanes), [=](sycl::nd_item<1> item) {
      const auto row = item.get_group(0), lane = item.get_local_id(0);
      // Match VT's strict '>' scan: NaN at index 0 wins; later NaNs are ignored.
      if (sycl::isnan(values[row * vocab])) { if (lane == 0) ids[row] = 0; return; }
      float value = -std::numeric_limits<float>::infinity();
      int64_t index = std::numeric_limits<int64_t>::max();
      for (int64_t i = lane; i < vocab; i += lanes) {
        const float v = values[row * vocab + i];
        if (v > value || (v == value && i < index)) { value = v; index = i; }
      }
      best_values[lane] = value; best_ids[lane] = index;
      item.barrier(sycl::access::fence_space::local_space);
      for (size_t step = lanes / 2; step; step /= 2) {
        if (lane < step) {
          const float v = best_values[lane + step]; const auto i = best_ids[lane + step];
          if (v > best_values[lane] || (v == best_values[lane] && i < best_ids[lane])) {
            best_values[lane] = v; best_ids[lane] = i;
          }
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      if (lane == 0) ids[row] = best_ids[0];
    });
  });
}
}  // namespace vt::xpu
