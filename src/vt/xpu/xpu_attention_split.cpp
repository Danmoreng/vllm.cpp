#include "xpu_common.h"
#include "xpu_fp8.h"
#include "xpu_kernels.h"
#include <limits>

namespace vt::xpu {
bool PagedAttentionSplitKernel(Queue& q, Tensor& out, const Tensor& query, const Tensor& key_cache,
                               const Tensor& value_cache, const Tensor& block_table,
                               const Tensor& seq_lens, const Tensor& query_start_loc,
                               const PagedAttentionArgs& args) {
  constexpr int SG = 16, Components = 16, Reuse = 2;
  constexpr size_t Workspace = 16 * 1024 * 1024;
  const int64_t tokens = query.shape[0], heads = query.shape[1], dim = query.shape[2];
  const int64_t kvheads = key_cache.shape[2], ratio = heads / kvheads;
  const int64_t page = key_cache.shape[1], capacity = block_table.shape[1] * page;
  const auto sizes = NativeQueue(q).get_device().get_info<sycl::info::device::sub_group_sizes>();
  if (tokens < 1 || tokens > 20 || heads > 24 || dim > SG * Components ||
      std::find(sizes.begin(), sizes.end(), SG) == sizes.end()) return false;
  const int64_t parts = std::min(int64_t{32}, std::max(int64_t{1}, (capacity + 255) / 256));
  const int64_t stride = dim + 2, pairs = (ratio + Reuse - 1) / Reuse;
  if (tokens * heads * parts * stride * sizeof(float) > Workspace) return false;
  const View qs(query), kc(key_cache), vc(value_cache), dst(out), bt(block_table);
  const auto* table = static_cast<const int32_t*>(bt.data);
  const auto* lengths = static_cast<const int32_t*>(seq_lens.data);
  const auto* offsets = static_cast<const int32_t*>(query_start_loc.data);
  const float scale = args.scale, cap = args.logits_soft_cap, kscale = args.k_scale, vscale = args.v_scale;
  const bool causal = args.causal;
  const int64_t left = args.window_size ? args.window_size->left : -1;
  const int64_t right = args.window_size ? args.window_size->right : -1;
  return WithAttentionWorkspace(q, Workspace, [&](void* storage) {
    auto* partial = static_cast<float*>(storage);
    const int64_t groups = tokens * kvheads * pairs * parts;
    const auto partial_event = NativeQueue(q).parallel_for(sycl::nd_range<1>(groups * SG, SG),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      const auto sg = item.get_sub_group();
      const int lane = item.get_local_id(0);
      int64_t group = item.get_group(0), part = group % parts; group /= parts;
      const int64_t pair = group % pairs; group /= pairs;
      const int64_t kh = group % kvheads, token = group / kvheads, head = kh * ratio + pair * Reuse;
      int64_t request = 0;
      while (token >= offsets[request + 1]) ++request;
      const int64_t position = lengths[request] - (offsets[request + 1] - offsets[request]) + token - offsets[request];
      const int64_t first = left < 0 ? 0 : sycl::max(int64_t{0}, position - left);
      int64_t last = causal ? position : int64_t(lengths[request]) - 1;
      if (right >= 0) last = sycl::min(last, position + right);
      const int64_t span = (last - first + 1 + parts - 1) / parts;
      const int64_t begin = first + part * span, end = sycl::min(last + 1, begin + span);
      float qv[Reuse][Components] = {}, acc[Reuse][Components] = {};
      float maximum[Reuse], denominator[Reuse] = {};
      #pragma unroll
      for (int r = 0; r < Reuse; ++r) {
        maximum[r] = -std::numeric_limits<float>::infinity();
        #pragma unroll
        for (int c = 0; c < Components; ++c)
          if (pair * Reuse + r < ratio && lane + c * SG < dim)
            qv[r][c] = Load(qs, (token * heads + head + r) * dim + lane + c * SG);
      }
      for (int64_t key = begin; key < end; ++key) {
        const int64_t block = table[request * bt.stride[0] + (key / page) * bt.stride[1]];
        const int64_t kb = block * kc.stride[0] + (key % page) * kc.stride[1] + kh * kc.stride[2];
        const int64_t vb = block * vc.stride[0] + (key % page) * vc.stride[1] + kh * vc.stride[2];
        float kval[Components], vval[Components];
        #pragma unroll
        for (int c = 0; c < Components; ++c) {
          kval[c] = lane + c * SG < dim ? LoadKV(kc, kb + lane + c * SG, kscale) : 0;
          vval[c] = lane + c * SG < dim ? LoadKV(vc, vb + lane + c * SG, vscale) : 0;
        }
        // Adjacent GQA heads share every K/V load; tail heads are masked.
        #pragma unroll
        for (int r = 0; r < Reuse; ++r) {
          float dot = 0;
          #pragma unroll
          for (int c = 0; c < Components; ++c) dot += qv[r][c] * kval[c];
          float score = sycl::reduce_over_group(sg, dot, sycl::plus<float>()) * scale;
          if (cap > 0) score = cap * sycl::tanh(score / cap);
          const float next = sycl::max(maximum[r], score);
          const float old = sycl::exp(maximum[r] - next), p = sycl::exp(score - next);
          denominator[r] = denominator[r] * old + p;
          #pragma unroll
          for (int c = 0; c < Components; ++c) acc[r][c] = acc[r][c] * old + p * vval[c];
          maximum[r] = next;
        }
      }
      #pragma unroll
      for (int r = 0; r < Reuse; ++r) if (pair * Reuse + r < ratio) {
        auto* result = partial + ((token * heads + head + r) * parts + part) * stride;
        if (lane == 0) { result[dim] = maximum[r]; result[dim + 1] = denominator[r]; }
        #pragma unroll
        for (int c = 0; c < Components; ++c) if (lane + c * SG < dim) result[lane + c * SG] = acc[r][c];
      }
    });
    RecordProfileEvent(q, "attention_split_partial", partial_event);
    const auto reduce_event = NativeQueue(q).parallel_for(sycl::range<1>(tokens * heads * dim), [=](sycl::id<1> item) {
      const int64_t row = item[0] / dim, d = item[0] % dim;
      const auto* src = partial + row * parts * stride;
      float maximum = -std::numeric_limits<float>::infinity();
      for (int p = 0; p < parts; ++p) if (src[p * stride + dim + 1] > 0)
        maximum = sycl::max(maximum, src[p * stride + dim]);
      float sum = 0, value = 0;
      for (int p = 0; p < parts; ++p) if (src[p * stride + dim + 1] > 0) {
        const float factor = sycl::exp(src[p * stride + dim] - maximum);
        sum += factor * src[p * stride + dim + 1];
        value += factor * src[p * stride + d];
      }
      Store(dst, item[0], value / sum);
    });
    RecordProfileEvent(q, "attention_split_reduce", reduce_event);
  });
}
}  // namespace vt::xpu
