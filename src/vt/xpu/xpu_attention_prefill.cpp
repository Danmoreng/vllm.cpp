#include "xpu_common.h"
#include "xpu_fp8.h"
#include "xpu_kernels.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <limits>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace vt::xpu {
namespace {
template<class Fn> struct PrefillGrf256 {
  Fn fn;
  void operator()(sycl::nd_item<1> item) const { fn(item); }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size<256>,
        sycl::ext::oneapi::experimental::sub_group_size<16>};
  }
};
}
// Native adaptation of the donor's tiled QK -> online softmax -> PV schedule.
// VT pages may be strided unbind views; no donor block-size/layout assumption.
// Q16/K32, eight SG16s. 37,056 bytes of SLM; F32 running softmax and output.
// 256 GRFs avoid the spill traffic of the 128-GRF compilation on B70.
// F32 queries/probabilities use FP16 high+residual XMX operands, avoiding the
// extra single-FP16 rounding at both sites. K/V share one staging buffer.
bool PagedAttentionPrefillKernel(Queue& q, Tensor& out, const Tensor& query, const Tensor& key_cache,
                                 const Tensor& value_cache, const Tensor& block_table,
                                 const Tensor& seq_lens, const Tensor& query_start_loc,
                                 const PagedAttentionArgs& args) {
  namespace mx = sycl::ext::oneapi::experimental::matrix;
  namespace imx = sycl::ext::intel::experimental::matrix;
  constexpr int Q = 16, K = 32, D = 256, WG = 128;
  const auto device = NativeQueue(q).get_device();
  const int64_t tokens = query.shape[0], heads = query.shape[1], ratio = heads / key_cache.shape[2];
  const int64_t requests = seq_lens.Numel(), page = key_cache.shape[1];
  if (tokens < Q || query.shape[2] != D || requests > 4 || heads > 24 ||
      !device.has(sycl::aspect::ext_intel_matrix) ||
      !device.has(sycl::aspect::ext_intel_device_id) ||
      device.get_info<sycl::ext::intel::info::device::device_id>() != 57891 ||
      device.get_info<sycl::info::device::local_mem_size>() < 37056) return false;
  const View qs(query), kc(key_cache), vc(value_cache), dst(out), bt(block_table);
  const auto* table = static_cast<const int32_t*>(bt.data);
  const auto* lengths = static_cast<const int32_t*>(seq_lens.data);
  const auto* offsets = static_cast<const int32_t*>(query_start_loc.data);
  const float scale = args.scale, cap = args.logits_soft_cap, kscale = args.k_scale, vscale = args.v_scale;
  const bool causal = args.causal;
  const int64_t left = args.window_size ? args.window_size->left : -1;
  const int64_t right = args.window_size ? args.window_size->right : -1;
  // One extra partial tile per sequence. The device maps this compact grid,
  // so empty requests and changing query lengths need no host mirror.
  const int64_t tiles = (tokens + Q - 1) / Q + requests - 1;
  Scratch status(q.device, sizeof(int));
  auto* unsafe = static_cast<int*>(status.data);
  const auto reset_event = NativeQueue(q).memset(unsafe, 0, sizeof(int));
  RecordProfileEvent(q, "attention_prefill_status_reset", reset_event);
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<sycl::half> query_tile(Q * D, h), kv(K * D, h), prob(Q * K, h);
    sycl::local_accessor<sycl::half> query_low(Q * D, h), prob_low(Q * K, h);
    sycl::local_accessor<float> scores(Q * K, h), maxval(Q, h), denom(Q, h), oldscale(Q, h);
    h.parallel_for(sycl::nd_range<1>(tiles * heads * WG, WG),
        PrefillGrf256{[=](sycl::nd_item<1> item) {
      const auto sg = item.get_sub_group();
      const int tid = item.get_local_id(0), subgroup = tid / 16;
      const int64_t head = item.get_group(0) % heads;
      int64_t tile = item.get_group(0) / heads, request = 0;
      while (request < requests) {
        const int64_t count = (offsets[request + 1] - offsets[request] + Q - 1) / Q;
        if (tile < count) break;
        tile -= count; ++request;
      }
      if (request == requests) return;
      const int64_t token = offsets[request] + tile * Q;
      const int rows = sycl::min(int64_t{Q}, int64_t(offsets[request + 1]) - token);
      const int64_t position = lengths[request] - (offsets[request + 1] - offsets[request]) + token - offsets[request];
      const int64_t first = left < 0 ? 0 : sycl::max(int64_t{0}, position - left);
      int64_t end = causal ? position + rows : lengths[request];
      if (right >= 0) end = sycl::min(end, position + rows + right);
      auto narrow = [&](float value) {
        if (!sycl::isfinite(value) || sycl::fabs(value) > 65504.0f) {
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
              sycl::access::address_space::global_space>(*unsafe).store(1);
          return sycl::half(0);
        }
        return sycl::half(value);
      };
      for (int i = tid; i < Q * D; i += WG) {
        const float value = i / D < rows ? Load(qs, ((token + i / D) * heads + head) * D + i % D) : 0;
        const auto high = narrow(value);
        query_tile[i] = high; query_low[i] = narrow(value - float(high));
      }
      if (tid < Q) { maxval[tid] = -std::numeric_limits<float>::infinity(); denom[tid] = 0; }
      mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> acc0, acc1;
      mx::joint_matrix_fill(sg, acc0, 0.0f); mx::joint_matrix_fill(sg, acc1, 0.0f);
      for (int64_t base = first; base < end; base += K) {
        for (int i = tid; i < K * D; i += WG) {
          const int key = i % K, d = i / K;
          float value = 0;
          if (base + key < end) {
            const int64_t index = base + key, block = table[request * bt.stride[0] + (index / page) * bt.stride[1]];
            value = LoadKV(kc, block * kc.stride[0] + (index % page) * kc.stride[1] + (head / ratio) * D + d, kscale);
          }
          kv[i] = narrow(value); // transposed K: [D,K]
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (subgroup < K / 16) {
          mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 16, 16, mx::layout::row_major> a;
          mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> b;
          mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> dot;
          mx::joint_matrix_fill(sg, dot, 0.0f);
          for (int d = 0; d < D; d += 16) {
            mx::joint_matrix_load(sg, a, query_tile.template get_multi_ptr<sycl::access::decorated::no>() + d, D);
            mx::joint_matrix_load(sg, b, kv.template get_multi_ptr<sycl::access::decorated::no>() + d * K + subgroup * 16, K);
            mx::joint_matrix_mad(sg, dot, a, b, dot);
            mx::joint_matrix_load(sg, a, query_low.template get_multi_ptr<sycl::access::decorated::no>() + d, D);
            mx::joint_matrix_mad(sg, dot, a, b, dot);
          }
          mx::joint_matrix_store(sg, dot, scores.template get_multi_ptr<sycl::access::decorated::no>() + subgroup * 16, K, mx::layout::row_major);
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (tid < Q) {
          float maximum = maxval[tid];
          for (int k = 0; k < K; ++k) {
            const int64_t key = base + k, p = position + tid;
            const bool valid = tid < rows && key < end && (!causal || key <= p) &&
                (left < 0 || key >= p - left) && (right < 0 || key <= p + right);
            float value = scores[tid * K + k] * scale;
            if (cap > 0) value = cap * sycl::tanh(value / cap);
            value = valid ? value : -std::numeric_limits<float>::infinity();
            scores[tid * K + k] = value; maximum = sycl::max(maximum, value);
          }
          const float old = sycl::isfinite(maximum) ? sycl::exp(maxval[tid] - maximum) : 0;
          float sum = 0;
          for (int k = 0; k < K; ++k) {
            const float value = sycl::isfinite(scores[tid * K + k]) ? sycl::exp(scores[tid * K + k] - maximum) : 0;
            sum += value; prob[tid * K + k] = sycl::half(value);
            prob_low[tid * K + k] = sycl::half(value - float(sycl::half(value)));
          }
          denom[tid] = denom[tid] * old + sum; maxval[tid] = maximum; oldscale[tid] = old;
        }
        for (int i = tid; i < K * D; i += WG) {
          const int key = i / D, d = i % D;
          float value = 0;
          if (base + key < end) {
            const int64_t index = base + key, block = table[request * bt.stride[0] + (index / page) * bt.stride[1]];
            value = LoadKV(vc, block * vc.stride[0] + (index % page) * vc.stride[1] + (head / ratio) * D + d, vscale);
          }
          kv[i] = narrow(value); // shared stage now holds V: [K,D]
        }
        item.barrier(sycl::access::fence_space::local_space);
        imx::joint_matrix_apply(sg, acc0, [&](float& value, size_t row, size_t) { value *= oldscale[row]; });
        imx::joint_matrix_apply(sg, acc1, [&](float& value, size_t row, size_t) { value *= oldscale[row]; });
        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 16, 16, mx::layout::row_major> a;
        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> b;
        for (int k = 0; k < K; k += 16) {
          mx::joint_matrix_load(sg, a, prob.template get_multi_ptr<sycl::access::decorated::no>() + k, K);
          mx::joint_matrix_load(sg, b, kv.template get_multi_ptr<sycl::access::decorated::no>() + k * D + subgroup * 32, D);
          mx::joint_matrix_mad(sg, acc0, a, b, acc0);
          mx::joint_matrix_load(sg, b, kv.template get_multi_ptr<sycl::access::decorated::no>() + k * D + subgroup * 32 + 16, D);
          mx::joint_matrix_mad(sg, acc1, a, b, acc1);
          mx::joint_matrix_load(sg, a, prob_low.template get_multi_ptr<sycl::access::decorated::no>() + k, K);
          mx::joint_matrix_mad(sg, acc1, a, b, acc1);
          mx::joint_matrix_load(sg, b, kv.template get_multi_ptr<sycl::access::decorated::no>() + k * D + subgroup * 32, D);
          mx::joint_matrix_mad(sg, acc0, a, b, acc0);
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      imx::joint_matrix_apply(sg, acc0, [&](float& value, size_t row, size_t col) {
        if (row < size_t(rows)) Store(dst, ((token + row) * heads + head) * D + subgroup * 32 + col, value / denom[row]);
      });
      imx::joint_matrix_apply(sg, acc1, [&](float& value, size_t row, size_t col) {
        if (row < size_t(rows)) Store(dst, ((token + row) * heads + head) * D + subgroup * 32 + 16 + col, value / denom[row]);
      });
    }});
  });
  RecordProfileEvent(q, "attention_prefill", event);
  int invalid = 0;
  GetBackend(q.device).Copy(q, &invalid, unsafe, sizeof(int));
  // If FP16 could not represent an operand, the caller re-runs its generic
  // F32 kernel. The original inputs are intact, including output aliases.
  return invalid == 0;
}
}  // namespace vt::xpu
