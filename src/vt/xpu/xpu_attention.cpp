#include "xpu_common.h"
#include "xpu_kernels.h"
#include "xpu_fp8.h"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <cstdlib>
#include <string_view>

namespace vt::xpu {
namespace {
int64_t Position(View pos, int64_t token) {
  return pos.dtype == DType::kI32 ? static_cast<const int32_t*>(pos.data)[token]
                                : static_cast<const int64_t*>(pos.data)[token];
}
double Frequency(int64_t pair, const RopeArgs& args) {
  double freq = sycl::pow(double(args.base), -2.0 * pair / args.rotary_dim);
  if (!(args.llama3_scaling_factor > 0)) return freq;
  constexpr double two_pi = 6.283185307179586476925286766559;
  const double low = args.llama3_low_freq_factor, high = args.llama3_high_freq_factor;
  const double original = args.llama3_orig_max_position, factor = args.llama3_scaling_factor;
  const double wavelength = two_pi / freq;
  if (wavelength < original / high) return freq;
  if (wavelength > original / low) return freq / factor;
  const double smooth = low == high ? 0.0 : (original / wavelength - low) / (high - low);
  return (1.0 - smooth) * freq / factor + smooth * freq;
}
void Rotate(View dst, int64_t token, int64_t head, int64_t first, int64_t second, float c, float s) {
  const auto base = token * dst.stride[0] + head * dst.stride[1];
  const float x = Load(dst, base + first), y = Load(dst, base + second);
  Store(dst, base + first, x * c - y * s);
  Store(dst, base + second, x * s + y * c);
}
void CopyElement(View dst, int64_t to, View src, int64_t from) {
  if (src.dtype == DType::kF32)
    static_cast<uint32_t*>(dst.data)[to] = static_cast<const uint32_t*>(src.data)[from];
  else
    static_cast<uint16_t*>(dst.data)[to] = static_cast<const uint16_t*>(src.data)[from];
}
}
void AttnGateSplitKernel(Queue& q, Tensor& queries, Tensor& gates, const Tensor& packed) {
  TraceXpuOp(OpId::kAttnGateSplit, q, {&queries, &gates, &packed});
  VT_CHECK(!Overlap(queries, packed) && !Overlap(gates, packed) && !Overlap(queries, gates),
           "XPU Q/gate split requires separate output storage");
  const View src(packed), dst(queries), gate(gates);
  const auto width = queries.shape[2];
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(queries.Numel()), [=](sycl::id<1> item) {
    const int64_t head = item[0] / width, col = item[0] % width;
    Store(dst, item[0], Load(src, head * 2 * width + col));
    Store(gate, item[0], Load(src, head * 2 * width + width + col));
  });
  RecordProfileEvent(q, "attn_gate_split", event);
}
void AttnQkNormRopeGateKernel(Queue& q, Tensor& q_out, Tensor& k_out,
                             Tensor& gate_out, const Tensor& qgate,
                             const Tensor& kf, const Tensor& q_norm,
                             const Tensor& k_norm, const Tensor& cos_sin,
                             const RmsNormArgs& norm_args, const RopeArgs& rope_args) {
  TraceXpuOp(OpId::kAttnQkNormRopeGate, q,
             {&q_out, &k_out, &gate_out, &qgate, &kf, &q_norm, &k_norm, &cos_sin});
  for (const Tensor* dst : {&q_out, &k_out, &gate_out})
    for (const Tensor* src : {&qgate, &kf, &q_norm, &k_norm, &cos_sin})
      VT_CHECK(!Overlap(*dst, *src), "XPU attention preamble requires separate outputs");
  VT_CHECK(!Overlap(q_out, k_out) && !Overlap(q_out, gate_out) &&
               !Overlap(k_out, gate_out),
           "XPU attention preamble outputs must not overlap");
  const View qo(q_out), ko(k_out), go(gate_out), qs(qgate), ks(kf),
             qw(q_norm), kw(k_norm), cs(cos_sin);
  const int64_t tokens = q_out.shape[0], hq = q_out.shape[1],
                hk = k_out.shape[1], dim = q_out.shape[2];
  const int64_t half = rope_args.rotary_dim / 2, rot = rope_args.rotary_dim;
  const float eps = norm_args.eps;
  const bool gemma = norm_args.gemma;
  enum class PreambleMode { kAuto, kReference, kSubgroup };
  static const PreambleMode mode = [] {
    const char* value = std::getenv("VT_XPU_ATTN_PREAMBLE");
    const std::string_view name = value ? value : "auto";
    VT_CHECK(name == "auto" || name == "reference" || name == "subgroup",
             "Invalid VT_XPU_ATTN_PREAMBLE");
    return name == "reference" ? PreambleMode::kReference :
           name == "subgroup" ? PreambleMode::kSubgroup : PreambleMode::kAuto;
  }();
  const bool shape_ok = dim == 256 && rot > 0 && rot <= dim && half % 16 == 0 &&
      qgate.dtype == DType::kF16 && kf.dtype == DType::kF16 &&
      q_out.dtype == DType::kF16 && k_out.dtype == DType::kF16;
  bool subgroup = false;
  if (shape_ok && mode != PreambleMode::kReference) {
    const auto sizes = NativeQueue(q).get_device().get_info<sycl::info::device::sub_group_sizes>();
    subgroup = std::find(sizes.begin(), sizes.end(), 16) != sizes.end();
  }
  VT_CHECK(mode != PreambleMode::kSubgroup || subgroup,
           "VT_XPU_ATTN_PREAMBLE=subgroup requires F16 D256, RoPE half divisible by 16 and SG16");
  if (subgroup) {
    constexpr int lanes = 16, tile = 16, workgroup = 128;
    const auto heads = tokens * (hq + hk);
    const auto global = ((heads * lanes + workgroup - 1) / workgroup) * workgroup;
    const auto event = NativeQueue(q).parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(workgroup)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      const int64_t index = item.get_global_linear_id() / lanes;
      if (index >= heads) return;
      const int lane = item.get_local_linear_id() % lanes;
      const int64_t token = index / (hq + hk), head = index % (hq + hk);
      const bool query = head < hq;
      const int64_t local_head = query ? head : head - hq;
      const View src = query ? qs : ks, weight = query ? qw : kw, dst = query ? qo : ko;
      const int64_t src_base = token * src.stride[0] + local_head * (query ? 2 * dim : dim);
      const int64_t out_base = (token * (query ? hq : hk) + local_head) * dim;
      float values[tile], norm[tile], sum = 0.0f;
      for (int j = 0; j < tile; ++j) {
        values[j] = Load(src, src_base + lane + lanes * j);
        sum += values[j] * values[j];
      }
      sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<float>());
      const float inv = 1.0f / sycl::sqrt(sum / static_cast<float>(dim) + eps);
      for (int j = 0; j < tile; ++j) {
        const int col = lane + lanes * j;
        const float w = Load(weight, col);
        norm[j] = values[j] * inv * (gemma ? 1.0f + w : w);
      }
      for (int j = 0; j < tile; ++j) {
        const int col = lane + lanes * j;
        float value = norm[j];
        if (col < rot) {
          const int pair = col < half ? col : col - half;
          const int first = pair / lanes, second = (pair + half) / lanes;
          const float c = Load(cs, token * rot + pair);
          const float s = Load(cs, token * rot + half + pair);
          value = col < half ? norm[first] * c - norm[second] * s
                             : norm[first] * s + norm[second] * c;
        }
        Store(dst, out_base + col, value);
        if (query) Store(go, out_base + col, Load(qs, src_base + dim + col));
      }
    });
    RecordProfileEvent(q, "attn_qk_norm_rope_gate_subgroup", event);
    return;
  }
  const auto event = NativeQueue(q).parallel_for(
      sycl::range<1>(tokens * (hq + hk)), [=](sycl::id<1> item) {
    const int64_t token = item[0] / (hq + hk), head = item[0] % (hq + hk);
    const bool query = head < hq;
    const int64_t local_head = query ? head : head - hq;
    const View src = query ? qs : ks, weight = query ? qw : kw,
               dst = query ? qo : ko;
    const int64_t src_base = token * src.stride[0] +
        local_head * (query ? 2 * dim : dim);
    const int64_t out_base = (token * (query ? hq : hk) + local_head) * dim;
    float sum = 0;
    for (int64_t i = 0; i < dim; ++i) {
      const float value = Load(src, src_base + i);
      sum += value * value;
    }
    const float inv = 1.0f / sycl::sqrt(sum / static_cast<float>(dim) + eps);
    for (int64_t i = 0; i < dim; ++i) {
      const auto norm = [&](int64_t col) {
        const float w = Load(weight, col);
        return Load(src, src_base + col) * inv * (gemma ? 1.0f + w : w);
      };
      float value = norm(i);
      if (i < rot) {
        const int64_t pair = i < half ? i : i - half;
        const float first = norm(pair), second = norm(pair + half);
        const float c = Load(cs, token * rot + pair);
        const float s = Load(cs, token * rot + half + pair);
        value = i < half ? first * c - second * s : first * s + second * c;
      }
      Store(dst, out_base + i, value);
      if (query) Store(go, out_base + i, Load(qs, src_base + dim + i));
    }
  });
  RecordProfileEvent(q, "attn_qk_norm_rope_gate", event);
}
void RopeNeoxKernel(Queue& q, Tensor& queries, Tensor& keys, const Tensor& positions, const RopeArgs& args) {
  TraceXpuOp(OpId::kRopeNeox, q, {&queries, &keys, &positions});
  VT_CHECK(NativeQueue(q).get_device().has(sycl::aspect::fp64), "XPU legacy RoPE requires FP64 frequency math");
  if (!args.rotary_dim) return;
  const View qs(queries), ks(keys), pos(positions);
  const int64_t hq = queries.shape[1], hk = keys.shape[1], half = args.rotary_dim / 2;
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(queries.shape[0] * (hq + hk) * half), [=](sycl::id<1> item) {
    const int64_t pair = item[0] % half, head = (item[0] / half) % (hq + hk), token = item[0] / (half * (hq + hk));
    const double angle = double(Position(pos, token)) * Frequency(pair, args);
    const float c = float(sycl::cos(angle)), s = float(sycl::sin(angle));
    Rotate(head < hq ? qs : ks, token, head < hq ? head : head - hq, pair, pair + half, c, s);
  });
  RecordProfileEvent(q, "rope_neox", event);
}
void RopeCosSinCacheKernel(Queue& q, Tensor& cache, const Tensor& positions, const RopeArgs& args) {
  TraceXpuOp(OpId::kRopeCosSinCache, q, {&cache, &positions});
  VT_CHECK(NativeQueue(q).get_device().has(sycl::aspect::fp64), "XPU legacy RoPE requires FP64 frequency math");
  if (!args.rotary_dim) return;
  const View dst(cache), pos(positions);
  const auto rot = args.rotary_dim, half = rot / 2;
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(cache.shape[0] * half), [=](sycl::id<1> item) {
    const int64_t row = item[0] / half, pair = item[0] % half, p = Position(pos, row);
    float c, s;
    if (args.linear_scaling_factor > 0) {
      // Round both pow and its reciprocal to F32 before the F32 angle.
      // B70 FP64 avoids the larger device float-pow approximation.
      const float exponent = float(2 * pair) / float(rot);
      const float power = float(sycl::pow(double(args.base), double(exponent)));
      const float inv = float(1.0 / double(power));
      const float angle = (float(p) / args.linear_scaling_factor) * inv;
      c = sycl::cos(angle); s = sycl::sin(angle);
    } else {
      const double angle = double(p) * Frequency(pair, args);
      c = float(sycl::cos(angle)); s = float(sycl::sin(angle));
    }
    Store(dst, row * rot + pair, c); Store(dst, row * rot + half + pair, s);
  });
  RecordProfileEvent(q, "rope_cache_produce", event);
}
void RopeFromCacheKernel(Queue& q, Tensor& queries, Tensor* keys, const Tensor& positions,
                          const Tensor& cache, const RopeArgs& args) {
  TraceXpuOp(OpId::kRopeFromCache, q, {&queries, keys, &positions, &cache});
  VT_CHECK(positions.rank == 1, "XPU text RoPE does not yet implement multi-axis vision positions");
  if (!args.rotary_dim) return;
  const View qs(queries), ks(keys ? *keys : queries), pos(positions), cs(cache);
  const auto tokens = queries.shape[0], hq = queries.shape[1], hk = keys ? keys->shape[1] : 0;
  const auto half = args.rotary_dim / 2, rot = args.rotary_dim;
  const auto count = cache.shape[0];
  CheckDeviceMetadata(q, [=] {
    for (int64_t i = 0; i < tokens; ++i) if (Position(pos, i) < 0 || Position(pos, i) >= count) return false;
    return true;
  }, "XPU RoPE position outside cache", {&positions});
  const bool neox = args.is_neox_style;
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(tokens * (hq + hk) * half), [=](sycl::id<1> item) {
    const int64_t pair = item[0] % half, head = (item[0] / half) % (hq + hk), token = item[0] / (half * (hq + hk));
    const auto base = Position(pos, token) * rot;
    Rotate(head < hq ? qs : ks, token, head < hq ? head : head - hq,
           neox ? pair : 2 * pair, neox ? pair + half : 2 * pair + 1,
           Load(cs, base + pair), Load(cs, base + half + pair));
  });
  RecordProfileEvent(q, "rope_cache_consume", event);
}
namespace {
template<bool Fp8>
void CacheWrite(Queue& q, const Tensor& keys, const Tensor& values, Tensor& key_cache,
                Tensor& value_cache, const Tensor& slots, float k_scale, float v_scale) {
  const int64_t count = slots.Numel(), page = key_cache.shape[1], blocks = key_cache.shape[0];
  const auto elements = keys.shape[1] * keys.shape[2];
  const auto* ids = static_cast<const int64_t*>(slots.data);
  CheckDeviceMetadata(q, [=] {
    for (int64_t t = 0; t < count; ++t) if (ids[t] >= blocks * page) return false;
    return true;
  }, "XPU KV slot outside cache", {&slots});
  if (!count || !elements) return;
  const View ks(keys), vs(values), kc(key_cache), vc(value_cache);
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<int> keep(1, h);
    h.parallel_for(sycl::nd_range<1>(count * 128, 128), [=](sycl::nd_item<1> item) {
      const int64_t token = item.get_group(0), slot = ids[token];
      if (item.get_local_id(0) == 0) {
        // Last writer wins, checked once per token rather than per K/V value.
        int active = slot >= 0;
        for (int64_t later = token + 1; active && later < count; ++later)
          if (ids[later] == slot) active = 0;
        keep[0] = active;
      }
      item.barrier(sycl::access::fence_space::local_space);
      if (!keep[0]) return;
      const auto block = slot / page, offset = slot % page;
      for (int64_t col = item.get_local_id(0); col < elements; col += 128) {
        const auto kd = block * kc.stride[0] + offset * kc.stride[1] + col;
        const auto vd = block * vc.stride[0] + offset * vc.stride[1] + col;
        if constexpr (Fp8) {
          static_cast<uint8_t*>(kc.data)[kd] = EncodeE4M3(Load(ks, token * ks.stride[0] + col) / k_scale);
          static_cast<uint8_t*>(vc.data)[vd] = EncodeE4M3(Load(vs, token * vs.stride[0] + col) / v_scale);
        } else {
          CopyElement(kc, kd, ks, token * ks.stride[0] + col);
          CopyElement(vc, vd, vs, token * vs.stride[0] + col);
        }
      }
    });
  });
  RecordProfileEvent(q, Fp8 ? "kv_write_fp8" : "kv_write", event);
}
}
void ReshapeAndCacheKernel(Queue& q, const Tensor& keys, const Tensor& values, Tensor& key_cache,
                            Tensor& value_cache, const Tensor& slots) {
  TraceXpuOp(OpId::kReshapeAndCache, q, {&keys, &values, &key_cache, &value_cache, &slots});
  CacheWrite<false>(q, keys, values, key_cache, value_cache, slots, 1, 1);
}
void ReshapeAndCacheFp8Kernel(Queue& q, const Tensor& keys, const Tensor& values, Tensor& key_cache,
                               Tensor& value_cache, const Tensor& slots, Fp8KVCacheDataType kind,
                               float k_scale, float v_scale) {
  TraceXpuOp(OpId::kReshapeAndCacheFp8, q, {&keys, &values, &key_cache, &value_cache, &slots});
  VT_CHECK(kind == Fp8KVCacheDataType::kFp8E4M3 && std::isfinite(k_scale) && std::isfinite(v_scale),
           "XPU FP8 KV requires E4M3 and finite positive scales");
  CacheWrite<true>(q, keys, values, key_cache, value_cache, slots, k_scale, v_scale);
}

void PagedAttentionKernel(Queue& q, Tensor& out, const Tensor& query, const Tensor& key_cache,
                           const Tensor& value_cache, const Tensor& block_table,
                           const Tensor& seq_lens, const Tensor& query_start_loc,
                           const PagedAttentionArgs& args) {
  TraceXpuOp(OpId::kPagedAttention, q, {&out, &query, &key_cache, &value_cache,
                                          &block_table, &seq_lens, &query_start_loc});
  VT_CHECK(args.kv_cache_dtype == Fp8KVCacheDataType::kAuto ||
               (args.kv_cache_dtype == Fp8KVCacheDataType::kFp8E4M3 &&
                std::isfinite(args.k_scale) && std::isfinite(args.v_scale)),
           "XPU paged attention requires float KV or E4M3 with finite positive scales");
  const auto tokens = query.shape[0], heads = query.shape[1], dim = query.shape[2];
  const auto page = key_cache.shape[1], blocks = key_cache.shape[0], ratio = heads / key_cache.shape[2];
  const auto requests = seq_lens.Numel(), columns = block_table.shape[1];
  const auto bt_row = block_table.stride[0], bt_col = block_table.stride[1];
  VT_CHECK(page > 0 && dim > 0, "XPU paged attention requires positive page/head size");
  const auto* lengths = static_cast<const int32_t*>(seq_lens.data);
  const auto* offsets = static_cast<const int32_t*>(query_start_loc.data);
  const auto* table = static_cast<const int32_t*>(block_table.data);
  CheckDeviceMetadata(q, [=] {
    if (offsets[0] != 0 || offsets[requests] != tokens) return false;
    for (int64_t r = 0; r < requests; ++r) {
      const int64_t first = offsets[r], end = offsets[r + 1], length = lengths[r];
      if (first < 0 || end < first || end > tokens) return false;
      if (end == first) continue;  // Padded/inactive rows need no valid cache entries.
      if (length < end - first) return false;
      const auto needed = (length + page - 1) / page;
      if (needed > columns) return false;
      for (int64_t b = 0; b < needed; ++b) {
        const auto id = table[r * bt_row + b * bt_col];
        if (id < 0 || id >= blocks) return false;
      }
    }
    return true;
  }, "XPU paged attention invalid sequence offsets, lengths or block table", {&seq_lens, &query_start_loc, &block_table});
  if (!tokens) return;
  size_t lanes = 1;
  while (lanes < static_cast<uint64_t>(dim)) lanes *= 2;
  VT_CHECK(lanes <= NativeQueue(q).get_device().get_info<sycl::info::device::max_work_group_size>(),
           "XPU paged attention head exceeds workgroup limit");
  const float scale = args.scale, cap = args.logits_soft_cap;
  const float k_scale = args.k_scale, v_scale = args.v_scale;
  const bool causal = args.causal;
  const int64_t left = args.window_size ? args.window_size->left : -1;
  const int64_t right = args.window_size ? args.window_size->right : -1;
  WithOutput(q, out, {&query, &key_cache, &value_cache, &block_table, &seq_lens, &query_start_loc}, [&](Tensor& target) {
    const char* setting = std::getenv("VT_XPU_ATTENTION");
    const std::string_view mode = setting ? setting : "auto";
    VT_CHECK(mode == "auto" || mode == "reference" || mode == "split" || mode == "prefill", "Invalid VT_XPU_ATTENTION");
    const auto device = NativeQueue(q).get_device();
    // Qualified against the pinned checkpoint's answer and probability corpus.
    // FP8 remains an explicit cache choice; its quantization has a separate
    // quality delta, so do not treat the BF16 qualification as an FP8 gate.
    const bool automatic = mode == "auto" && query.shape[1] == 24 && dim == 256 &&
        key_cache.shape[2] == 4 &&
        (key_cache.dtype == DType::kBF16 || key_cache.dtype == DType::kF16) &&
        device.has(sycl::aspect::ext_intel_device_id) &&
        device.get_info<sycl::ext::intel::info::device::device_id>() == 57891 &&
        std::string_view(__VERSION__) == "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)" &&
        device.get_info<sycl::info::device::driver_version>() == "1.17.39758+10" &&
        device.get_platform().get_info<sycl::info::platform::version>() == "1.17";
    const bool try_split = automatic || mode == "split" || mode == "prefill";
    const bool split = try_split && PagedAttentionSplitKernel(
        q, target, query, key_cache, value_cache,
        block_table, seq_lens, query_start_loc, args);
    const bool try_prefill = !split && (automatic || mode == "prefill");
    const bool prefill = try_prefill && PagedAttentionPrefillKernel(
        q, target, query, key_cache, value_cache,
        block_table, seq_lens, query_start_loc, args);
    if (const char* trace = std::getenv("VT_XPU_TRACE_FAST_PATH");
        trace != nullptr && trace[0] == '1' && trace[1] == '\0') {
      const char* reason = split || prefill ? "eligible" :
          mode == "reference" ? "mode_reference" :
          mode == "auto" && !automatic ? "stack_gate_or_shape" :
          "kernel_declined";
      std::fprintf(stderr,
                   "{\"event\":\"xpu_fast_path\",\"operator\":\"paged_attention\","
                   "\"selected\":\"%s\",\"reason\":\"%s\","
                   "\"mode\":\"%.*s\",\"auto_eligible\":%s,"
                   "\"tokens\":%lld}\n",
                   split ? "split" : prefill ? "prefill" : "reference", reason,
                   static_cast<int>(mode.size()), mode.data(),
                   automatic ? "true" : "false", static_cast<long long>(tokens));
    }
    if (split || prefill) return;
    const View qs(query), kc(key_cache), vc(value_cache), dst(target);
    const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
      sycl::local_accessor<float, 1> partial(sycl::range<1>(lanes), h);
      // One group per query/head; each lane owns one value component. Scores
      // reduce in F32, and online softmax never materializes a scores matrix.
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(tokens * heads * lanes), sycl::range<1>(lanes)),
                     [=](sycl::nd_item<1> item) {
        const auto lane = item.get_local_id(0);
        const bool active = lane < static_cast<size_t>(dim);
        const int64_t token = item.get_group(0) / heads, head = item.get_group(0) % heads;
        int64_t request = 0;
        while (token >= offsets[request + 1]) ++request;
        const int64_t position = lengths[request] - (offsets[request + 1] - offsets[request])
                                   + token - offsets[request];
        const int64_t first = left < 0 ? 0 : sycl::max(int64_t{0}, position - left);
        int64_t last = causal ? position : int64_t(lengths[request]) - 1;
        if (right >= 0) last = sycl::min(last, position + right);
        const auto qbase = (token * heads + head) * dim;
        const float qvalue = active ? Load(qs, qbase + lane) : 0;
        float maximum = -std::numeric_limits<float>::infinity(), denominator = 0, accumulator = 0;
        for (int64_t key = first; key <= last; ++key) {
          const int64_t block = table[request * bt_row + (key / page) * bt_col];
          const auto kbase = block * kc.stride[0] + (key % page) * kc.stride[1] + (head / ratio) * kc.stride[2];
          const auto vbase = block * vc.stride[0] + (key % page) * vc.stride[1] + (head / ratio) * vc.stride[2];
          partial[lane] = active ? qvalue * LoadKV(kc, kbase + lane, k_scale) : 0;
          item.barrier(sycl::access::fence_space::local_space);
          for (size_t step = lanes / 2; step > 0; step /= 2) {
            if (lane < step) partial[lane] += partial[lane + step];
            item.barrier(sycl::access::fence_space::local_space);
          }
          float score = partial[0] * scale;
          if (cap > 0) score = cap * sycl::tanh(score / cap);
          const float next = sycl::max(maximum, score);
          const float old_scale = sycl::exp(maximum - next), probability = sycl::exp(score - next);
          denominator = denominator * old_scale + probability;
          if (active) accumulator = accumulator * old_scale + probability * LoadKV(vc, vbase + lane, v_scale);
          maximum = next;
          item.barrier(sycl::access::fence_space::local_space);
        }
        if (active) Store(dst, qbase + lane, accumulator / denominator);
      });
    });
    RecordProfileEvent(q, "attention_reference", event);
  });
}
}  // namespace vt::xpu
