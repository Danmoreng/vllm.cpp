#include "xpu_common.h"
#include "xpu_kernels.h"
#include <limits>

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
  TraceOpTensors(OpId::kAttnGateSplit, q, {&queries, &gates, &packed});
  VT_CHECK(!Overlap(queries, packed) && !Overlap(gates, packed) && !Overlap(queries, gates),
           "XPU Q/gate split requires separate output storage");
  const View src(packed), dst(queries), gate(gates);
  const auto width = queries.shape[2];
  NativeQueue(q).parallel_for(sycl::range<1>(queries.Numel()), [=](sycl::id<1> item) {
    const int64_t head = item[0] / width, col = item[0] % width;
    Store(dst, item[0], Load(src, head * 2 * width + col));
    Store(gate, item[0], Load(src, head * 2 * width + width + col));
  });
}
void RopeNeoxKernel(Queue& q, Tensor& queries, Tensor& keys, const Tensor& positions, const RopeArgs& args) {
  TraceOpTensors(OpId::kRopeNeox, q, {&queries, &keys, &positions});
  VT_CHECK(NativeQueue(q).get_device().has(sycl::aspect::fp64), "XPU legacy RoPE requires FP64 frequency math");
  if (!args.rotary_dim) return;
  const View qs(queries), ks(keys), pos(positions);
  const int64_t hq = queries.shape[1], hk = keys.shape[1], half = args.rotary_dim / 2;
  NativeQueue(q).parallel_for(sycl::range<1>(queries.shape[0] * (hq + hk) * half), [=](sycl::id<1> item) {
    const int64_t pair = item[0] % half, head = (item[0] / half) % (hq + hk), token = item[0] / (half * (hq + hk));
    const double angle = double(Position(pos, token)) * Frequency(pair, args);
    const float c = float(sycl::cos(angle)), s = float(sycl::sin(angle));
    Rotate(head < hq ? qs : ks, token, head < hq ? head : head - hq, pair, pair + half, c, s);
  });
}
void RopeCosSinCacheKernel(Queue& q, Tensor& cache, const Tensor& positions, const RopeArgs& args) {
  TraceOpTensors(OpId::kRopeCosSinCache, q, {&cache, &positions});
  VT_CHECK(NativeQueue(q).get_device().has(sycl::aspect::fp64), "XPU legacy RoPE requires FP64 frequency math");
  if (!args.rotary_dim) return;
  const View dst(cache), pos(positions);
  const auto rot = args.rotary_dim, half = rot / 2;
  NativeQueue(q).parallel_for(sycl::range<1>(cache.shape[0] * half), [=](sycl::id<1> item) {
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
}
void RopeFromCacheKernel(Queue& q, Tensor& queries, Tensor* keys, const Tensor& positions,
                          const Tensor& cache, const RopeArgs& args) {
  TraceOpTensors(OpId::kRopeFromCache, q, {&queries, keys, &positions, &cache});
  VT_CHECK(positions.rank == 1, "XPU text RoPE does not yet implement multi-axis vision positions");
  if (!args.rotary_dim) return;
  const View qs(queries), ks(keys ? *keys : queries), pos(positions), cs(cache);
  const auto tokens = queries.shape[0], hq = queries.shape[1], hk = keys ? keys->shape[1] : 0;
  const auto half = args.rotary_dim / 2, rot = args.rotary_dim;
  const auto count = cache.shape[0];
  CheckDeviceMetadata(q, [=] {
    for (int64_t i = 0; i < tokens; ++i) if (Position(pos, i) < 0 || Position(pos, i) >= count) return false;
    return true;
  }, "XPU RoPE position outside cache");
  const bool neox = args.is_neox_style;
  NativeQueue(q).parallel_for(sycl::range<1>(tokens * (hq + hk) * half), [=](sycl::id<1> item) {
    const int64_t pair = item[0] % half, head = (item[0] / half) % (hq + hk), token = item[0] / (half * (hq + hk));
    const auto base = Position(pos, token) * rot;
    Rotate(head < hq ? qs : ks, token, head < hq ? head : head - hq,
           neox ? pair : 2 * pair, neox ? pair + half : 2 * pair + 1,
           Load(cs, base + pair), Load(cs, base + half + pair));
  });
}
void ReshapeAndCacheKernel(Queue& q, const Tensor& keys, const Tensor& values, Tensor& key_cache,
                            Tensor& value_cache, const Tensor& slots) {
  TraceOpTensors(OpId::kReshapeAndCache, q, {&keys, &values, &key_cache, &value_cache, &slots});
  const int64_t count = slots.Numel(), page = key_cache.shape[1], blocks = key_cache.shape[0];
  const auto elements = keys.shape[1] * keys.shape[2];
  const auto* ids = static_cast<const int64_t*>(slots.data);
  CheckDeviceMetadata(q, [=] {
    for (int64_t t = 0; t < count; ++t) if (ids[t] >= blocks * page) return false;
    return true;
  }, "XPU KV slot outside cache");
  const View ks(keys), vs(values), kc(key_cache), vc(value_cache);
  NativeQueue(q).parallel_for(sycl::range<1>(count * elements), [=](sycl::id<1> item) {
    const int64_t token = item[0] / elements, col = item[0] % elements, slot = ids[token];
    if (slot < 0) return;
    // Match the CPU loop if a caller repeats a slot: the last token wins.
    for (int64_t later = token + 1; later < count; ++later) if (ids[later] == slot) return;
    const auto block = slot / page, offset = slot % page;
    CopyElement(kc, block * kc.stride[0] + offset * kc.stride[1] + col, ks, token * ks.stride[0] + col);
    CopyElement(vc, block * vc.stride[0] + offset * vc.stride[1] + col, vs, token * vs.stride[0] + col);
  });
}

void PagedAttentionKernel(Queue& q, Tensor& out, const Tensor& query, const Tensor& key_cache,
                           const Tensor& value_cache, const Tensor& block_table,
                           const Tensor& seq_lens, const Tensor& query_start_loc,
                           const PagedAttentionArgs& args) {
  TraceOpTensors(OpId::kPagedAttention, q, {&out, &query, &key_cache, &value_cache,
                                          &block_table, &seq_lens, &query_start_loc});
  VT_CHECK(args.kv_cache_dtype == Fp8KVCacheDataType::kAuto, "XPU paged attention requires float KV cache");
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
  }, "XPU paged attention invalid sequence offsets, lengths or block table");
  if (!tokens) return;
  size_t lanes = 1;
  while (lanes < static_cast<uint64_t>(dim)) lanes *= 2;
  VT_CHECK(lanes <= NativeQueue(q).get_device().get_info<sycl::info::device::max_work_group_size>(),
           "XPU paged attention head exceeds workgroup limit");
  const float scale = args.scale, cap = args.logits_soft_cap;
  const bool causal = args.causal;
  const int64_t left = args.window_size ? args.window_size->left : -1;
  const int64_t right = args.window_size ? args.window_size->right : -1;
  WithOutput(q, out, {&query, &key_cache, &value_cache, &block_table, &seq_lens, &query_start_loc}, [&](Tensor& target) {
    const View qs(query), kc(key_cache), vc(value_cache), dst(target);
    NativeQueue(q).submit([&](sycl::handler& h) {
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
          partial[lane] = active ? qvalue * Load(kc, kbase + lane) : 0;
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
          if (active) accumulator = accumulator * old_scale + probability * Load(vc, vbase + lane);
          maximum = next;
          item.barrier(sycl::access::fence_space::local_space);
        }
        if (active) Store(dst, qbase + lane, accumulator / denominator);
      });
    });
  });
}
}  // namespace vt::xpu
