#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include <numeric>

namespace {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;
using xpu_test::Values;
using xpu_test::Close;
using xpu_test::SameBytes;
vt::Tensor FlatHeads(const vt::Tensor& tensor) {
  return vt::Tensor::Contiguous(tensor.data, tensor.dtype, tensor.device,
                                {tensor.shape[0] * tensor.shape[1], tensor.shape[2]});
}
// Actual cache allocation is [blocks,2,page,Hkv,D]; K/V are unbind views.
vt::Tensor CacheView(const vt::Tensor& storage, int side, int64_t blocks, int64_t page,
                      int64_t heads, int64_t dim) {
  auto view = vt::Tensor::Contiguous(static_cast<char*>(storage.data) + side * page * heads * dim * vt::SizeOf(storage.dtype),
                                    storage.dtype, storage.device, {blocks, page, heads, dim});
  view.stride[0] *= 2;
  return view;
}
}

TEST_CASE("XPU attention preamble: Q/gate split, Q/K RMSNorm, partial RoPE at real geometry") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int hq = 24, hk = 4, dim = 256, rot = 64;
  for (int tokens : {1, 5}) for (auto packed_type : {DType::kF32, DType::kBF16})
    for (auto dtype : {DType::kF32, DType::kBF16}) {
      std::vector<std::vector<float>> expected;
      for (auto* q : {&cpu.q, &gpu.q}) {
        Buffer packed(*q, packed_type, {tokens, hq * 2 * dim}), queries(*q, dtype, {tokens, hq, dim});
        Buffer gates(*q, DType::kF32, {tokens, hq, dim}), keys(*q, dtype, {tokens, hk, dim});
        Buffer qw(*q, DType::kF32, {dim}), kw(*q, DType::kF32, {dim}), pos(*q, DType::kI64, {tokens});
        packed.put(Values(tokens * hq * 2 * dim)); keys.put(Values(tokens * hk * dim, 5));
        qw.put(Values(dim, 3)); kw.put(Values(dim, 7));
        const int64_t positions[] = {0, 1, 137, 65535, 262143}; pos.upload(positions);
        vt::AttnGateSplit(*q, queries.tensor, gates.tensor, packed.tensor);
        auto qflat = FlatHeads(queries.tensor), kflat = FlatHeads(keys.tensor);
        vt::RmsNorm(*q, qflat, qflat, qw.tensor, {1e-6f, true});
        vt::RmsNorm(*q, kflat, kflat, kw.tensor, {1e-6f, true});
        std::vector<std::vector<float>> actual{queries.floats(), keys.floats(), gates.floats()};
        vt::RopeNeox(*q, queries.tensor, keys.tensor, pos.tensor, {10000000.0f, rot});
        actual.push_back(queries.floats()); actual.push_back(keys.floats());
        // RoPE must leave every non-rotary channel untouched.
        for (int side = 0; side < 2; ++side) {
          const int heads = side == 0 ? hq : hk;
          bool unchanged = true;
          for (int row = 0; row < tokens * heads; ++row) for (int col = rot; col < dim; ++col)
            unchanged &= actual[side][row * dim + col] == actual[side + 3][row * dim + col];
          CHECK(unchanged);
        }
        if (q == &cpu.q) expected = actual;
        else for (size_t i = 0; i < actual.size(); ++i)
          Close(actual[i], expected[i], dtype == DType::kBF16 && i != 2 ? 0.008f : 4e-6f);
      }
    }
}

TEST_CASE("XPU cached RoPE: supplied positions, strided heads, optional K and both pair styles") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int tokens = 4, hq = 24, hk = 4, dim = 256, rot = 64, count = 130;
  for (int scaling : {0, 1, 2}) for (bool neox : {false, true}) for (bool have_keys : {false, true}) {
    CAPTURE(scaling);
    CAPTURE(neox);
    CAPTURE(have_keys);
    std::vector<float> expected_cache, expected_q, expected_k;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer queries(*q, DType::kF32, {tokens, hq + 1, dim + 3});
      Buffer keys(*q, DType::kF32, {tokens, hk + 2, dim + 7});
      Buffer cache(*q, DType::kF32, {count, rot}), allpos(*q, DType::kI32, {count}), pos(*q, DType::kI32, {tokens});
      queries.put(Values(tokens * (hq + 1) * (dim + 3))); keys.put(Values(tokens * (hk + 2) * (dim + 7), 4));
      queries.tensor.shape[1] = hq; queries.tensor.shape[2] = dim;
      keys.tensor.shape[1] = hk; keys.tensor.shape[2] = dim;
      std::vector<int32_t> all_positions(count); std::iota(all_positions.begin(), all_positions.end(), 0);
      allpos.upload(all_positions.data()); const int32_t positions[] = {129, 3, 64, 0}; pos.upload(positions);
      vt::RopeArgs args{10000000.0f, rot}; args.is_neox_style = neox;
      if (scaling == 1) args.linear_scaling_factor = 2.0f;
      if (scaling == 2) {
        args.llama3_scaling_factor = 8; args.llama3_low_freq_factor = 1;
        args.llama3_high_freq_factor = 4; args.llama3_orig_max_position = 8192;
      }
      vt::RopeCosSinCache(*q, cache.tensor, allpos.tensor, args);
      vt::RopeFromCache(*q, queries.tensor, have_keys ? &keys.tensor : nullptr, pos.tensor, cache.tensor, args);
      if (q == &cpu.q) { expected_cache = cache.floats(); expected_q = queries.floats(); expected_k = keys.floats(); }
      else {
        // CPU vector powf/sincos and device math need not round identically.
        // F32 inverse-frequency rounding becomes an angle error with position.
        const float absolute = scaling == 1 ? 1e-5f : 1e-6f;
        Close(cache.floats(), expected_cache, 4e-6f, absolute);
        Close(queries.floats(), expected_q, 4e-6f, absolute);
        Close(keys.floats(), expected_k, 4e-6f, absolute);
        if (scaling == 1) {
          std::vector<float> scalar(count * rot);
          for (int p = 0; p < count; ++p) for (int i = 0; i < rot / 2; ++i) {
            const float exponent = float(2 * i) / float(rot);
            const float power = float(std::pow(double(args.base), double(exponent)));
            const float inv = float(1.0 / double(power));
            const float angle = (float(p) / args.linear_scaling_factor) * inv;
            scalar[p * rot + i] = float(std::cos(double(angle)));
            scalar[p * rot + rot / 2 + i] = float(std::sin(double(angle)));
          }
          Close(cache.floats(), scalar, 1e-6f);
        }
        const int32_t invalid[] = {130, 3, 64, 0}; pos.upload(invalid);
        CHECK_THROWS(vt::RopeFromCache(*q, queries.tensor, nullptr, pos.tensor, cache.tensor, args));
      }
    }
  }
}

TEST_CASE("XPU KV write: unbind strides, padded inputs, null/repeated slots and raw bit preservation") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int tokens = 7, blocks = 4, heads = 4, dim = 256;
  for (int page : {3, 16}) for (auto dtype : {DType::kF32, DType::kBF16, DType::kF16}) {
    std::vector<unsigned char> expected;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer keys(*q, dtype, {tokens, heads + 1, dim}), values(*q, dtype, {tokens, heads + 2, dim});
      Buffer storage(*q, dtype, {blocks, 2 * page, heads, dim}), slots(*q, DType::kI64, {tokens});
      // Includes NaNs and signed zero; a KV store is a bit copy, not a cast.
      std::vector<uint32_t> raw_keys((keys.bytes + 3) / 4), raw_values((values.bytes + 3) / 4);
      for (size_t i = 0; i < raw_keys.size(); ++i) raw_keys[i] = uint32_t(i * 2654435761u);
      for (size_t i = 0; i < raw_values.size(); ++i) raw_values[i] = uint32_t(i * 2246822519u);
      keys.upload(raw_keys.data()); values.upload(raw_values.data()); storage.put(Values(storage.tensor.Numel(), 7));
      keys.tensor.shape[1] = values.tensor.shape[1] = heads;
      auto kc = CacheView(storage.tensor, 0, blocks, page, heads, dim);
      auto vc = CacheView(storage.tensor, 1, blocks, page, heads, dim);
      const int64_t mapping[] = {2 * page, -1, page - 1, page, 0, blocks * page - 1, 0}; slots.upload(mapping);
      vt::ReshapeAndCache(*q, keys.tensor, values.tensor, kc, vc, slots.tensor);
      if (q == &cpu.q) expected = storage.download();
      else {
        SameBytes(storage.download(), expected);
        const int64_t invalid[] = {blocks * page, -1, 0, 0, 0, 0, 0}; slots.upload(invalid);
        CHECK_THROWS(vt::ReshapeAndCache(*q, keys.tensor, values.tensor, kc, vc, slots.tensor));
        SameBytes(storage.download(), expected);
      }
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 0);
}

TEST_CASE("XPU paged attention: appended queries, GQA, causal alignment, strides and BF16 cache") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int requests : {1, 4}) for (int chunk : {1, 5}) for (int page : {3, 16})
    for (auto dtype : {DType::kF32, DType::kBF16}) for (int mode : {0, 1, 2}) {
      CAPTURE(requests);
      CAPTURE(chunk);
      CAPTURE(page);
      CAPTURE(dtype);
      CAPTURE(mode);
      constexpr int heads = 24, kvheads = 4, dim = 256, columns = 6;
      const int blocks = requests * columns;
      std::vector<int32_t> offsets{0}, lengths(requests), table(requests * (2 * columns + 1), -1);
      std::vector<int64_t> slots;
      for (int r = 0; r < requests; ++r) {
        const int count = chunk == 1 ? 1 : chunk - r;
        const int context = page + r + 1;
        offsets.push_back(offsets.back() + count); lengths[r] = context + count;
        for (int b = 0; b < columns; ++b) table[r * (2 * columns + 1) + 2 * b] = blocks - 1 - (r * columns + b);
        for (int t = 0; t < count; ++t) {
          const auto pos = context + t;
          slots.push_back(int64_t(table[r * (2 * columns + 1) + 2 * (pos / page)]) * page + pos % page);
        }
      }
      const int tokens = offsets.back();
      std::vector<float> expected;
      for (auto* q : {&cpu.q, &gpu.q}) {
        Buffer query(*q, dtype, {tokens, heads, dim}), output(*q, dtype, {tokens, heads, dim});
        Buffer storage(*q, DType::kBF16, {blocks, 2 * page, kvheads, dim});
        Buffer keys(*q, DType::kBF16, {tokens, kvheads, dim}), values(*q, DType::kBF16, {tokens, kvheads, dim});
        Buffer bt(*q, DType::kI32, {requests, 2 * columns + 1}), lens(*q, DType::kI32, {requests});
        Buffer qsl(*q, DType::kI32, {requests + 1}), ids(*q, DType::kI64, {tokens});
        query.put(Values(query.tensor.Numel(), 7, 0.2f)); storage.put(Values(storage.tensor.Numel(), 3, 0.08f));
        keys.put(Values(keys.tensor.Numel(), 13, 0.2f)); values.put(Values(values.tensor.Numel(), 11));
        bt.upload(table.data()); lens.upload(lengths.data()); qsl.upload(offsets.data()); ids.upload(slots.data());
        bt.tensor.shape[1] = columns; bt.tensor.stride[1] = 2;
        auto kc = CacheView(storage.tensor, 0, blocks, page, kvheads, dim);
        auto vc = CacheView(storage.tensor, 1, blocks, page, kvheads, dim);
        vt::ReshapeAndCache(*q, keys.tensor, values.tensor, kc, vc, ids.tensor);
        vt::PagedAttentionArgs args;
        args.scale = 1.0f / 16.0f; args.causal = mode != 1;
        if (mode == 2) { args.window_size = vt::AttentionWindow{3, 2}; args.logits_soft_cap = 0.7f; }
        // Also exercise alias snapshot on the one-token decode path.
        auto& target = q == &gpu.q && chunk == 1 ? query.tensor : output.tensor;
        vt::PagedAttention(*q, target, query.tensor, kc, vc, bt.tensor, lens.tensor, qsl.tensor, args);
        if (q == &cpu.q) expected = output.floats();
        else Close(chunk == 1 ? query.floats() : output.floats(), expected,
                     dtype == DType::kBF16 ? 0.008f : 1e-4f, dtype == DType::kBF16 ? 2e-5f : 5e-6f);
      }
    }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 0);
}

TEST_CASE("XPU paged attention: head tails, empty requests, F16 queries and invalid device metadata") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int tokens = 3, heads = 6, kvheads = 2, dim = 13, page = 3, blocks = 4;
  std::vector<float> expected;
  for (auto* q : {&cpu.q, &gpu.q}) {
    Buffer query(*q, DType::kF16, {tokens, heads, dim}), output(*q, DType::kF32, {tokens, heads, dim});
    Buffer storage(*q, DType::kBF16, {blocks, 2 * page, kvheads, dim});
    Buffer table(*q, DType::kI32, {4, 2}), lens(*q, DType::kI32, {4}), qsl(*q, DType::kI32, {5});
    query.put(Values(query.tensor.Numel(), 4)); storage.put(Values(storage.tensor.Numel(), 9));
    const int32_t offsets[] = {0, 0, 2, 2, 3}, lengths[] = {-1, 5, -1, 4}, ids[] = {-1, -1, 2, 0, -1, -1, 3, 1};
    qsl.upload(offsets); lens.upload(lengths); table.upload(ids);
    auto kc = CacheView(storage.tensor, 0, blocks, page, kvheads, dim);
    auto vc = CacheView(storage.tensor, 1, blocks, page, kvheads, dim);
    vt::PagedAttentionArgs args; args.scale = 1.0f / std::sqrt(float(dim));
    auto run = [&] { vt::PagedAttention(*q, output.tensor, query.tensor, kc, vc, table.tensor, lens.tensor, qsl.tensor, args); };
    run();
    if (q == &cpu.q) expected = output.floats();
    else {
      Close(output.floats(), expected, 2e-5f);
      const int32_t bad_offsets[] = {0, 2, 1, 2, 3}; qsl.upload(bad_offsets); CHECK_THROWS(run()); qsl.upload(offsets);
      for (const auto& bad : {std::vector<int32_t>{-1, 1, -1, 4}, std::vector<int32_t>{-1, 7, -1, 4}}) {
        lens.upload(bad.data()); CHECK_THROWS(run());
      }
      lens.upload(lengths);
      for (int invalid : {-1, blocks}) {
        auto bad = std::vector<int32_t>(ids, ids + 8); bad[2] = invalid;
        table.upload(bad.data()); CHECK_THROWS(run());
      }
      Close(output.floats(), expected, 2e-5f);
    }
  }
}
