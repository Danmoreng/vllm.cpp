#include <doctest/doctest.h>
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/xpu.h"
#include "vllm/v1/sample/sampler.h"
#include "vllm/platforms/interface.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {
using vt::DType;
struct Queues {
  vt::Queue cpu = vt::CreateQueue({vt::DeviceType::kCPU, 0});
  vt::Queue gpu = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  ~Queues() { vt::DestroyQueue(gpu); vt::DestroyQueue(cpu); }
};
struct Buffer {
  vt::Queue& q;
  vt::Tensor t;
  size_t bytes;
  Buffer(vt::Queue& queue, DType dtype, std::initializer_list<int64_t> shape) : q(queue) {
    t = vt::Tensor::Contiguous(nullptr, dtype, q.device, shape);
    bytes = t.Bytes();
    t.data = vt::Alloc(q.device, std::max(bytes, size_t{1}));
  }
  ~Buffer() { vt::Free(q.device, t.data); }
  Buffer(const Buffer&) = delete;
  void upload(const void* data) {
    auto& b = vt::GetBackend(q.device); b.Copy(q, t.data, data, bytes); b.Synchronize(q);
  }
  void put(const std::vector<float>& values) {
    REQUIRE(values.size() * vt::SizeOf(t.dtype) == bytes);
    if (t.dtype == DType::kF32) { upload(values.data()); return; }
    std::vector<uint16_t> packed(values.size());
    for (size_t i = 0; i < values.size(); ++i)
      packed[i] = t.dtype == DType::kF16 ? vt::F32ToF16(values[i]) : vt::F32ToBF16(values[i]);
    upload(packed.data());
  }
  std::vector<unsigned char> raw() {
    std::vector<unsigned char> data(bytes);
    auto& b = vt::GetBackend(q.device); b.Copy(q, data.data(), t.data, bytes); b.Synchronize(q);
    return data;
  }
  std::vector<float> floats() {
    const auto data = raw(); std::vector<float> values(bytes / vt::SizeOf(t.dtype));
    for (size_t i = 0; i < values.size(); ++i) {
      if (t.dtype == DType::kF32) std::memcpy(&values[i], data.data() + i * 4, 4);
      else { uint16_t v; std::memcpy(&v, data.data() + i * 2, 2);
             values[i] = t.dtype == DType::kF16 ? vt::F16ToF32(v) : vt::BF16ToF32(v); }
    }
    return values;
  }
};
std::vector<float> Values(size_t n, int salt = 0) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = (static_cast<int>((i * 7 + salt) % 41) - 20) / 16.0f;
  return v;
}
void Close(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance) {
  REQUIRE(actual.size() == expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    CAPTURE(i);
    CAPTURE(actual[i]);
    CAPTURE(expected[i]);
    if (std::isnan(expected[i])) CHECK(std::isnan(actual[i]));
    else if (std::isinf(expected[i])) CHECK(actual[i] == expected[i]);
    else CHECK(std::abs(actual[i] - expected[i]) <= tolerance * (1 + std::abs(expected[i])));
  }
}
constexpr DType floats[] = {DType::kF32, DType::kF16, DType::kBF16};
}

TEST_CASE("XPU copy and casts: all float pairs, strides, tails and overlapping transpose") {
  Queues qs;
  for (auto src_type : floats) for (auto dst_type : floats) for (int width : {1, 3, 127, 129}) {
    CAPTURE(src_type);
    CAPTURE(dst_type);
    CAPTURE(width);
    std::vector<unsigned char> expected;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      Buffer src(*q, src_type, {3, width + 4}), dst(*q, dst_type, {3, width + 2});
      auto values = Values(3 * (width + 4));
      values[0] = -0.0f; values[1] = std::numeric_limits<float>::infinity();
      values[2] = std::numeric_limits<float>::quiet_NaN();
      src.put(values); dst.put(std::vector<float>(3 * (width + 2), 17));
      src.t.shape[1] = dst.t.shape[1] = width;
      vt::Copy(*q, dst.t, src.t);
      if (q == &qs.cpu) expected = dst.raw(); else CHECK(dst.raw() == expected);
    }
  }
  for (auto* q : {&qs.cpu, &qs.gpu}) {
    Buffer b(*q, DType::kF32, {3, 3}); b.put({0,1,2,3,4,5,6,7,8});
    auto transposed = b.t; transposed.stride[0] = 1; transposed.stride[1] = 3;
    vt::Copy(*q, transposed, b.t);
    CHECK(b.floats() == std::vector<float>{0,3,6,1,4,7,2,5,8});
    Buffer bf(*q, DType::kBF16, {3, 3}), fp(*q, DType::kF32, {3, 3}), half(*q, DType::kF16, {3, 3});
    vt::CastBf16(*q, bf.t, b.t); vt::CastF32(*q, fp.t, bf.t); vt::CastF16(*q, half.t, bf.t);
    CHECK(fp.floats() == b.floats()); CHECK(half.floats() == b.floats());
  }
}

TEST_CASE("XPU Add, SiLU, MoeSiluMul and sigmoid: input rounding and aliases") {
  Queues qs;
  for (auto dtype : floats) for (int width : {1, 129}) {
    std::vector<std::vector<float>> expected;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      Buffer a(*q, dtype, {2, width}), b(*q, dtype, {width});
      Buffer result(*q, DType::kBF16, {2, width}), gateup(*q, dtype, {2, 2 * width});
      Buffer gate(*q, DType::kF32, {2, width});
      a.put(Values(2 * width)); b.put(Values(width, 3)); gateup.put(Values(4 * width, 5)); gate.put(Values(2 * width, 9));
      std::vector<std::vector<float>> observed;
      vt::Add(*q, result.t, a.t, b.t); observed.push_back(result.floats());
      vt::MoeSiluMul(*q, result.t, a.t, a.t); observed.push_back(result.floats());
      vt::SiluAndMul(*q, result.t, gateup.t); observed.push_back(result.floats());
      vt::SigmoidGateBf16(*q, result.t, gate.t, gate.t); observed.push_back(result.floats());
      vt::MoeSiluMul(*q, result.t, result.t, result.t); observed.push_back(result.floats());
      vt::Add(*q, result.t, result.t, result.t); observed.push_back(result.floats());
      if (q == &qs.cpu) {
        // Oracle is the non-overlapping CPU result. Its row-parallel kernel
        // does not promise snapshot semantics for a packed, overlapping output.
        observed.push_back(observed[2]);
      } else {
        auto alias = gateup.t; alias.dtype = DType::kBF16;
        alias.shape[1] = width; alias.stride[0] = width;
        vt::SiluAndMul(*q, alias, gateup.t);
        vt::Copy(*q, result.t, alias); observed.push_back(result.floats());
      }
      if (q == &qs.cpu) expected = observed;
      else for (size_t i = 0; i < observed.size(); ++i) Close(observed[i], expected[i], 0.008f);
    }
  }
}

TEST_CASE("XPU RMSNorm: weight versus 1+weight, residual rounding and in-place aliases") {
  Queues qs;
  for (auto dtype : floats) for (bool gemma : {false, true}) for (int width : {1, 129, 5120}) {
    CAPTURE(dtype);
    CAPTURE(gemma);
    CAPTURE(width);
    std::vector<float> expected, expected_residual;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      Buffer x(*q, dtype, {2, width}), w(*q, dtype, {width});
      Buffer out(*q, DType::kF32, {2, width}), residual(*q, DType::kBF16, {2, width});
      x.put(Values(2 * width)); w.put(Values(width, 4)); residual.put(Values(2 * width, 11));
      vt::RmsNorm(*q, out.t, x.t, w.t, {1e-6f, gemma}, &residual.t);
      if (q == &qs.cpu) { expected = out.floats(); expected_residual = residual.floats(); }
      else { Close(out.floats(), expected, 2e-5f); CHECK(residual.floats() == expected_residual); }
    }
  }
  for (bool with_residual : {false, true}) {
    std::vector<float> expected;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      Buffer x(*q, DType::kBF16, {3, 129}), w(*q, DType::kBF16, {129});
      x.put(Values(3 * 129)); w.put(Values(129));
      vt::RmsNorm(*q, x.t, x.t, w.t, {1e-6f, true}, with_residual ? &x.t : nullptr);
      if (q == &qs.cpu) expected = x.floats(); else Close(x.floats(), expected, 0.008f);
    }
  }
}

TEST_CASE("XPU gather/scatter and embedding: padded rows, duplicate indices, invalid indices") {
  Queues qs;
  for (auto dtype : floats) {
    std::vector<float> expected_gather, expected_scatter, expected_embed;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      Buffer base(*q, dtype, {4, 132}), result(*q, dtype, {3, 129});
      Buffer idx(*q, DType::kI32, {3}), embed(*q, DType::kF32, {3, 132});
      const int32_t indices[] = {2, 0, 2}; idx.upload(indices); base.put(Values(4 * 132));
      vt::Embedding(*q, embed.t, base.t, idx.t);
      base.t.shape[1] = 129;
      vt::IndexSelect(*q, result.t, base.t, idx.t);
      if (q == &qs.cpu) { expected_gather = result.floats(); expected_embed = embed.floats(); }
      else { CHECK(result.floats() == expected_gather); CHECK(embed.floats() == expected_embed); }
      result.put(Values(3 * 129, 8)); vt::IndexCopy(*q, base.t, result.t, idx.t);
      if (q == &qs.cpu) expected_scatter = base.floats(); else CHECK(base.floats() == expected_scatter);
      const int32_t invalid[] = {0, -1, 4}; idx.upload(invalid);
      CHECK_THROWS(vt::IndexSelect(*q, result.t, base.t, idx.t));
    }
  }
  // Equal-dtype copies of I64 indices must never round through float.
  for (auto* q : {&qs.cpu, &qs.gpu}) {
    Buffer base(*q, DType::kI64, {3, 1}), idx(*q, DType::kI32, {3}), out(*q, DType::kI64, {3, 1});
    const int64_t values[] = {INT64_MAX, INT64_MIN, 16777217}; const int32_t indices[] = {2, 1, 0};
    base.upload(values); idx.upload(indices); vt::IndexSelect(*q, out.t, base.t, idx.t);
    const int64_t expected[] = {16777217, INT64_MIN, INT64_MAX};
    auto raw = out.raw(); CHECK(std::memcmp(raw.data(), expected, sizeof(expected)) == 0);
  }
}

TEST_CASE("XPU Matmul and MatmulBT: mixed dtypes, strided activation and real BA dimensions") {
  Queues qs;
  for (auto a_type : floats) for (auto b_type : floats) for (bool transpose : {false, true}) {
    std::vector<float> expected;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      const int k = 13, n = 9;
      Buffer a(*q, a_type, {3, transpose ? k + 4 : k});
      Buffer b(*q, b_type, {transpose ? n : k, transpose ? k : n});
      Buffer out(*q, DType::kF32, {3, n});
      a.put(Values(a.t.Numel())); b.put(Values(b.t.Numel(), 5)); a.t.shape[1] = k;
      if (transpose) vt::MatmulBT(*q, out.t, a.t, b.t); else vt::Matmul(*q, out.t, a.t, b.t);
      if (q == &qs.cpu) expected = out.floats(); else Close(out.floats(), expected, 1e-5f);
    }
  }
  std::vector<float> expected;
  for (auto* q : {&qs.cpu, &qs.gpu}) {
    Buffer a(*q, DType::kBF16, {1, 5120}), b(*q, DType::kF16, {48, 5120});
    Buffer out(*q, DType::kBF16, {1, 48}); a.put(Values(5120)); b.put(Values(48 * 5120));
    vt::MatmulBT(*q, out.t, a.t, b.t);
    if (q == &qs.cpu) expected = out.floats(); else CHECK(out.floats() == expected);
  }
}

TEST_CASE("XPU greedy: lowest tie, NaN/Inf contract, tails and production token readback") {
  Queues qs;
  for (int vocab : {1, 129, 248320}) {
    auto values = Values(6 * vocab);
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    values[0] = nan;
    for (int j = 0; j < vocab; ++j) { values[vocab + j] = -inf; values[2 * vocab + j] = inf; }
    if (vocab > 1) { values[3 * vocab + 1] = nan; values[4 * vocab] = 9; values[5 * vocab - 1] = 9; }
    std::vector<unsigned char> expected;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      Buffer logits(*q, DType::kF32, {6, vocab}), ids(*q, DType::kI64, {6}); logits.put(values);
      vt::GreedyArgmax(*q, ids.t, logits.t);
      if (q == &qs.cpu) expected = ids.raw(); else CHECK(ids.raw() == expected);
      vllm::v1::Sampler sampler; vllm::v1::SamplingMetadata metadata;
      metadata.all_greedy = true; metadata.all_random = false; metadata.no_penalties = true;
      const auto output = sampler.forward(*q, logits.t, metadata);
      REQUIRE(output.sampled_token_ids.size() == 6);
      for (size_t row = 0; row < 6; ++row) {
        int64_t id; std::memcpy(&id, expected.data() + row * sizeof(id), sizeof(id));
        REQUIRE(output.sampled_token_ids[row].size() == 1);
        CHECK(output.sampled_token_ids[row][0] == id);
      }
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  auto& platform = vllm::platforms::GetPlatform(vt::DeviceType::kXPU);
  CHECK(platform.needs_weight_staging()); CHECK_FALSE(platform.supports_graph_capture());
  CHECK_FALSE(platform.get_device_capability().present());
}
