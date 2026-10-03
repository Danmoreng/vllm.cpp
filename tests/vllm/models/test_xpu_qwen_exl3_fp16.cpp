// Synthetic native execution check, not pinned-checkpoint/producer parity.
// Exercises the dense model's FP16 policy through GDN, attention, MLP and head.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/dense_exl3_linear.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/model_executor/models/qwen3_5_gdn_block.h"
#include "vllm/model_executor/models/qwen3_5_attn_block.h"
#include "vllm/model_executor/models/act_dump.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/exl3_fixture.h"
#include "vt/xpu_test_helpers.h"

namespace {
using vt::DType;
using vllm::OwnedTensor;

OwnedTensor Own(DType dtype, std::initializer_list<int64_t> shape,
                const void* data, size_t bytes) {
  OwnedTensor w;
  w.dtype = dtype;
  w.rank = static_cast<int>(shape.size());
  std::copy(shape.begin(), shape.end(), w.shape);
  w.bytes.resize(bytes);
  std::memcpy(w.bytes.data(), data, bytes);
  return w;
}

OwnedTensor Values(DType dtype, std::initializer_list<int64_t> shape, uint32_t seed) {
  int64_t n = 1;
  for (int64_t dim : shape) n *= dim;
  exl3_test::Rng rng;
  rng.s = seed;
  std::vector<float> f(static_cast<size_t>(n));
  for (auto& v : f) v = rng.next(0.04f);
  if (dtype == DType::kF32) return Own(dtype, shape, f.data(), f.size() * 4);
  std::vector<uint16_t> h(f.size());
  for (size_t i = 0; i < f.size(); ++i) h[i] = vt::F32ToF16(f[i]);
  return Own(dtype, shape, h.data(), h.size() * 2);
}

vllm::Exl3Weight Packed(int64_t k, int64_t n, int bits, uint32_t seed,
                        const std::string& name) {
  auto f = exl3_test::MakeFixture(k, n, bits, seed);
  for (auto& v : f.svh) v = vt::F32ToF16(0.03125f);
  vllm::Exl3Weight w;
  w.name = name;
  w.codebook = 2;
  w.trellis = Own(DType::kI8, {k / 16, n / 16, 32 * bits},
                   f.trellis.data(), f.trellis.size() * 2);
  w.suh = Own(DType::kF16, {k}, f.suh.data(), f.suh.size() * 2);
  w.svh = Own(DType::kF16, {n}, f.svh.data(), f.svh.size() * 2);
  return w;
}

vllm::HfConfig Config() {
  vllm::HfConfig c;
  c.hidden_size = c.vocab_size = c.intermediate_size = 128;
  c.num_hidden_layers = 2;
  c.num_attention_heads = c.num_key_value_heads = 1;
  c.head_dim = 128;
  c.linear_num_key_heads = c.linear_num_value_heads = 1;
  c.linear_key_head_dim = c.linear_value_head_dim = 128;
  c.linear_conv_kernel_dim = 4;
  c.layer_types = {"linear_attention", "full_attention"};
  c.torch_dtype = "bfloat16";  // exported config; runtime policy is FP16
  c.mamba_ssm_dtype = "float32";
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 10000;
  c.rotary_dim = 64;
  c.max_position_embeddings = 8;
  return c;
}

vllm::Qwen3_5DenseWeights Weights(const vllm::HfConfig& c, bool split_ba) {
  vllm::Qwen3_5DenseWeights w;
  w.exl3_checkpoint = true;
  w.precision = vllm::ResolveQwen3_5DensePrecision(c, false, true);
  w.embed_tokens = Values(DType::kF16, {128, 128}, 1);
  w.final_norm = Values(DType::kF16, {128}, 2);
  w.lm_head_exl3 = Packed(128, 128, 6, 3, "lm_head");
  for (int layer = 0; layer < 2; ++layer) {
    vllm::Qwen3_5DenseLayerWeights l;
    const uint32_t seed = 100 + 100 * layer;
    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
    l.input_layernorm = Values(DType::kF16, {128}, seed);
    l.post_attention_layernorm = Values(DType::kF16, {128}, seed + 1);
    l.mlp.gate_proj_exl3 = Packed(128, 128, 4, seed + 2, prefix + "mlp.gate_proj");
    l.mlp.up_proj_exl3 = Packed(128, 128, 4, seed + 3, prefix + "mlp.up_proj");
    l.mlp.down_proj_exl3 = Packed(128, 128, 4, seed + 4, prefix + "mlp.down_proj");
    l.is_linear_attention = layer == 0;
    if (l.is_linear_attention) {
      auto& g = l.gdn;
      g.in_proj_qkv_exl3 = Packed(128, 384, 4, seed + 5, prefix + "linear_attn.in_proj_qkv");
      g.in_proj_z_exl3 = Packed(128, 128, 4, seed + 6, prefix + "linear_attn.in_proj_z");
      g.out_proj_exl3 = Packed(128, 128, 4, seed + 7, prefix + "linear_attn.out_proj");
      g.in_proj_ba = Values(DType::kF16, {2, 128}, seed + 8);
      g.in_proj_ba.nk = true;
      if (split_ba) {
        g.in_proj_b = Own(DType::kF16, {1, 128}, g.in_proj_ba.bytes.data(), 256);
        g.in_proj_a = Own(DType::kF16, {1, 128}, g.in_proj_ba.bytes.data() + 256, 256);
        g.in_proj_b.nk = g.in_proj_a.nk = true;
        g.in_proj_ba = OwnedTensor{};
      }
      g.conv1d_weight = Values(DType::kF16, {384, 4}, seed + 9);
      g.norm_weight = Values(DType::kF16, {128}, seed + 10);
      g.a_log = Values(DType::kF32, {1}, seed + 11);
      g.dt_bias = Values(DType::kF32, {1}, seed + 12);
    } else {
      auto& a = l.attn;
      a.q_proj_exl3 = Packed(128, 256, 4, seed + 5, prefix + "self_attn.q_proj");
      a.k_proj_exl3 = Packed(128, 128, 4, seed + 6, prefix + "self_attn.k_proj");
      a.v_proj_exl3 = Packed(128, 128, 4, seed + 7, prefix + "self_attn.v_proj");
      a.o_proj_exl3 = Packed(128, 128, 4, seed + 8, prefix + "self_attn.o_proj");
      a.q_norm = Values(DType::kF16, {128}, seed + 9);
      a.k_norm = Values(DType::kF16, {128}, seed + 10);
    }
    w.layers.push_back(std::move(l));
  }
  return w;
}
}  // namespace

namespace {
std::vector<float> CapturedFloats(const vllm::StTensor& value) {
  REQUIRE((value.dtype == "F16" || value.dtype == "F32"));
  const size_t size = value.dtype == "F16" ? 2 : 4;
  std::vector<float> result(value.nbytes/size);
  for (size_t i = 0; i < result.size(); ++i) {
    if (size == 4) std::memcpy(&result[i], value.data + i*4, 4);
    else { uint16_t bit; std::memcpy(&bit, value.data + i*2, 2); result[i] = vt::F16ToF32(bit); }
  }
  return result;
}

void CapturedClose(const std::string& stage, const std::vector<float>& got,
                   const vllm::StTensor& expected, float rtol, float atol) {
  const auto want = CapturedFloats(expected);
  REQUIRE(got.size() == want.size());
  double error = 0, norm = 0; float max_error = 0; size_t different = 0;
  bool finite = true;
  for (size_t i = 0; i < got.size(); ++i) {
    finite &= std::isfinite(got[i]) && std::isfinite(want[i]);
    const double delta = double(got[i]) - want[i]; error += delta*delta; norm += double(want[i])*want[i];
    max_error = std::max(max_error, std::abs(got[i] - want[i])); different += got[i] != want[i];
  }
  REQUIRE(finite); REQUIRE(norm > 0);
  std::cout << "REAL_BLOCK_STAGE stage=" << stage << " relative=" << std::sqrt(error/norm)
            << " max_error=" << max_error << " different_values=" << different << '\n';
  CHECK(std::sqrt(error/norm) < 2e-3);
  xpu_test::Close(got, want, rtol, atol);
}

struct RealLayer0 {
  vllm::HfConfig config;
  std::optional<vllm::SafetensorsFile> shard, oracle;
  vllm::Qwen3_5DenseLayerWeights layer;
  RealLayer0() {
    const char* model = std::getenv("VT_B70_EXL3_MODEL");
    const char* captures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
    if (!model || !captures) {
      std::cerr << "SKIP: set VT_B70_EXL3_MODEL and VT_B70_EXL3_S1_FIXTURES.\n";
      std::exit(77);
    }
    config = vllm::LoadHfConfig((std::filesystem::path(model) / "config.json").string());
    shard = vllm::SafetensorsFile::Open((std::filesystem::path(model) / "model-00001-of-00002.safetensors").string());
    const auto get = [&](const std::string& name) -> const vllm::StTensor& { return shard->Get(name); };
    const auto has = [&](const std::string& name) {
      return std::find(shard->Names().begin(), shard->Names().end(), name) != shard->Names().end();
    };
    layer = vllm::LoadQwen3_5DenseLayer(get, has, "linear_attention", 0, "model.language_model.");
    oracle = vllm::SafetensorsFile::Open((std::filesystem::path(captures) / "real_block_oracle.safetensors").string());
    REQUIRE(config.hidden_size == 5120);
  }
};
}  // namespace

TEST_CASE("XPU EXL3 real attention: identical original operands FP8 bytes P128 D1") {
  const char* captures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  if (!captures) { std::cerr << "Set VT_B70_EXL3_S1_FIXTURES.\n"; std::exit(77); }
  const auto dir = std::filesystem::path(captures);
  const auto oracle = vllm::SafetensorsFile::Open((dir / "real_attention3_oracle.safetensors").string());
  std::ifstream record_stream(dir / "real_attention3_oracle.json");
  REQUIRE(record_stream.good());
  const auto record = nlohmann::json::parse(record_stream);
  REQUIRE(record.at("active_cache_continuity_exact").get<bool>());
  const int32_t pblock = *reinterpret_cast<const int32_t*>(oracle.Get("p128_block_table").data);
  const int32_t dblock = *reinterpret_cast<const int32_t*>(oracle.Get("d1_block_table").data);
  REQUIRE(pblock >= 0); REQUIRE(dblock == pblock);
  const int64_t blocks = int64_t(pblock) + 1;
  constexpr int64_t page = 1600, heads = 24, kvheads = 4, dim = 256;
  constexpr int64_t block_bytes = page * kvheads * 2 * dim;
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  // Match the observed interleaved physical layout, including head/page
  // strides. Every inactive byte starts poisoned and must remain untouched.
  xpu_test::Buffer storage(gpu.q, DType::kI8, {blocks, page, kvheads, 2 * dim});
  std::vector<uint8_t> poison(static_cast<size_t>(blocks * block_bytes), 0x7f);
  storage.upload(poison.data());
  auto kc = storage.tensor;
  kc.shape[3] = dim;
  auto vc = kc;
  vc.data = static_cast<uint8_t*>(vc.data) + dim;
  for (const std::string phase : {"p128", "d1"}) {
    const int64_t rows = phase == "p128" ? 128 : 1;
    const int64_t length = phase == "p128" ? 128 : 129;
    const auto& step = record.at("steps").at(phase == "p128" ? 0 : 1);
    const float kscale = step.at("scales").at("_k_scale").at("values").get<float>();
    const float vscale = step.at("scales").at("_v_scale").at("values").get<float>();
    REQUIRE(kscale > 0); REQUIRE(vscale > 0);
    const auto get = [&](const std::string& label) -> const vllm::StTensor& {
      return oracle.Get(phase + "_" + label);
    };
    REQUIRE(get("q_rope").dtype == "F16");
    REQUIRE(get("k_rope").dtype == "F16");
    REQUIRE(get("value").dtype == "F16");
    xpu_test::Buffer query(gpu.q, DType::kF16, {rows, heads, dim});
    xpu_test::Buffer key(gpu.q, DType::kF16, {rows, kvheads, dim});
    xpu_test::Buffer value(gpu.q, DType::kF16, {rows, kvheads, dim});
    xpu_test::Buffer output(gpu.q, DType::kF16, {rows, heads, dim});
    xpu_test::Buffer slots(gpu.q, DType::kI64, {rows});
    xpu_test::Buffer table(gpu.q, DType::kI32, {1, 1});
    xpu_test::Buffer lengths(gpu.q, DType::kI32, {1});
    xpu_test::Buffer offsets(gpu.q, DType::kI32, {2});
    query.upload(get("q_rope").data); key.upload(get("k_rope").data);
    value.upload(get("value").data); slots.upload(get("slot_mapping").data);
    table.upload(get("block_table").data); lengths.upload(get("seq_lens").data);
    offsets.upload(get("query_start_loc").data);
    vt::ReshapeAndCacheFp8(gpu.q, key.tensor, value.tensor, kc, vc, slots.tensor,
                           vt::Fp8KVCacheDataType::kFp8E4M3, kscale, vscale);
    const auto raw = storage.download();
    const auto& expected_k = get("key_bytes_after");
    const auto& expected_v = get("value_bytes_after");
    REQUIRE(expected_k.dtype == "U8"); REQUIRE(expected_v.dtype == "U8");
    REQUIRE(expected_k.nbytes == static_cast<size_t>(length * kvheads * dim));
    REQUIRE(expected_v.nbytes == expected_k.nbytes);
    size_t key_diff = 0, value_diff = 0, inactive_diff = 0;
    for (int64_t b = 0; b < blocks; ++b)
      for (int64_t p = 0; p < page; ++p)
        for (int64_t h = 0; h < kvheads; ++h)
          for (int64_t c = 0; c < 2 * dim; ++c) {
            const size_t physical = static_cast<size_t>(b * block_bytes + p * kvheads * 2 * dim + h * 2 * dim + c);
            if (b == pblock && p < length) {
              const size_t logical = static_cast<size_t>((p * kvheads + h) * dim + c % dim);
              if (c < dim) key_diff += raw[physical] != expected_k.data[logical];
              else value_diff += raw[physical] != expected_v.data[logical];
            } else inactive_diff += raw[physical] != 0x7f;
          }
    std::cout << "REAL_ATTN_BYTES phase=" << phase << " key_diff=" << key_diff
              << " value_diff=" << value_diff << " inactive_diff=" << inactive_diff << '\n';
    CHECK(key_diff == 0); CHECK(value_diff == 0); CHECK(inactive_diff == 0);
    vt::PagedAttentionArgs args;
    args.scale = 1.0f / 16.0f;
    args.causal = true;
    args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
    args.k_scale = kscale; args.v_scale = vscale;
    args.max_seq_len = length;
    const int32_t host_offsets[] = {0, static_cast<int32_t>(rows)};
    args.query_start_loc_host = host_offsets;
    vt::PagedAttention(gpu.q, output.tensor, query.tensor, kc, vc, table.tensor,
                        lengths.tensor, offsets.tensor, args);
    const auto got = output.floats();
    const auto want = CapturedFloats(get("attention_output"));
    REQUIRE(got.size() == want.size());
    size_t finite_failures = 0, band_failures = 0; float maximum = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      finite_failures += !std::isfinite(got[i]) || !std::isfinite(want[i]);
      const float delta = std::abs(got[i] - want[i]);
      maximum = std::max(maximum, delta);
      band_failures += delta > 0.003f + 0.01f * std::abs(want[i]);
    }
    std::cout << "REAL_ATTN_CORE phase=" << phase << " max_error=" << maximum
              << " band_failures=" << band_failures << '\n';
    CHECK(finite_failures == 0); CHECK(band_failures == 0);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 attention RoPE: actual FP16 operands and coefficients") {
  const char* fixtures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  if (!fixtures || !model) std::exit(77);
  const auto oracle = vllm::SafetensorsFile::Open(
      (std::filesystem::path(fixtures) / "real_attention3_rope_oracle.safetensors").string());
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const auto shard = vllm::SafetensorsFile::Open(
      (std::filesystem::path(model) / "model-00001-of-00002.safetensors").string());
  const auto get = [&](const std::string& n) -> const vllm::StTensor& { return shard.Get(n); };
  const auto has = [&](const std::string& n) {
    return std::find(shard.Names().begin(), shard.Names().end(), n) != shard.Names().end();
  };
  auto layer = vllm::LoadQwen3_5DenseLayer(get, has, "full_attention", 3, "model.language_model.");
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  const auto qw = vllm::dense_attn::ResidentWeightF32(d, layer.attn.q_norm, {256});
  const auto kw = vllm::dense_attn::ResidentWeightF32(d, layer.attn.k_norm, {256});
  std::vector<float> coefficients = CapturedFloats(oracle.Get("p128_rope_cos_sin"));
  const auto last = CapturedFloats(oracle.Get("d1_rope_cos_sin"));
  coefficients.insert(coefficients.end(), last.begin(), last.end());
  REQUIRE(coefficients.size() == 129 * 64);
  xpu_test::Buffer cache(gpu.q, DType::kF32, {129, 64});
  cache.put(coefficients);
  vt::RopeArgs args{10000000.0f, 64};
  args.fp16_intermediates = true;
  for (const std::string phase : {"p128", "d1"}) {
    const int64_t rows = phase == "p128" ? 128 : 1;
    xpu_test::Buffer query(gpu.q, DType::kF16, {rows, 24, 256});
    xpu_test::Buffer key(gpu.q, DType::kF16, {rows, 4, 256});
    xpu_test::Buffer positions(gpu.q, DType::kI32, {rows});
    std::vector<int32_t> ids(rows);
    std::iota(ids.begin(), ids.end(), phase == "p128" ? 0 : 128);
    positions.upload(ids.data());
    query.upload(oracle.Get(phase + "_rope_q_input").data);
    key.upload(oracle.Get(phase + "_rope_k_input").data);
    vt::RopeFromCache(gpu.q, query.tensor, &key.tensor, positions.tensor, cache.tensor, args);
    for (const auto& [label, actual] : {std::pair<std::string, std::vector<uint8_t>>{"q", query.download()},
                                       {"k", key.download()}}) {
      const auto& expected = oracle.Get(phase + "_" + label + "_rope");
      const std::vector<uint8_t> want(expected.data, expected.data + expected.nbytes);
      xpu_test::SameBytes(actual, want);
    }
    xpu_test::Buffer merged(gpu.q, DType::kF16, {rows, 14336});
    xpu_test::Buffer gates(gpu.q, DType::kF32, {rows, 24, 256});
    merged.upload(oracle.Get(phase + "_qkv_output").data);
    auto qgate = merged.tensor.Slice(1, 0, 12288);
    auto raw_key = merged.tensor.Slice(1, 12288, 13312);
    auto active_cache = cache.tensor.Slice(0, phase == "p128" ? 0 : 128,
                                         phase == "p128" ? 128 : 129);
    vt::AttnQkNormRopeGate(gpu.q, query.tensor, key.tensor, gates.tensor,
                          qgate, raw_key, qw, kw, active_cache,
                          vt::RmsNormArgs{1e-6f, true}, args);
    CapturedClose(phase + ".fp16_fused_q", query.floats(), oracle.Get(phase + "_q_rope"), 0.01f, 0.003f);
    CapturedClose(phase + ".fp16_fused_k", key.floats(), oracle.Get(phase + "_k_rope"), 0.01f, 0.003f);
  }
  xpu_test::Buffer positions(gpu.q, DType::kI32, {129});
  std::vector<int32_t> ids(129); std::iota(ids.begin(), ids.end(), 0); positions.upload(ids.data());
  xpu_test::Buffer generated(gpu.q, DType::kF32, {129, 64});
  args.linear_scaling_factor = 1.0f;
  vt::RopeCosSinCache(gpu.q, generated.tensor, positions.tensor, args);
  auto values = generated.floats();
  size_t different = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    const auto half = vt::F32ToF16(values[i]);
    if (vt::F16ToF32(half) == coefficients[i]) continue;
    if (different < 16)
      std::cout << "REAL_ROPE_COEFFICIENT_DIFFERENCE position=" << i / 64
                << " column=" << i % 64 << " generated=" << std::hexfloat << values[i]
                << " captured_half=" << coefficients[i] << std::defaultfloat
                << " generated_half_bits=" << half
                << " captured_half_bits=" << vt::F32ToF16(coefficients[i]) << '\n';
    ++different;
  }
  std::cout << "REAL_ROPE_COEFFICIENTS half_differences=" << different << '\n';
  CHECK(different == 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real attention mixer: own cache P128 D1 from original hidden") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* fixtures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  const char* dumps = vllm::actdump::StreamDir();
  if (!model || !fixtures || !dumps) {
    std::cerr << "Set VT_B70_EXL3_MODEL, VT_B70_EXL3_S1_FIXTURES and VT_DUMP_ACT.\n";
    std::exit(77);
  }
  const auto oracle = vllm::SafetensorsFile::Open(
      (std::filesystem::path(fixtures) / "real_attention3_oracle.safetensors").string());
  const auto shard = vllm::SafetensorsFile::Open(
      (std::filesystem::path(model) / "model-00001-of-00002.safetensors").string());
  const auto config = vllm::LoadHfConfig((std::filesystem::path(model) / "config.json").string());
  const auto get = [&](const std::string& n) -> const vllm::StTensor& { return shard.Get(n); };
  const auto has = [&](const std::string& n) {
    return std::find(shard.Names().begin(), shard.Names().end(), n) != shard.Names().end();
  };
  auto layer = vllm::LoadQwen3_5DenseLayer(get, has, "full_attention", 3, "model.language_model.");
  REQUIRE(layer.attn.IsExl3()); REQUIRE(config.hidden_size == 5120);
  int32_t active;
  std::memcpy(&active, oracle.Get("p128_block_table").data, sizeof(active));
  REQUIRE(active >= 0); REQUIRE(active < 212);
  const int64_t blocks = active + 1;
  constexpr int64_t page = 1600, hkv = 4, dim = 256, columns = hkv * dim;
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  xpu_test::Buffer storage(gpu.q, DType::kI8, {blocks, 2, page, columns});
  std::vector<uint8_t> poison(storage.bytes, 0x7f);
  storage.upload(poison.data());
  vllm::PagedKvCache cache;
  cache.data = storage.tensor.data; cache.dtype = DType::kI8;
  cache.num_blocks = blocks; cache.block_size = page;
  cache.num_kv_heads = hkv; cache.head_size = dim;
  cache.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
  cache.k_scale = cache.v_scale = 1.0f;
  for (int step_idx = 0; step_idx < 2; ++step_idx) {
    const std::string phase = step_idx ? "d1" : "p128";
    const int32_t rows = step_idx ? 1 : 128;
    const auto captured = [&](const std::string& label) -> const vllm::StTensor& {
      return oracle.Get(phase + "_" + label);
    };
    REQUIRE(captured("positions").dtype == "I64");
    REQUIRE(captured("positions").shape == std::vector<int64_t>{3, rows});
    std::vector<int32_t> positions(rows);
    for (int axis = 0; axis < 3; ++axis) for (int32_t row = 0; row < rows; ++row) {
      int64_t p;
      std::memcpy(&p, captured("positions").data + (axis * rows + row) * 8, 8);
      CHECK(p == (step_idx ? 128 : row)); positions[row] = static_cast<int32_t>(p);
    }
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1; am.num_actual_tokens = rows;
    am.query_start_loc = am.query_start_loc_cpu = {0, rows};
    am.seq_lens = am.seq_lens_cpu = {step_idx ? 129 : 128};
    am.num_computed_tokens_cpu = {step_idx ? 128 : 0};
    am.max_query_len = rows; am.max_seq_len = am.seq_lens[0];
    am.block_table_num_cols = 1; am.block_table_tensor = {active};
    am.slot_mapping.resize(rows);
    REQUIRE(captured("slot_mapping").dtype == "I64");
    REQUIRE(captured("slot_mapping").nbytes == static_cast<size_t>(rows) * 8);
    std::memcpy(am.slot_mapping.data(), captured("slot_mapping").data, rows * 8);
    int32_t expected_block;
    std::memcpy(&expected_block, captured("block_table").data, 4);
    CHECK(expected_block == active);
    for (int32_t row = 0; row < rows; ++row)
      CHECK(am.slot_mapping[row] == int64_t(active) * page + positions[row]);
    vllm::dense_attn::DBuf input(d, DType::kF16, {rows, 5120}, captured("input_norm_output").data);
    auto step = vllm::BuildFullAttnStepInputs(gpu.q, positions, am, config, DType::kF16);
    const vllm::actdump::LayerScope scope(step_idx, 3);
    const auto output = vllm::RunFullAttnBlockPaged(gpu.q, layer.attn, config,
        input.t(), step, am, cache, rows, DType::kF16);
    REQUIRE(output.tensor.dtype == DType::kF16);
    const auto read_stage = [&](const std::string& label, const std::string& dtype, int64_t width) {
      const auto path = std::filesystem::path(dumps) /
          ("s" + std::to_string(step_idx) + "_l3_attn_" + label + ".bin");
      const size_t bytes = static_cast<size_t>(rows * width) * (dtype == "F16" ? 2 : 4);
      REQUIRE(std::filesystem::file_size(path) == bytes);
      std::vector<uint8_t> raw(bytes);
      std::ifstream stream(path, std::ios::binary);
      stream.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
      REQUIRE(stream.good());
      vllm::StTensor tensor; tensor.dtype = dtype; tensor.data = raw.data(); tensor.nbytes = bytes;
      return CapturedFloats(tensor);
    };
    const auto qkv = CapturedFloats(captured("qkv_output"));
    REQUIRE(qkv.size() == static_cast<size_t>(rows) * 14336);
    for (const auto& [label, start, width] :
         {std::tuple<std::string, int, int>{"qgate", 0, 12288}, {"key_raw", 12288, 1024},
          {"value", 13312, 1024}}) {
      std::vector<float> want;
      for (int32_t row = 0; row < rows; ++row)
        want.insert(want.end(), qkv.begin() + row * 14336 + start,
                    qkv.begin() + row * 14336 + start + width);
      const auto actual = read_stage(label, "F16", width);
      CHECK(actual == want);
    }
    const auto compare = [&](const std::string& label, const std::vector<float>& actual,
                             const std::vector<float>& expected) {
      REQUIRE(actual.size() == expected.size());
      size_t failures = 0, different = 0, nonfinite = 0; float maximum = 0;
      double error = 0, norm = 0;
      for (size_t i = 0; i < actual.size(); ++i) {
        nonfinite += !std::isfinite(actual[i]) || !std::isfinite(expected[i]);
        const float delta = std::abs(actual[i] - expected[i]);
        failures += delta > 0.003f + 0.01f * std::abs(expected[i]);
        different += actual[i] != expected[i]; maximum = std::max(maximum, delta);
        error += double(delta) * delta; norm += double(expected[i]) * expected[i];
      }
      std::cout << "REAL_ATTN_MIXER phase=" << phase << " stage=" << label
                << " relative=" << std::sqrt(error/norm) << " max_error=" << maximum
                << " different=" << different << " band_failures=" << failures << '\n';
      CHECK(nonfinite == 0); CHECK(failures == 0); CHECK(std::sqrt(error/norm) < 2e-3);
    };
    for (const auto& [label, dtype, width, reference] : {
         std::tuple<std::string, std::string, int64_t, std::string>{"q_rope", "F16", 6144, "q_rope"},
         {"k_rope", "F16", 1024, "k_rope"}, {"gate", "F32", 6144, "gate"},
         {"core", "F16", 6144, "attention_output"}, {"gated", "F16", 6144, "gated_attention"}})
      compare(label, read_stage(label, dtype, width), CapturedFloats(captured(reference)));
    xpu_test::Buffer copy(gpu.q, DType::kF16, {rows, 5120});
    vt::Copy(gpu.q, copy.tensor, output.tensor);
    compare("mixer", copy.floats(), CapturedFloats(captured("mixer_output")));
    const auto raw = storage.download();
    size_t key_diff = 0, value_diff = 0, inactive_diff = 0;
    for (int64_t b = 0; b < blocks; ++b) for (int64_t plane = 0; plane < 2; ++plane)
      for (int64_t p = 0; p < page; ++p) for (int64_t c = 0; c < columns; ++c) {
        const size_t offset = static_cast<size_t>(((b * 2 + plane) * page + p) * columns + c);
        if (b == active && p < am.seq_lens[0]) {
          const auto& expected = captured(plane ? "value_bytes_after" : "key_bytes_after");
          const bool differs = raw[offset] != expected.data[p * columns + c];
          if (plane) value_diff += differs; else key_diff += differs;
        } else inactive_diff += raw[offset] != 0x7f;
      }
    std::cout << "REAL_ATTN_MIXER_BYTES phase=" << phase << " key_diff=" << key_diff
              << " value_diff=" << value_diff << " inactive_diff=" << inactive_diff << '\n';
    CHECK(key_diff == 0); CHECK(value_diff == 0); CHECK(inactive_diff == 0);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real block GDN prep: original FP32 Conv normalization boundary") {
  RealLayer0 real;
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  const auto producer = vllm::SafetensorsFile::Open(
      (std::filesystem::path(std::getenv("VT_B70_EXL3_S1_FIXTURES")) / "real_gdn_oracle.safetensors").string());
  const auto& raw_qkvz = real.oracle->Get("p128_qkvz_output");
  const auto& raw_ba = real.oracle->Get("p128_ba_output");
  vllm::dense_attn::DBuf qkvz(d, DType::kF16, {128, 16384}, raw_qkvz.data);
  vllm::dense_attn::DBuf ba(d, DType::kF16, {128, 96}, raw_ba.data);
  auto input = qkvz.t().Slice(1, 0, 10240);
  auto b = ba.t().Slice(1, 0, 48), a = ba.t().Slice(1, 48, 96);
  xpu_test::Buffer conv(gpu.q, DType::kF32, {128, 10240});
  xpu_test::Buffer state(gpu.q, DType::kF32, {1, 10240, 3});
  state.put(std::vector<float>(10240*3, 7));
  xpu_test::Buffer qsl(gpu.q, DType::kI32, {2}), initial(gpu.q, DType::kI32, {1});
  const int32_t offsets[] = {0, 128}, cold[] = {0};
  qsl.upload(offsets); initial.upload(cold);
  auto cw = vllm::dense_attn::ResidentWeight(d, real.layer.gdn.conv1d_weight, {10240, 4});
  auto alog = vllm::dense_attn::ResidentWeight(d, real.layer.gdn.a_log, {48});
  auto bias = vllm::dense_attn::ResidentWeight(d, real.layer.gdn.dt_bias, {48});
  vt::CausalConv1dFwd(gpu.q, conv.tensor, input, cw, nullptr, state.tensor,
                     qsl.tensor, initial.tensor, vt::CausalConv1dArgs{true});
  if (const char* dir = vllm::actdump::StreamDir()) {
    const auto path = std::filesystem::path(dir) / "s0_l0_gdn_prep_conv.bin";
    VT_CHECK(!std::filesystem::exists(path), "GDN prep dump must not overwrite an existing receipt");
    const auto bytes = conv.download();
    vllm::actdump::WriteBlob("VT_DUMP_ACT", dir, 0, 0, "gdn_prep_conv",
                            DType::kF32, 128, 10240, bytes.data(), bytes.size());
  }
  xpu_test::Buffer q(gpu.q, DType::kF16, {128, 16, 128});
  xpu_test::Buffer k(gpu.q, DType::kF16, {128, 16, 128});
  xpu_test::Buffer v(gpu.q, DType::kF16, {128, 48, 128});
  xpu_test::Buffer g(gpu.q, DType::kF32, {128, 48});
  xpu_test::Buffer beta(gpu.q, DType::kF32, {128, 48});
  vt::GdnPostConv(gpu.q, q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor,
                  conv.tensor, a, b, alog, bias, vt::GdnPostConvArgs{1e-6f, true});
  for (const auto& [label, buffer] :
       std::vector<std::pair<std::string, xpu_test::Buffer*>>{{"q", &q}, {"k", &k}, {"v", &v}}) {
    CAPTURE(label);
    auto expected = CapturedFloats(producer.Get("p128_conv_" + label));
    const auto actual = buffer->floats();
    expected.resize(actual.size());  // discard original zero-padded capacity
    const auto different = std::inner_product(actual.begin(), actual.end(), expected.begin(),
        size_t{0}, std::plus<size_t>(), [](float x, float y) { return size_t(x != y); });
    std::cout << "REAL_GDN_PREP stage=" << label << " different_values=" << different << '\n';
    CHECK(different == 0);
  }
  const auto producer_beta = CapturedFloats(producer.Get("p128_conv_b"));
  const auto actual_beta = beta.floats();
  const int padded_rows = static_cast<int>(producer.Get("p128_conv_b").shape[1]);
  for (int t = 0; t < 128; ++t) for (int h = 0; h < 48; ++h)
    CHECK(actual_beta[t*48+h] == producer_beta[h*padded_rows+t]);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real GDN raw gate: native FP16 P128 D1 state chain") {
  RealLayer0 real;
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  const auto producer = vllm::SafetensorsFile::Open(
      (std::filesystem::path(std::getenv("VT_B70_EXL3_S1_FIXTURES")) / "real_gdn_oracle.safetensors").string());
  const auto upload = [&](const char* name, int heads) {
    return vllm::dense_attn::DBuf(d, DType::kF16, {128, heads, 128},
                                 producer.Get(name).data);
  };
  auto q = upload("p128_conv_q", 16), k = upload("p128_conv_k", 16);
  auto v = upload("p128_conv_v", 48);
  const auto& raw_ba = real.oracle->Get("p128_ba_output");
  vllm::dense_attn::DBuf ba(d, DType::kF16, {128, 96}, raw_ba.data);
  auto raw_a = ba.t().Slice(1, 48, 96);  // Exercise the actual merged BA row stride.
  const auto captured_beta = CapturedFloats(producer.Get("p128_conv_b"));
  std::vector<float> token_beta(128 * 48);
  for (int t = 0; t < 128; ++t) for (int h = 0; h < 48; ++h)
    token_beta[t * 48 + h] = captured_beta[h * 191 + t];
  xpu_test::Buffer beta(gpu.q, DType::kF32, {128, 48});
  beta.put(token_beta);
  auto alog = vllm::dense_attn::ResidentWeight(d, real.layer.gdn.a_log, {48});
  auto bias = vllm::dense_attn::ResidentWeight(d, real.layer.gdn.dt_bias, {48});
  xpu_test::Buffer state(gpu.q, DType::kF32, {1, 48, 128, 128});
  state.put(std::vector<float>(48 * 128 * 128, 0));
  xpu_test::Buffer out(gpu.q, DType::kF16, {128, 48, 128});
  xpu_test::Buffer qsl(gpu.q, DType::kI32, {2});
  const int32_t offsets[] = {0, 128};
  qsl.upload(offsets);
  vt::GdnPrefillRawGate(gpu.q, out.tensor, q.t(), k.t(), v.t(), raw_a,
                        beta.tensor, alog, bias, state.tensor, qsl.tensor,
                        vt::GdnArgs{1.0f});
  for (const auto& [label, buffer, capture] :
       std::vector<std::tuple<const char*, xpu_test::Buffer*, const char*>>{
           {"core", &out, "p128_core"}, {"state", &state, "p128_ssm_state"}}) {
    const auto actual = buffer->floats();
    const auto expected = CapturedFloats(producer.Get(capture));
    REQUIRE(actual.size() == expected.size());
    const auto different = std::inner_product(actual.begin(), actual.end(), expected.begin(),
        size_t{0}, std::plus<size_t>(), [](float x, float y) { return size_t(x != y); });
    std::cout << "REAL_GDN_RAW_GATE stage=" << label << " different_values=" << different << '\n';
    CHECK(different == 0);
  }
  // Continue from the native P128 result, never a captured-state replacement.
  // A padded packed row and strided BA views retain the real input contracts.
  constexpr int slots = 5, active = 4, state_elements = 48 * 128 * 128;
  xpu_test::Buffer cache(gpu.q, DType::kF32, {slots, 48, 128, 128});
  cache.put(std::vector<float>(slots * state_elements, 0.125f));
  xpu_test::Buffer indices(gpu.q, DType::kI32, {1});
  const int32_t index[] = {active};
  indices.upload(index);
  vt::GdnStateScatter(gpu.q, cache.tensor, state.tensor, indices.tensor);
  std::vector<uint16_t> packed(10304, 0x7e00);  // poisoned physical tail
  std::memcpy(packed.data(), producer.Get("d1_conv_q").data, 2048 * 2);
  std::memcpy(packed.data() + 2048, producer.Get("d1_conv_k").data, 2048 * 2);
  std::memcpy(packed.data() + 4096, producer.Get("d1_conv_v").data, 6144 * 2);
  vllm::dense_attn::DBuf mixed_owner(d, DType::kF16, {1, 10304}, packed.data());
  auto mixed = mixed_owner.t().Slice(1, 0, 10240);
  std::vector<uint16_t> decode_ba(96);
  std::memcpy(decode_ba.data(), producer.Get("d1_conv_b").data, 48 * 2);
  std::memcpy(decode_ba.data() + 48, producer.Get("d1_conv_a").data, 48 * 2);
  vllm::dense_attn::DBuf decode_ba_owner(d, DType::kF16, {1, 96}, decode_ba.data());
  auto decode_b = decode_ba_owner.t().Slice(1, 0, 48);
  auto decode_a = decode_ba_owner.t().Slice(1, 48, 96);
  xpu_test::Buffer decode_out(gpu.q, DType::kF16, {1, 48, 128});
  vt::GdnPackedDecode(gpu.q, decode_out.tensor, mixed, decode_a, decode_b,
                      alog, bias, cache.tensor, indices.tensor,
                      vt::GdnArgs{1.0f / std::sqrt(128.0f)});
  const auto decode_actual = decode_out.floats();
  const auto decode_expected = CapturedFloats(producer.Get("d1_core"));
  REQUIRE(decode_actual.size() == decode_expected.size());
  const auto core_different = std::inner_product(decode_actual.begin(), decode_actual.end(),
      decode_expected.begin(), size_t{0}, std::plus<size_t>(),
      [](float x, float y) { return size_t(x != y); });
  const auto cache_actual = cache.floats();
  const auto state_expected = CapturedFloats(producer.Get("d1_ssm_state"));
  REQUIRE(state_expected.size() == state_elements);
  const auto state_different = std::inner_product(cache_actual.begin() + active * state_elements,
      cache_actual.begin() + (active + 1) * state_elements, state_expected.begin(),
      size_t{0}, std::plus<size_t>(), [](float x, float y) { return size_t(x != y); });
  const auto inactive_different = std::count_if(cache_actual.begin(),
      cache_actual.begin() + active * state_elements, [](float x) { return x != 0.125f; });
  std::cout << "REAL_GDN_NATIVE_CHAIN D1_core_diff=" << core_different
            << " D1_state_diff=" << state_different
            << " inactive_diff=" << inactive_different << '\n';
  CHECK(core_different == 0);
  CHECK(state_different == 0);
  CHECK(inactive_different == 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real block GDN: captured P128 and D1 mixer/state continuation") {
  RealLayer0 real;
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  constexpr int slots = 5, active = 4, channels = 10240, heads = 48, dim = 128, history = 3;
  REQUIRE(real.config.linear_num_value_heads == heads);
  REQUIRE(real.config.linear_value_head_dim == dim);
  REQUIRE(real.config.linear_key_head_dim == dim);
  REQUIRE(real.config.linear_conv_kernel_dim == history+1);
  // Preserve the captured discrete slot4. Inactive slots are initialized to
  // witnesses; cold active contents are deliberately nonzero and must be ignored.
  xpu_test::Buffer conv(gpu.q, DType::kF16, {slots, channels, history});
  xpu_test::Buffer ssm(gpu.q, DType::kF32, {slots, heads, dim, dim});
  std::vector<float> conv_seed(slots*channels*history, 0.125f);
  std::vector<float> ssm_seed(slots*heads*dim*dim, 0.125f);
  std::fill(conv_seed.begin()+active*channels*history, conv_seed.end(), -2.0f);
  std::fill(ssm_seed.begin()+active*heads*dim*dim, ssm_seed.end(), 7.0f);
  conv.put(conv_seed); ssm.put(ssm_seed);
  vllm::GdnStateCache state; state.conv_state = conv.tensor; state.ssm_state = ssm.tensor;
  for (bool prefill : {true, false}) {
    CAPTURE(prefill);
    const vllm::actdump::LayerScope dump_scope(prefill ? 0 : 1, 0);
    const int rows = prefill ? 128 : 1;
    const std::string phase = prefill ? "p128" : "d1";
    const auto& x = real.oracle->Get(phase + "_input_norm_output");
    REQUIRE(x.dtype == "F16"); REQUIRE(x.shape == std::vector<int64_t>{rows, 5120});
    vllm::dense_attn::DBuf input(d, DType::kF16, {rows, 5120}, x.data);
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1; am.num_actual_tokens = rows;
    am.query_start_loc = am.query_start_loc_cpu = {0, rows};
    am.seq_lens = am.seq_lens_cpu = {prefill ? 128 : 129};
    am.max_query_len = rows; am.max_seq_len = am.seq_lens[0];
    am.block_table_num_cols = 1; am.block_table_tensor = {0};
    am.slot_mapping.assign(rows, 0);
    vllm::v1::GDNAttentionMetadata gm;
    gm.num_actual_tokens = rows;
    gm.non_spec_state_indices_tensor = std::vector<int32_t>{active};
    gm.non_spec_query_start_loc = std::vector<int32_t>{0, rows};
    if (prefill) {
      gm.num_prefills = 1; gm.num_prefill_tokens = rows;
      gm.has_initial_state = std::vector<uint8_t>{0};
      gm.prefill_query_start_loc = std::vector<int32_t>{0, rows};
      gm.prefill_state_indices = std::vector<int32_t>{active};
      gm.prefill_has_initial_state = std::vector<uint8_t>{0};
      const auto chunks = vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = chunks.batch_ptr; gm.token_chunk_offset_ptr = chunks.token_chunk_offset_ptr;
    } else { gm.num_decodes = gm.num_decode_tokens = 1; }
    const auto& captured_pos = real.oracle->Get(phase + "_positions");
    REQUIRE(captured_pos.dtype == "I64"); REQUIRE(captured_pos.shape == std::vector<int64_t>{3, rows});
    std::vector<int32_t> positions(rows);
    for (int i = 0; i < rows; ++i) for (int axis = 0; axis < 3; ++axis) {
      int64_t p; std::memcpy(&p, captured_pos.data + (axis*rows+i)*8, 8);
      CHECK(p == (prefill ? i : 128)); positions[i] = static_cast<int32_t>(p);
    }
    auto step = vllm::BuildGdnStepInputs(gpu.q, positions, am, gm, slots);
    auto out = vllm::RunGdnBlockPaged(gpu.q, real.layer.gdn, real.config,
                                    input.t(), step, gm, state, rows, nullptr, DType::kF16);
    REQUIRE(out.tensor.dtype == DType::kF16);
    xpu_test::Buffer copy(gpu.q, DType::kF16, {rows, 5120});
    vt::Copy(gpu.q, copy.tensor, out.tensor);
    CapturedClose(phase+".mixer", copy.floats(), real.oracle->Get(phase+"_mixer_output"), 0.01f, 0.003f);
    const auto cv = conv.floats(), ss = ssm.floats();
    CHECK(std::equal(cv.begin(), cv.begin()+active*channels*history, conv_seed.begin()));
    CHECK(std::equal(ss.begin(), ss.begin()+active*heads*dim*dim, ssm_seed.begin()));
    std::vector<float> active_conv(channels*history);
    for (int c = 0; c < channels; ++c) for (int h = 0; h < history; ++h)
      active_conv[h*channels+c] = cv[(active*channels+c)*history+h];
    const auto expected_conv = CapturedFloats(real.oracle->Get(phase+"_conv_state_after"));
    CHECK(active_conv == expected_conv);
    const std::vector<float> active_ssm(ss.begin()+active*heads*dim*dim, ss.end());
    // Existing strict FP32 GDN state comparison band, fixed before first run.
    CapturedClose(phase+".ssm", active_ssm, real.oracle->Get(phase+"_ssm_state_after"), 1e-4f, 1e-5f);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real block GDN: D1 from matched original Conv and SSM state") {
  RealLayer0 real;
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  constexpr int slots = 5, active = 4, channels = 10240, heads = 48, dim = 128, history = 3;
  // This independent D1 gate restores identical original starting states.
  // It does not qualify P128 or replace the native-state continuation test.
  const auto captured_conv = CapturedFloats(real.oracle->Get("d1_conv_state_before"));
  const auto captured_ssm = CapturedFloats(real.oracle->Get("d1_ssm_state_before"));
  REQUIRE(captured_conv.size() == channels * history);
  REQUIRE(captured_ssm.size() == heads * dim * dim);
  std::vector<float> conv_seed(slots * channels * history, 0.125f);
  std::vector<float> ssm_seed(slots * heads * dim * dim, 0.125f);
  for (int c = 0; c < channels; ++c) for (int h = 0; h < history; ++h)
    conv_seed[(active * channels + c) * history + h] = captured_conv[h * channels + c];
  std::copy(captured_ssm.begin(), captured_ssm.end(), ssm_seed.begin() + active * heads * dim * dim);
  xpu_test::Buffer conv(gpu.q, DType::kF16, {slots, channels, history});
  xpu_test::Buffer ssm(gpu.q, DType::kF32, {slots, heads, dim, dim});
  conv.put(conv_seed); ssm.put(ssm_seed);
  vllm::GdnStateCache state; state.conv_state = conv.tensor; state.ssm_state = ssm.tensor;
  vllm::dense_attn::DBuf input(d, DType::kF16, {1, 5120},
                              real.oracle->Get("d1_input_norm_output").data);
  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = am.num_actual_tokens = 1;
  am.query_start_loc = am.query_start_loc_cpu = {0, 1};
  am.seq_lens = am.seq_lens_cpu = {129};
  am.max_query_len = 1; am.max_seq_len = 129;
  am.block_table_num_cols = 1; am.block_table_tensor = {0}; am.slot_mapping = {0};
  vllm::v1::GDNAttentionMetadata gm;
  gm.num_actual_tokens = gm.num_decodes = gm.num_decode_tokens = 1;
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{active};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1};
  auto step = vllm::BuildGdnStepInputs(gpu.q, {128}, am, gm, slots);
  const vllm::actdump::LayerScope dump_scope(1, 0);
  auto out = vllm::RunGdnBlockPaged(gpu.q, real.layer.gdn, real.config,
                                  input.t(), step, gm, state, 1, nullptr, DType::kF16);
  REQUIRE(out.tensor.dtype == DType::kF16);
  xpu_test::Buffer copy(gpu.q, DType::kF16, {1, 5120});
  vt::Copy(gpu.q, copy.tensor, out.tensor);
  CapturedClose("matched_d1.mixer", copy.floats(), real.oracle->Get("d1_mixer_output"), 0.01f, 0.003f);
  const auto cv = conv.floats(), ss = ssm.floats();
  CHECK(std::equal(cv.begin(), cv.begin() + active * channels * history, conv_seed.begin()));
  CHECK(std::equal(ss.begin(), ss.begin() + active * heads * dim * dim, ssm_seed.begin()));
  std::vector<float> active_conv(channels * history);
  for (int c = 0; c < channels; ++c) for (int h = 0; h < history; ++h)
    active_conv[h * channels + c] = cv[(active * channels + c) * history + h];
  CHECK(active_conv == CapturedFloats(real.oracle->Get("d1_conv_state_after")));
  const std::vector<float> active_ssm(ss.begin() + active * heads * dim * dim, ss.end());
  CapturedClose("matched_d1.ssm", active_ssm, real.oracle->Get("d1_ssm_state_after"), 1e-4f, 1e-5f);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real block MLP: captured P128 and D1 complete native MLP") {
  RealLayer0 real;
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  vllm::OwnedTensor empty;
  for (const std::string phase : {"p128", "d1"}) {
    CAPTURE(phase);
    const int rows = phase == "p128" ? 128 : 1;
    const auto& x = real.oracle->Get(phase+"_post_norm_output");
    REQUIRE(x.dtype == "F16"); REQUIRE(x.shape == std::vector<int64_t>{rows, 5120});
    vllm::dense_attn::DBuf input(d, DType::kF16, {rows, 5120}, x.data);
    auto act = vllm::dense_exl3::GateUp(d, input.t(), empty,
        real.layer.mlp.gate_proj_exl3, real.layer.mlp.up_proj_exl3,
        real.config.intermediate_size, &real.layer.mlp.gate_up_exl3);
    std::vector<uint16_t> raw(rows*real.config.intermediate_size);
    act.Download(d, raw.data());
    std::vector<float> activated(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) activated[i] = vt::F16ToF32(raw[i]);
    CapturedClose(phase+".swiglu", activated, real.oracle->Get(phase+"_swiglu_output"), 0.002f, 1e-4f);
    auto down = vllm::dense_exl3::Linear(d, act.t(), empty, real.layer.mlp.down_proj_exl3, DType::kF16);
    std::vector<uint16_t> down_raw(rows*5120); down.Download(d, down_raw.data());
    std::vector<float> got(down_raw.size());
    for (size_t i = 0; i < down_raw.size(); ++i) got[i] = vt::F16ToF32(down_raw[i]);
    CapturedClose(phase+".down", got, real.oracle->Get(phase+"_hidden_out"), 0.002f, 1e-4f);
    CHECK(empty.Empty());
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real block Gemma: FP32 sum before FP16 residual rounding") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  // The independent producer IR computes norm(x.float()+res.float()) and
  // returns that sum narrowed separately as residual. These exact half inputs
  // distinguish that boundary from normalizing the narrowed residual, in both
  // the narrow leaf and the actual target's 5120-column work-group reduction.
  for (int width : {2, 5120}) for (int alias : {0, 1, 2}) {
    CAPTURE(width);
    CAPTURE(alias);
    xpu_test::Buffer input(gpu.q, DType::kF16, {1, width});
    xpu_test::Buffer residual(gpu.q, DType::kF16, {1, width});
    xpu_test::Buffer weight(gpu.q, DType::kF16, {width});
    xpu_test::Buffer output(gpu.q, DType::kF16, {1, width});
    std::vector<float> res(width), expected(width), stored(width);
    for (int i = 0; i < width; ++i) {
      res[i] = i % 2 ? 0.000244140625f : 0.00341796875f;
      expected[i] = i % 2 ? 0.99853515625f : 1.001953125f;
      stored[i] = i % 2 ? 1.0f : 1.00390625f;
    }
    input.put(std::vector<float>(width, 1)); residual.put(res);
    weight.put(std::vector<float>(width, 0));
    auto target = alias == 1 ? input.tensor : alias == 2 ? residual.tensor : output.tensor;
    vt::RmsNorm(gpu.q, target, input.tensor, weight.tensor,
                vt::RmsNormArgs{1e-6f, true}, &residual.tensor);
    std::vector<uint16_t> got(width);
    vt::GetBackend(gpu.q.device).Copy(gpu.q, got.data(), target.data, width*2);
    vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
    bool same = true;
    for (int i = 0; i < width; ++i) same &= got[i] == vt::F32ToF16(expected[i]);
    CHECK(same);
    // If output aliases residual, the output replaces the updated residual
    // after both old inputs have been consumed; otherwise both remain visible.
    CHECK(residual.floats() == (alias == 2 ? expected : stored));
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real block Gemma: captured P128 and D1 normalization stages") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* captures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  if (!model || !captures) {
    std::cerr << "SKIP: set VT_B70_EXL3_MODEL and VT_B70_EXL3_S1_FIXTURES.\n";
    std::exit(77);
  }
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const auto config = vllm::LoadHfConfig((std::filesystem::path(model) / "config.json").string());
  auto shard = vllm::SafetensorsFile::Open((std::filesystem::path(model) / "model-00001-of-00002.safetensors").string());
  const auto get = [&](const std::string& name) -> const vllm::StTensor& { return shard.Get(name); };
  const auto has = [&](const std::string& name) {
    return std::find(shard.Names().begin(), shard.Names().end(), name) != shard.Names().end();
  };
  const auto layer = vllm::LoadQwen3_5DenseLayer(get, has, "linear_attention", 0, "model.language_model.");
  auto oracle = vllm::SafetensorsFile::Open((std::filesystem::path(captures) / "real_block_oracle.safetensors").string());
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  REQUIRE(config.hidden_size == 5120);
  for (const std::string phase : {"p128", "d1"}) for (bool post : {false, true}) {
    CAPTURE(phase);
    CAPTURE(post);
    const int64_t rows = phase == "p128" ? 128 : 1;
    const auto& input = oracle.Get(phase + (post ? "_post_norm_input" : "_hidden_in"));
    const auto& expected = oracle.Get(phase + (post ? "_post_norm_output" : "_input_norm_output"));
    REQUIRE(input.dtype == "F16"); REQUIRE(expected.dtype == "F16");
    REQUIRE(input.shape == std::vector<int64_t>{rows, 5120});
    REQUIRE(expected.shape == input.shape);
    xpu_test::Buffer x(gpu.q, DType::kF16, {rows, 5120});
    xpu_test::Buffer out(gpu.q, DType::kF16, {rows, 5120});
    xpu_test::Buffer res(gpu.q, DType::kF16, {rows, 5120});
    x.upload(input.data);
    if (post) res.upload(oracle.Get(phase + "_post_norm_residual_input").data);
    const auto w = vllm::dense_attn::ResidentWeight(d, post ? layer.post_attention_layernorm : layer.input_layernorm);
    vt::RmsNorm(gpu.q, out.tensor, x.tensor, w,
                vt::RmsNormArgs{static_cast<float>(config.rms_norm_eps), true}, post ? &res.tensor : nullptr);
    const auto actual = out.floats();
    std::vector<float> want(actual.size());
    double error = 0, norm = 0; float max_error = 0; size_t different = 0;
    bool finite = true;
    const auto bytes = out.download();
    for (size_t i = 0; i < want.size(); ++i) {
      uint16_t bit; std::memcpy(&bit, expected.data + i*2, 2); want[i] = vt::F16ToF32(bit);
      finite &= std::isfinite(actual[i]);
      const double delta = double(actual[i]) - want[i]; error += delta*delta; norm += double(want[i])*want[i];
      max_error = std::max(max_error, std::abs(actual[i] - want[i]));
      different += std::memcmp(bytes.data() + i*2, expected.data + i*2, 2) != 0;
    }
    REQUIRE(finite);
    REQUIRE(norm > 0);
    std::cout << "REAL_BLOCK_NORM phase=" << phase << " post=" << post
              << " relative=" << std::sqrt(error/norm) << " max_error=" << max_error
              << " different_half_values=" << different << '\n';
    // Fixed independent producer IR fused_add_rms_norm tolerance.
    CHECK(std::sqrt(error/norm) < 2e-3);
    xpu_test::Close(actual, want, 2e-3f, 1e-2f);
    if (post) {
      const auto& expected_res = oracle.Get(phase + "_post_norm_residual_output");
      REQUIRE(expected_res.shape == input.shape);
      xpu_test::SameBytes(res.download(), std::vector<unsigned char>(expected_res.data, expected_res.data + expected_res.nbytes));
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

static void RunRealEagerTarget(int decode_steps) {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* fixtures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  if (!model || !fixtures) std::exit(77);
  const std::filesystem::path model_dir(model), receipts(fixtures);
  REQUIRE((decode_steps == 1 || decode_steps == 64));
  const auto capture_dir = receipts / (decode_steps == 1 ? "target-repeats" : "target-d64");
  const int repeats = decode_steps == 1 ? 3 : 1;
  std::ifstream record(capture_dir / "repeat-0.json");
  const auto captured_ids = nlohmann::json::parse(record).at("output_ids").at(0)
                                .get<std::vector<int32_t>>();
  REQUIRE(captured_ids.size() == static_cast<size_t>(decode_steps + 1));
  std::vector<vllm::SafetensorsFile> oracles;
  for (int repeat = 0; repeat < repeats; ++repeat)
    oracles.push_back(vllm::SafetensorsFile::Open(
        (capture_dir / ("repeat-" + std::to_string(repeat) + ".safetensors")).string()));
  const auto config = vllm::LoadHfConfig((model_dir / "config.json").string());
  REQUIRE(config.num_hidden_layers == 64);
  REQUIRE(config.vocab_size == 248320);
  std::vector<vllm::SafetensorsFile> shards;
  for (const char* name : {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"})
    shards.push_back(vllm::SafetensorsFile::Open((model_dir / name).string()));
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  auto weights = vllm::LoadQwen3_5Dense(shards, config, &gpu.q);
  REQUIRE(weights.exl3_checkpoint);
  REQUIRE(weights.precision.activation == DType::kF16);
  REQUIRE(weights.layers.size() == 64);
  REQUIRE(weights.lm_head_exl3.Bits() == 6);
  REQUIRE(weights.lm_head_exl3.OutFeatures() == 248320);
  REQUIRE(weights.lm_head.Empty());
  std::cout << "REAL_TARGET_LOAD layers=64 head_bits=6 vocab=248320\n" << std::flush;
  constexpr int slots = 5, active = 4, page = 1600;
  std::vector<std::unique_ptr<xpu_test::Buffer>> owners;
  std::vector<vllm::GdnStateCache> states;
  std::vector<vllm::PagedKvCache> caches;
  for (const auto& layer : weights.layers) {
    if (layer.is_linear_attention) {
      auto conv = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kF16,
          std::initializer_list<int64_t>{slots, 10240, 3});
      auto ssm = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kF32,
          std::initializer_list<int64_t>{slots, 48, 128, 128});
      conv->put(std::vector<float>(slots * 10240 * 3, 0.125f));
      ssm->put(std::vector<float>(slots * 48 * 128 * 128, 0.125f));
      vllm::GdnStateCache state; state.conv_state = conv->tensor; state.ssm_state = ssm->tensor;
      states.push_back(state);
      owners.push_back(std::move(conv)); owners.push_back(std::move(ssm));
    } else {
      auto kv = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kI8,
          std::initializer_list<int64_t>{2 * page * 4 * 256});
      std::vector<uint8_t> poison(2 * page * 4 * 256, 0x7f);
      kv->upload(poison.data());
      vllm::PagedKvCache cache;
      cache.data = kv->tensor.data; cache.dtype = DType::kI8;
      cache.num_blocks = 1; cache.block_size = page; cache.num_kv_heads = 4; cache.head_size = 256;
      cache.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
      caches.push_back(cache); owners.push_back(std::move(kv));
    }
  }
  REQUIRE(states.size() == 48); REQUIRE(caches.size() == 16);
  for (int step = 0; step <= decode_steps; ++step) {
    const bool prefill = step == 0;
    const int rows = prefill ? 128 : 1;
    const std::string phase = prefill ? "p128" : "d" + std::to_string(step);
    std::vector<int32_t> ids(rows), positions(rows);
    for (int i = 0; i < rows; ++i) {
      ids[i] = prefill ? 1000 + (i * 37) % 4096 : captured_ids[step - 1];
      positions[i] = prefill ? i : 127 + step;
    }
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1; am.num_actual_tokens = rows;
    am.query_start_loc = am.query_start_loc_cpu = {0, rows};
    am.seq_lens = am.seq_lens_cpu = {128 + step};
    am.max_query_len = rows; am.max_seq_len = am.seq_lens[0];
    am.block_table_num_cols = 1; am.block_table_tensor = {0};
    am.slot_mapping.assign(positions.begin(), positions.end());
    am.causal = true;
    vllm::v1::GDNAttentionMetadata gm;
    gm.num_actual_tokens = rows;
    gm.non_spec_state_indices_tensor = std::vector<int32_t>{active};
    gm.non_spec_query_start_loc = std::vector<int32_t>{0, rows};
    if (prefill) {
      gm.num_prefills = 1; gm.num_prefill_tokens = rows;
      gm.has_initial_state = std::vector<uint8_t>{0};
      gm.prefill_query_start_loc = std::vector<int32_t>{0, rows};
      gm.prefill_state_indices = std::vector<int32_t>{active};
      gm.prefill_has_initial_state = std::vector<uint8_t>{0};
      const auto chunks = vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = chunks.batch_ptr; gm.token_chunk_offset_ptr = chunks.token_chunk_offset_ptr;
    } else { gm.num_decodes = gm.num_decode_tokens = 1; }
    std::cout << "REAL_TARGET_FORWARD_START " << phase << '\n' << std::flush;
    const auto out = vllm::Qwen3_5DenseModel::ForwardDeviceTap(
        ids, positions, am, gm, caches, states, weights, config, gpu.q, nullptr, {rows - 1});
    REQUIRE(out.rows == 1); REQUIRE(out.device_tensor.dtype == DType::kF32);
    std::vector<float> logits(248320);
    auto& backend = vt::GetBackend(gpu.q.device.type);
    backend.Copy(gpu.q, logits.data(), out.device_tensor.data, logits.size() * sizeof(float));
    backend.Synchronize(gpu.q);
    REQUIRE(std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }));
    const char* output_prefix = std::getenv("VT_B70_EXL3_TARGET_OUTPUT_PREFIX");
    const std::string prefix = output_prefix ? output_prefix :
        (decode_steps == 1 ? "target-native" : "target-native-d64");
    const auto file = receipts / (prefix + "-" + phase + "-logits.f32");
    REQUIRE(!std::filesystem::exists(file));
    std::ofstream raw(file, std::ios::binary);
    raw.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
    raw.close(); REQUIRE(raw.good());
    const auto probabilities = [](const std::vector<float>& x) {
      const double maximum = *std::max_element(x.begin(), x.end());
      std::vector<double> p(x.size()); double sum = 0;
      for (size_t i = 0; i < x.size(); ++i) sum += p[i] = std::exp(double(x[i]) - maximum);
      for (auto& v : p) v /= sum;
      return p;
    };
    const auto top10 = [](const std::vector<float>& x) {
      std::vector<size_t> indices(x.size()); std::iota(indices.begin(), indices.end(), 0);
      std::partial_sort(indices.begin(), indices.begin() + 10, indices.end(),
                        [&](size_t a, size_t b) { return x[a] > x[b]; });
      indices.resize(10); return indices;
    };
    const auto actual_p = probabilities(logits);
    const auto actual_top = top10(logits);
    for (int repeat = 0; repeat < repeats; ++repeat) {
      const auto expected = CapturedFloats(oracles[repeat].Get(phase + "_logits"));
      REQUIRE(expected.size() == logits.size());
      const auto expected_p = probabilities(expected);
      const auto expected_top = top10(expected);
      double tv = 0, kl = 0;
      for (size_t i = 0; i < logits.size(); ++i) {
        tv += std::abs(actual_p[i] - expected_p[i]) * 0.5;
        if (expected_p[i] > 0) kl += expected_p[i] * std::log(expected_p[i] / actual_p[i]);
      }
      int overlap = 0;
      for (const auto id : actual_top)
        overlap += std::find(expected_top.begin(), expected_top.end(), id) != expected_top.end();
      std::cout << "REAL_TARGET_COMPARE " << phase << " repeat=" << repeat << " TV=" << tv
                << " KL=" << kl << " top10=" << overlap << " greedy=" << actual_top[0]
                << " reference_greedy=" << captured_ids[step] << '\n';
      CHECK(tv <= 0.02); CHECK(kl <= 0.002); CHECK(overlap >= 9);
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real target: eager P128 D1 full vocabulary comparison") {
  RunRealEagerTarget(1);
}

TEST_CASE("XPU EXL3 real target: eager P128 D64 full vocabulary continuation") {
  RunRealEagerTarget(64);
}

TEST_CASE("XPU dense EXL3 FP16: QKVZ/QKV groups preserve independent source projections") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const auto c = Config();
  auto w = Weights(c, false);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  for (bool qkv : {false, true}) {
    CAPTURE(qkv);
    auto& g = w.layers[0].gdn;
    auto& a = w.layers[1].attn;
    const std::vector<const vllm::Exl3Weight*> sources = qkv
        ? std::vector<const vllm::Exl3Weight*>{&a.q_proj_exl3, &a.k_proj_exl3, &a.v_proj_exl3}
        : std::vector<const vllm::Exl3Weight*>{&g.in_proj_qkv_exl3, &g.in_proj_z_exl3};
    auto& cache = qkv ? a.qkv_proj_exl3 : g.in_proj_qkvz_exl3;
    const void* resident = nullptr;
    for (int64_t m : {1, 4}) {
      CAPTURE(m);
      const auto values = xpu_test::Values(m * 128, 13, 0.007f);
      std::vector<uint16_t> input_bits(values.size());
      for (size_t i = 0; i < values.size(); ++i) input_bits[i] = vt::F32ToF16(values[i]);
      vllm::dense_attn::DBuf input(d, DType::kF16, {m, 128}, input_bits.data());
      auto merged = vllm::dense_exl3::GroupedLinear(d, input.t(), sources, cache);
      REQUIRE(merged.t().shape[1] == 512);
      REQUIRE(cache.suh.shape[0] == static_cast<int64_t>(sources.size()));
      CHECK(cache.trellis.host_released);
      CHECK(cache.source_map.host_released);
      if (resident) CHECK(cache.trellis.d_dev.get() == resident);
      else for (auto* source : sources) CHECK(source->trellis.d_dev == nullptr);
      resident = cache.trellis.d_dev.get();
      // Materialize logical views with the same native Copy used by the
      // unfused attention consumer, and compare every row/column independently.
      int64_t offset = 0;
      for (auto* source : sources) {
        const int64_t width = source->OutFeatures();
        auto view = merged.t().Slice(1, offset, offset + width);
        REQUIRE(view.stride[0] == 512);
        vllm::dense_attn::DBuf copy(d, DType::kF16, {m, width});
        vt::Copy(d.q, copy.t(), view);
        vllm::OwnedTensor none;
        auto separate = vllm::dense_exl3::Linear(d, input.t(), none, *source, DType::kF16);
        std::vector<uint16_t> got(m * width), expected(m * width);
        copy.Download(d, got.data());
        separate.Download(d, expected.data());
        CHECK(got == expected);
        CHECK(std::any_of(got.begin(), got.end(), [](uint16_t v) { return vt::F16ToF32(v) != 0; }));
        offset += width;
      }
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU dense EXL3 FP16: paged hybrid prefill and decode preserve model/head precision") {
  vt::EnableOpProviderCallStats(true);
  const auto grouped_before = vt::GetOpProviderStats(vt::OpId::kExl3GroupedLinear, vt::DeviceType::kXPU).selections;
  const auto packed_before = vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kXPU).selections;
  bool split_ba = false;
  SUBCASE("merged BA owner") {}
  SUBCASE("split BA owners") { split_ba = true; }
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const auto c = Config();
  const auto w = Weights(c, split_ba);
  xpu_test::Buffer kv(gpu.q, DType::kF16, {2 * 2 * 8 * 128});
  xpu_test::Buffer conv(gpu.q, DType::kF16, {2, 384, 3});
  xpu_test::Buffer ssm(gpu.q, DType::kF32, {2, 1, 128, 128});
  kv.put(std::vector<float>(2 * 2 * 8 * 128, 0));
  conv.put(std::vector<float>(2 * 384 * 3, 0));
  ssm.put(std::vector<float>(2 * 128 * 128, 0));
  vllm::PagedKvCache cache;
  cache.data = kv.tensor.data;
  cache.dtype = DType::kF16;
  cache.num_blocks = 2;
  cache.block_size = 8;
  cache.num_kv_heads = 1;
  cache.head_size = 128;
  std::vector<vllm::GdnStateCache> states(1);
  states[0].conv_state = conv.tensor;
  states[0].ssm_state = ssm.tensor;
  const std::vector<vllm::PagedKvCache> caches{cache};
  for (int step : {0, 1}) {
    CAPTURE(step);
    const std::vector<int32_t> ids = step == 0 ? std::vector<int32_t>{3, 7, 2, 5}
                                             : std::vector<int32_t>{9};
    const std::vector<int32_t> positions = step == 0 ? std::vector<int32_t>{0, 1, 2, 3}
                                                   : std::vector<int32_t>{4};
    const int32_t rows = static_cast<int32_t>(ids.size());
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1;
    am.num_actual_tokens = rows;
    am.query_start_loc = am.query_start_loc_cpu = {0, rows};
    am.seq_lens = am.seq_lens_cpu = {step == 0 ? 4 : 5};
    am.max_query_len = rows;
    am.max_seq_len = am.seq_lens[0];
    am.block_table_num_cols = 1;
    am.block_table_tensor = {1};
    for (int32_t p : positions) am.slot_mapping.push_back(8 + p);
    am.causal = true;
    vllm::v1::GDNAttentionMetadata gm;
    gm.num_actual_tokens = rows;
    gm.non_spec_state_indices_tensor = std::vector<int32_t>{1};
    gm.non_spec_query_start_loc = std::vector<int32_t>{0, rows};
    if (step == 0) {
      gm.num_prefills = 1;
      gm.num_prefill_tokens = rows;
      gm.has_initial_state = std::vector<uint8_t>{0};
      gm.prefill_query_start_loc = std::vector<int32_t>{0, rows};
      gm.prefill_state_indices = std::vector<int32_t>{1};
      gm.prefill_has_initial_state = std::vector<uint8_t>{0};
      const auto metadata = vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = metadata.batch_ptr;
      gm.token_chunk_offset_ptr = metadata.token_chunk_offset_ptr;
    } else {
      gm.num_decodes = gm.num_decode_tokens = 1;
    }
    vllm::Qwen3_5MTPHiddenStates tap;
    const auto logits = vllm::Qwen3_5DenseModel::ForwardDeviceTap(
        ids, positions, am, gm, caches, states, w, c, gpu.q, &tap, {rows - 1});
    REQUIRE(logits.device_tensor.dtype == DType::kF32);
    REQUIRE(logits.rows == 1);
    REQUIRE(tap.tensor.dtype == DType::kF16);
    REQUIRE(tap.tensor.shape[0] == rows);
    vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
    auto head = vllm::dense_exl3::Linear(d, tap.tensor.Slice(0, rows - 1, rows),
                                       w.lm_head, w.lm_head_exl3, DType::kF16);
    std::vector<uint16_t> expected(128);
    head.Download(d, expected.data());
    std::vector<float> got(128);
    auto& backend = vt::GetBackend(gpu.q.device.type);
    backend.Copy(gpu.q, got.data(), logits.device_tensor.data, got.size() * 4);
    backend.Synchronize(gpu.q);
    bool nonzero = false;
    for (size_t i = 0; i < got.size(); ++i) {
      CHECK(std::isfinite(got[i]));
      CHECK(got[i] == vt::F16ToF32(expected[i]));
      nonzero |= got[i] != 0;
    }
    CHECK(nonzero);
    if (step == 0) {
      // Pooling replays the same fresh prefill and gathers reordered hidden
      // rows before widening. The tap is an independently owned output.
      const auto pooled = vllm::Qwen3_5DenseModel::ForwardHidden(
          ids, positions, am, gm, caches, states, w, c, gpu.q, {3, 0});
      REQUIRE(pooled.rows == 2);
      REQUIRE(pooled.host.size() == 256);
      std::vector<uint16_t> hidden(4 * 128);
      backend.Copy(gpu.q, hidden.data(), tap.tensor.data, hidden.size() * 2);
      backend.Synchronize(gpu.q);
      for (int i = 0; i < 128; ++i) {
        CHECK(pooled.host[i] == vt::F16ToF32(hidden[3 * 128 + i]));
        CHECK(pooled.host[128 + i] == vt::F16ToF32(hidden[i]));
      }
    }
    const auto recurrent = ssm.floats();
    CHECK(std::all_of(recurrent.begin(), recurrent.begin() + 128 * 128,
                      [](float v) { return v == 0; }));
    CHECK(std::any_of(recurrent.begin() + 128 * 128, recurrent.end(),
                      [](float v) { return v != 0; }));
    auto invalid = states;
    invalid[0].conv_state.dtype = DType::kBF16;
    CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseModel::ForwardDevice(
        ids, positions, am, gm, caches, invalid, w, c, gpu.q, {rows - 1}),
        doctest::Contains("GDN convolution state must be FP16"), std::runtime_error);
  }
  CHECK(vt::GetOpProviderStats(vt::OpId::kExl3GroupedLinear, vt::DeviceType::kXPU).selections > grouped_before);
  CHECK(vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kXPU).selections == packed_before);
  REQUIRE(w.layers[0].gdn.in_proj_qkvz_exl3.suh.shape[0] == 2);
  REQUIRE(w.layers[1].attn.qkv_proj_exl3.suh.shape[0] == 3);
  CHECK(w.layers[0].gdn.in_proj_qkv_exl3.trellis.d_dev == nullptr);
  CHECK(w.layers[0].gdn.in_proj_z_exl3.trellis.d_dev == nullptr);
  CHECK(w.layers[1].attn.q_proj_exl3.trellis.d_dev == nullptr);
  CHECK(w.layers[1].attn.k_proj_exl3.trellis.d_dev == nullptr);
  CHECK(w.layers[1].attn.v_proj_exl3.trellis.d_dev == nullptr);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU dense EXL3 FP16: unpaged model entries preserve hidden and head rounding") {
  vt::EnableOpProviderCallStats(true);
  const auto grouped_before = vt::GetOpProviderStats(vt::OpId::kExl3GroupedLinear, vt::DeviceType::kXPU).selections;
  const auto packed_before = vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kXPU).selections;
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const auto c = Config();
  const auto w = Weights(c, false);
  const std::vector<int32_t> ids{3, 7, 2, 5}, positions{0, 1, 2, 3};
  const auto full = vllm::Qwen3_5DenseModel::ForwardDense(ids, positions, w, c, gpu.q);
  const auto last = vllm::Qwen3_5DenseModel::ForwardDenseLastLogits(ids, positions, w, c, gpu.q);
  const auto hidden = vllm::Qwen3_5DenseModel::ForwardDenseHidden(ids, positions, w, c, gpu.q);
  REQUIRE(full.size() == 4 * 128);
  REQUIRE(last.size() == 128);
  REQUIRE(hidden.size() == 4 * 128);
  // Producer M1 GEMV has half tile-local accumulation; the M4 DPAS head
  // accumulates in F32. They need numerical agreement, not bit identity.
  double difference = 0, magnitude = 0;
  for (size_t i = 0; i < last.size(); ++i) {
    const double delta = double(last[i]) - full[3 * 128 + i];
    difference += delta * delta;
    magnitude += double(full[3 * 128 + i]) * full[3 * 128 + i];
  }
  REQUIRE(magnitude > 0);
  const double relative = std::sqrt(difference / magnitude);
  std::cout << "MODEL_SMALLM_HEAD M1_vs_M4_relative=" << relative << '\n';
  CHECK(std::isfinite(relative));
  CHECK(relative < 2e-3);
  for (const auto* values : {&full, &hidden}) {
    bool nonzero = false;
    for (float value : *values) {
      CHECK(std::isfinite(value));
      CHECK(value == vt::F16ToF32(vt::F32ToF16(value)));
      nonzero |= value != 0;
    }
    CHECK(nonzero);
  }
  // The hidden API stops before the head. Independently invoke the F16 head
  // on its selected row to check that the logits API does not omit rounding.
  std::vector<uint16_t> row(128);
  for (size_t i = 0; i < row.size(); ++i) row[i] = vt::F32ToF16(hidden[3 * 128 + i]);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  vllm::dense_attn::DBuf input(d, DType::kF16, {1, 128}, row.data());
  auto head = vllm::dense_exl3::Linear(d, input.t(), w.lm_head, w.lm_head_exl3, DType::kF16);
  std::vector<uint16_t> expected(128);
  head.Download(d, expected.data());
  for (size_t i = 0; i < row.size(); ++i) CHECK(last[i] == vt::F16ToF32(expected[i]));
  // Independently check every full-logit row against the producer M4 head,
  // so changing the selected-row comparison cannot hide an indexing error.
  std::vector<uint16_t> all_hidden(hidden.size()), all_expected(full.size());
  for (size_t i = 0; i < hidden.size(); ++i) all_hidden[i] = vt::F32ToF16(hidden[i]);
  vllm::dense_attn::DBuf all_input(d, DType::kF16, {4, 128}, all_hidden.data());
  auto all_head = vllm::dense_exl3::Linear(d, all_input.t(), w.lm_head,
                                         w.lm_head_exl3, DType::kF16);
  all_head.Download(d, all_expected.data());
  for (size_t i = 0; i < full.size(); ++i) CHECK(full[i] == vt::F16ToF32(all_expected[i]));
  CHECK(vt::GetOpProviderStats(vt::OpId::kExl3GroupedLinear, vt::DeviceType::kXPU).selections > grouped_before);
  CHECK(vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kXPU).selections == packed_before);
  CHECK(w.layers[0].gdn.in_proj_qkvz_exl3.trellis.host_released);
  CHECK(w.layers[1].attn.qkv_proj_exl3.trellis.host_released);
  CHECK(vt::GetReferenceTierHits() == 0);
}
