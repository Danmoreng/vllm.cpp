#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gptq4_weight.h"
#include "vllm/model_executor/models/dense_gptq4_linear.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/unaligned.h"
#include <nlohmann/json.hpp>

namespace {

struct Fixture {
  std::map<std::string, std::vector<uint8_t>> storage;
  std::map<std::string, vllm::StTensor> tensors;

  template <typename T>
  void Add(const std::string& name, const std::string& dtype,
           std::vector<int64_t> shape, const std::vector<T>& values) {
    auto& bytes = storage[name];
    bytes.resize(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    auto& tensor = tensors[name];
    tensor.dtype = dtype;
    tensor.shape = std::move(shape);
    tensor.data = bytes.data();
    tensor.nbytes = bytes.size();
  }

  void AddProjection(const std::string& name, int n, int seed, int k = 256) {
    std::vector<uint32_t> words(k / 8 * n);
    for (int p = 0; p < k / 8; ++p)
      for (int output = 0; output < n; ++output)
        words[p * n + output] = static_cast<uint32_t>(seed + p * 100 + output);
    std::vector<uint16_t> scales(k / 128 * n);
    for (int group = 0; group < k / 128; ++group)
      for (int output = 0; output < n; ++output)
        scales[group * n + output] =
            vt::F32ToF16(static_cast<float>(seed + group * 10 + output + 1));
    std::vector<uint32_t> qzeros(k / 128 * n / 8, 0x77777777u);
    std::vector<int32_t> g_idx(k);
    for (int i = 0; i < k; ++i) g_idx[i] = i / 128;
    Add(name + ".qweight", "I32", {k / 8, n}, words);
    Add(name + ".scales", "F16", {k / 128, n}, scales);
    Add(name + ".qzeros", "I32", {k / 128, n / 8}, qzeros);
    Add(name + ".g_idx", "I32", {k}, g_idx);
  }

  vllm::TensorResolver Resolver() const {
    return [this](const std::string& name) -> const vllm::StTensor& {
      return tensors.at(name);
    };
  }

  std::string WriteSafetensors() const {
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch().count();
    const std::string path =
        (std::filesystem::temp_directory_path() /
         ("test_gptq4_weight_" + std::to_string(nonce) + ".safetensors"))
            .string();
    nlohmann::json header = nlohmann::json::object();
    size_t offset = 0;
    for (const auto& [name, tensor] : tensors) {
      header[name] = {{"dtype", tensor.dtype},
                      {"shape", tensor.shape},
                      {"data_offsets", {offset, offset + tensor.nbytes}}};
      offset += tensor.nbytes;
    }
    std::string encoded = header.dump();
    encoded.append((8 - encoded.size() % 8) % 8, ' ');
    std::ofstream file(path, std::ios::binary);
    const uint64_t header_size = encoded.size();
    file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    file.write(encoded.data(), encoded.size());
    for (const auto& [name, bytes] : storage)
      file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    file.close();
    return path;
  }
};

uint32_t Word(const vllm::OwnedTensor& tensor, size_t index) {
  return vt::LoadUnaligned<uint32_t>(tensor.bytes.data() + index * 4);
}

uint16_t Scale(const vllm::OwnedTensor& tensor, size_t index) {
  return vt::LoadUnaligned<uint16_t>(tensor.bytes.data() + index * 2);
}

void CheckAgainstPostLoad(const vllm::Gptq4Weight& weight,
                          const std::string& path) {
  const auto oracle = vllm::SafetensorsFile::Open(path);
  const auto& packed = oracle.Get("qweight_nt_int32");
  const auto& scales = oracle.Get("scales_f16");
  const auto& zero = oracle.Get("effective_zero_point_i8");
  REQUIRE(packed.nbytes == weight.qweight.bytes.size());
  REQUIRE(scales.nbytes == weight.scales.bytes.size());
  REQUIRE(zero.nbytes == weight.zero_point.bytes.size());
  CHECK(std::memcmp(packed.data, weight.qweight.bytes.data(), packed.nbytes) == 0);
  CHECK(std::memcmp(scales.data, weight.scales.bytes.data(), scales.nbytes) == 0);
  CHECK(std::memcmp(zero.data, weight.zero_point.bytes.data(), zero.nbytes) == 0);
}

vllm::OwnedTensor ConstantHalf(std::vector<int64_t> shape, float value,
                               bool nk = false) {
  auto result = vllm::dense_loaders::MakeOwned(vt::DType::kF16, shape);
  result.nk = nk;
  const uint16_t bits = vt::F32ToF16(value);
  for (size_t i = 0; i < result.bytes.size(); i += 2)
    std::memcpy(result.bytes.data() + i, &bits, 2);
  return result;
}

vllm::Gptq4Weight ConstantPacked(int64_t k, int64_t n,
                                  float scale = 0.00390625f) {
  vllm::Gptq4Weight result;
  result.k = k;
  result.n = n;
  result.qweight = vllm::dense_loaders::MakeOwned(vt::DType::kI32,
                                                  {n, k / 8});
  result.scales = vllm::dense_loaders::MakeOwned(vt::DType::kF16,
                                                 {k / 128, n});
  result.zero_point = vllm::dense_loaders::MakeOwned(vt::DType::kI8, {1});
  const uint32_t words = 0x99999999u;
  const uint16_t scale_bits = vt::F32ToF16(scale);
  for (size_t i = 0; i < result.qweight.bytes.size(); i += 4)
    std::memcpy(result.qweight.bytes.data() + i, &words, 4);
  for (size_t i = 0; i < result.scales.bytes.size(); i += 2)
    std::memcpy(result.scales.bytes.data() + i, &scale_bits, 2);
  result.zero_point.bytes.data()[0] = 8;
  return result;
}

struct XpuQueueOwner {
  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  ~XpuQueueOwner() { vt::DestroyQueue(queue); }
};

struct XpuCacheAllocations {
  vt::Queue& queue;
  std::vector<void*> blocks;
  explicit XpuCacheAllocations(vt::Queue& q) : queue(q) {}
  void* Zero(size_t bytes) {
    void* data = vt::Alloc(queue.device, bytes);
    vt::GetBackend(queue.device).Memset(queue, data, 0, bytes);
    blocks.push_back(data);
    return data;
  }
  ~XpuCacheAllocations() {
    vt::GetBackend(queue.device).Synchronize(queue);
    for (void* data : blocks) vt::Free(queue.device, data);
  }
};

}  // namespace

TEST_CASE("GPTQ4 loader transposes packed words and merges scales by group") {
  Fixture fixture;
  fixture.AddProjection("gate", 8, 1);
  fixture.AddProjection("up", 16, 2);
  const auto weight = vllm::LoadMergedGptq4Weight(
      fixture.Resolver(), {"gate", "up"}, 256, {8, 16});
  CHECK(weight.k == 256);
  CHECK(weight.n == 24);
  CHECK(weight.group_size == 128);
  CHECK(weight.packing_version == vllm::Gptq4Weight::kPackingVersion);
  CHECK(weight.disk_zero_offset == 1);
  CHECK(weight.qweight.dtype == vt::DType::kI32);
  CHECK(weight.qweight.shape[0] == 24);
  CHECK(weight.qweight.shape[1] == 32);
  CHECK(weight.scales.shape[0] == 2);
  CHECK(weight.scales.shape[1] == 24);
  CHECK(weight.zero_point.bytes.size() == 1);
  CHECK(weight.zero_point.bytes.data()[0] == 8);
  for (int p = 0; p < 32; ++p) {
    for (int output = 0; output < 24; ++output) {
      const int source_output = output < 8 ? output : output - 8;
      const int seed = output < 8 ? 1 : 2;
      CHECK(Word(weight.qweight, output * 32 + p) ==
            static_cast<uint32_t>(seed + p * 100 + source_output));
    }
  }
  for (int group = 0; group < 2; ++group) {
    for (int output = 0; output < 24; ++output) {
      const int source_output = output < 8 ? output : output - 8;
      const int seed = output < 8 ? 1 : 2;
      CHECK(Scale(weight.scales, group * 24 + output) ==
            vt::F32ToF16(
                static_cast<float>(seed + group * 10 + source_output + 1)));
    }
  }
}

TEST_CASE("GPTQ4 loader rejects changed zero, ordering and geometry") {
  Fixture fixture;
  fixture.AddProjection("q", 8, 1);
  CHECK_NOTHROW(vllm::LoadGptq4Weight(fixture.Resolver(), "q", 256, 8));
  CHECK_THROWS_AS(vllm::LoadGptq4Weight(fixture.Resolver(), "q", 256, 16),
                  std::runtime_error);
  fixture.storage["q.qzeros"][0] = 0;
  CHECK_THROWS_AS(vllm::LoadGptq4Weight(fixture.Resolver(), "q", 256, 8),
                  std::runtime_error);
  fixture.storage["q.qzeros"][0] = 0x77;
  fixture.storage["q.g_idx"][0] = 1;
  CHECK_THROWS_AS(vllm::LoadGptq4Weight(fixture.Resolver(), "q", 256, 8),
                  std::runtime_error);
}

TEST_CASE("GPTQ4 resident uploads once and releases the host mirror") {
  Fixture fixture;
  fixture.AddProjection("q", 8, 1);
  auto weight = vllm::LoadGptq4Weight(fixture.Resolver(), "q", 256, 8);
  REQUIRE(weight.ResidentBytes() == 0);
  const auto before = vllm::load_stats::Snapshot().device_upload_bytes;
  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kCPU, 0});
  const auto first = vllm::PrepareGptq4Resident(weight, queue, false);
  CHECK(weight.ResidentBytes() == 1057);
  CHECK(weight.qweight.HasHostBytes());
  CHECK(std::memcmp(first.qweight.data, weight.qweight.bytes.data(),
                    weight.qweight.bytes.size()) == 0);
  CHECK(std::memcmp(first.scales.data, weight.scales.bytes.data(),
                    weight.scales.bytes.size()) == 0);
  CHECK(vllm::load_stats::Snapshot().device_upload_bytes - before == 1057);
  const auto second = vllm::PrepareGptq4Resident(weight, queue, true);
  CHECK(second.qweight.data == first.qweight.data);
  CHECK(second.scales.data == first.scales.data);
  CHECK(second.zero_point.data == first.zero_point.data);
  CHECK(!weight.qweight.HasHostBytes());
  CHECK(!weight.scales.HasHostBytes());
  CHECK(!weight.zero_point.HasHostBytes());
  CHECK(!weight.qweight.Empty());
  CHECK(vllm::load_stats::Snapshot().device_upload_bytes - before == 1057);
  vt::DestroyQueue(queue);
}

TEST_CASE("GPTQ4 model host-release visitor includes packed residents") {
  Fixture fixture;
  fixture.AddProjection("q", 8, 1);
  vllm::Qwen3_5DenseWeights model;
  model.layers.resize(1);
  auto& weight = model.layers[0].gptq4.mlp_down;
  weight = vllm::LoadGptq4Weight(fixture.Resolver(), "q", 256, 8);
  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kCPU, 0});
  (void)vllm::PrepareGptq4Resident(weight, queue, false);
  REQUIRE(weight.qweight.HasHostBytes());
  CHECK(vllm::ReleaseResidentQwen3_5DenseHostWeights(model) == 1057);
  CHECK(!weight.qweight.HasHostBytes());
  CHECK(!weight.scales.HasHostBytes());
  CHECK(!weight.zero_point.HasHostBytes());
  CHECK(weight.ResidentBytes() == 1057);
  CHECK(vllm::ReleaseResidentQwen3_5DenseHostWeights(model) == 0);
  vt::DestroyQueue(queue);
}

TEST_CASE("GPTQ4 text inventory loads only the declared packed projections") {
  Fixture fixture;
  const std::string base = "model.language_model.layers.0.";
  fixture.AddProjection(base + "mlp.gate_proj", 128, 1, 128);
  fixture.AddProjection(base + "mlp.up_proj", 128, 2, 128);
  fixture.AddProjection(base + "mlp.down_proj", 128, 3, 128);
  fixture.AddProjection(base + "linear_attn.in_proj_qkv", 384, 4, 128);
  fixture.AddProjection(base + "linear_attn.in_proj_z", 128, 5, 128);
  fixture.AddProjection(base + "linear_attn.out_proj", 128, 6, 128);
  const std::vector<uint16_t> one(128, vt::F32ToF16(1.0f));
  fixture.Add("model.language_model.embed_tokens.weight", "F16", {16, 128},
              std::vector<uint16_t>(16 * 128, vt::F32ToF16(0.5f)));
  fixture.Add("model.language_model.norm.weight", "F16", {128}, one);
  fixture.Add("lm_head.weight", "F16", {16, 128},
              std::vector<uint16_t>(16 * 128, vt::F32ToF16(0.25f)));
  fixture.Add(base + "input_layernorm.weight", "F16", {128}, one);
  fixture.Add(base + "post_attention_layernorm.weight", "F16", {128}, one);
  fixture.Add(base + "linear_attn.in_proj_b.weight", "F16", {1, 128}, one);
  fixture.Add(base + "linear_attn.in_proj_a.weight", "F16", {1, 128},
              std::vector<uint16_t>(128, vt::F32ToF16(2.0f)));
  fixture.Add(base + "linear_attn.conv1d.weight", "F16", {384, 1, 4},
              std::vector<uint16_t>(384 * 4, vt::F32ToF16(0.5f)));
  fixture.Add(base + "linear_attn.A_log", "F16", {1},
              std::vector<uint16_t>{vt::F32ToF16(2.0f)});
  fixture.Add(base + "linear_attn.dt_bias", "F16", {1},
              std::vector<uint16_t>{vt::F32ToF16(3.0f)});
  fixture.Add(base + "linear_attn.norm.weight", "F16", {128}, one);
  const std::string path = fixture.WriteSafetensors();
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(path));
  vllm::HfConfig config;
  config.hidden_size = 128;
  config.vocab_size = 16;
  config.intermediate_size = 128;
  config.num_hidden_layers = 1;
  config.num_attention_heads = 1;
  config.num_key_value_heads = 1;
  config.head_dim = 128;
  config.linear_num_key_heads = 1;
  config.linear_key_head_dim = 128;
  config.linear_num_value_heads = 1;
  config.linear_value_head_dim = 128;
  config.linear_conv_kernel_dim = 4;
  config.torch_dtype = "float16";
  config.mamba_ssm_dtype = "float32";
  config.layer_types = {"linear_attention"};
  config.raw = {{"quantization_config",
                 {{"quant_method", "gptq"}, {"format", "gptq"},
                  {"bits", 4}, {"group_size", 128}, {"sym", true},
                  {"desc_act", false}, {"pack_dtype", "int32"}}}};
  const auto layers =
      vllm::LoadQwen3_5DenseGptq4TextProjections(shards, config);
  REQUIRE(layers.size() == 1);
  CHECK(layers[0].gdn_qkvz.n == 512);
  CHECK(layers[0].gdn_out.k == 128);
  CHECK(layers[0].mlp_gate_up.n == 256);
  CHECK(layers[0].attn_qkv.k == 0);
  CHECK(layers[0].ResidentBytes() == 0);
  const auto full = vllm::LoadQwen3_5Dense(shards, config);
  REQUIRE(full.gptq4_checkpoint);
  CHECK(full.precision.activation == vt::DType::kF16);
  CHECK(full.precision.dense_weight == vt::DType::kF16);
  CHECK(full.precision.kv_auto == vt::DType::kF16);
  CHECK(full.precision.gdn_conv_state == vt::DType::kF16);
  CHECK(full.precision.gdn_recurrent_state == vt::DType::kF32);
  REQUIRE(full.layers.size() == 1);
  CHECK(full.layers[0].gptq4.gdn_qkvz.n == 512);
  CHECK(full.layers[0].gdn.in_proj_ba.dtype == vt::DType::kF16);
  CHECK(full.layers[0].gdn.in_proj_ba.nk);
  CHECK(full.layers[0].gdn.in_proj_ba.shape[0] == 2);
  CHECK(full.layers[0].gdn.in_proj_ba.shape[1] == 128);
  CHECK(Scale(full.layers[0].gdn.in_proj_ba, 0) == vt::F32ToF16(1.0f));
  CHECK(Scale(full.layers[0].gdn.in_proj_ba, 128) == vt::F32ToF16(2.0f));
  CHECK(full.layers[0].gdn.a_log.dtype == vt::DType::kF32);
  CHECK(full.lm_head.dtype == vt::DType::kF16);
  CHECK(full.lm_head.nk);
  CHECK(!full.embed_tokens.Empty());
  CHECK(full.layers[0].mlp.gate_up_proj.Empty());
  fixture.Add(base + "mlp.gate_proj.weight", "F16", {128, 128},
              std::vector<uint16_t>(128 * 128, vt::F32ToF16(1.0f)));
  const std::string conflict_path = fixture.WriteSafetensors();
  std::vector<vllm::SafetensorsFile> conflict_shards;
  conflict_shards.push_back(vllm::SafetensorsFile::Open(conflict_path));
  CHECK_THROWS_AS(vllm::LoadQwen3_5Dense(conflict_shards, config),
                  std::runtime_error);
  std::remove(conflict_path.c_str());
  config.raw["quantization_config"]["desc_act"] = true;
  CHECK_THROWS_AS(vllm::LoadQwen3_5DenseGptq4TextProjections(shards, config),
                  std::runtime_error);
  std::remove(path.c_str());
}

TEST_CASE("GPTQ4 precision policy preserves ordinary dense defaults") {
  vllm::HfConfig config;
  config.torch_dtype = "bfloat16";
  const auto ordinary = vllm::ResolveQwen3_5DensePrecision(config, false);
  CHECK(ordinary.activation == vt::DType::kBF16);
  CHECK(ordinary.dense_weight == vt::DType::kBF16);
  CHECK(ordinary.kv_auto == vt::DType::kBF16);
  CHECK(ordinary.sampler == vt::DType::kF32);
  CHECK_THROWS_AS(vllm::ResolveQwen3_5DensePrecision(config, true),
                  std::runtime_error);
  config.torch_dtype = "float16";
  config.mamba_ssm_dtype = "float32";
  const auto gptq = vllm::ResolveQwen3_5DensePrecision(config, true);
  CHECK(gptq.activation == vt::DType::kF16);
  CHECK(gptq.gdn_recurrent_state == vt::DType::kF32);
}

TEST_CASE("GPTQ4 cache spec stores FP16 KV and convolution with FP32 recurrence") {
  vllm::HfConfig config;
  config.torch_dtype = "float16";
  config.mamba_ssm_dtype = "float32";
  config.raw = {{"quantization_config", {{"quant_method", "gptq"}}}};
  config.num_key_value_heads = 4;
  config.head_dim = 256;
  config.linear_num_key_heads = 16;
  config.linear_num_value_heads = 48;
  config.linear_key_head_dim = 128;
  config.linear_value_head_dim = 128;
  config.linear_conv_kernel_dim = 4;
  auto model = vllm::MakeQwen3_5DenseLoadedModel(vllm::Qwen3_5DenseWeights{});
  const auto kv = vllm::ModelRegistry::MakeKVCache(*model, config, 16, 8);
  REQUIRE(kv.kv_cache_groups.size() == 2);
  const auto* attn = dynamic_cast<const vllm::v1::FullAttentionSpec*>(
      kv.kv_cache_groups[0].kv_cache_spec.get());
  const auto* mamba = dynamic_cast<const vllm::v1::MambaSpec*>(
      kv.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(attn != nullptr);
  REQUIRE(mamba != nullptr);
  CHECK(attn->dtype == vt::DType::kF16);
  REQUIRE(mamba->dtypes.size() == 2);
  CHECK(mamba->dtypes[0] == vt::DType::kF16);
  CHECK(mamba->dtypes[1] == vt::DType::kF32);
  config.raw = nlohmann::json::object();
  const auto ordinary = vllm::ModelRegistry::MakeKVCache(*model, config, 16, 8);
  const auto* ordinary_mamba = dynamic_cast<const vllm::v1::MambaSpec*>(
      ordinary.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(ordinary_mamba != nullptr);
  CHECK(ordinary_mamba->dtypes[0] == vt::DType::kBF16);
}

TEST_CASE("GPTQ4 model boundaries keep F16 hidden tap and F32 gathered logits") {
  const auto owned_half = [](std::vector<int64_t> shape,
                             const std::vector<float>& values, bool nk = false) {
    vllm::OwnedTensor weight;
    weight.dtype = vt::DType::kF16;
    weight.rank = static_cast<int>(shape.size());
    weight.nk = nk;
    for (int i = 0; i < weight.rank; ++i) weight.shape[i] = shape[i];
    weight.bytes.resize(values.size() * sizeof(uint16_t));
    for (size_t i = 0; i < values.size(); ++i) {
      const uint16_t bits = vt::F32ToF16(values[i]);
      std::memcpy(weight.bytes.data() + i * sizeof(bits), &bits, sizeof(bits));
    }
    return weight;
  };
  vllm::HfConfig config;
  config.hidden_size = 4;
  config.vocab_size = 6;
  config.num_hidden_layers = 0;
  config.rms_norm_eps = 1e-6;
  vllm::Qwen3_5DenseWeights weights;
  weights.gptq4_checkpoint = true;
  weights.precision.activation = vt::DType::kF16;
  weights.precision.gdn_conv_state = vt::DType::kF16;
  const std::vector<float> embedding{
      0.25f, 0.5f, -0.25f, 0.125f,
      0.5f, -0.25f, 0.125f, 0.375f,
      -0.375f, 0.25f, 0.5f, -0.125f,
      0.125f, 0.375f, -0.5f, 0.25f,
      0.375f, 0.125f, 0.25f, -0.5f,
      -0.25f, 0.5f, 0.375f, 0.125f};
  const std::vector<float> gamma{0.0f, 0.125f, -0.25f, 0.5f};
  const std::vector<float> head{
      0.5f, 0.25f, -0.125f, 0.375f,
      -0.25f, 0.5f, 0.375f, -0.125f,
      0.125f, -0.5f, 0.25f, 0.375f,
      0.375f, 0.125f, -0.25f, 0.5f,
      -0.125f, 0.375f, 0.5f, 0.25f,
      0.25f, -0.125f, 0.375f, -0.5f};
  weights.embed_tokens = owned_half({6, 4}, embedding);
  weights.final_norm = owned_half({4}, gamma);
  weights.lm_head = owned_half({6, 4}, head, true);
  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  const std::vector<int32_t> ids{1, 2, 3}, positions{0, 1, 2};
  vllm::v1::CommonAttentionMetadata attn;
  attn.num_reqs = 1;
  attn.num_actual_tokens = 3;
  attn.query_start_loc = {0, 3};
  attn.seq_lens = {3};
  attn.block_table_num_cols = 1;
  attn.block_table_tensor = {0};
  attn.slot_mapping = {0, 1, 2};
  vllm::Qwen3_5MTPHiddenStates tap;
  const auto logits = vllm::Qwen3_5DenseModel::ForwardDeviceTap(
      ids, positions, attn, {}, {}, {}, weights, config, queue,
      &tap, {0, 2});
  REQUIRE(tap.tensor.dtype == vt::DType::kF16);
  REQUIRE(tap.tensor.shape[0] == 3);
  REQUIRE(tap.tensor.shape[1] == 4);
  REQUIRE(logits.on_device());
  REQUIRE(logits.device_tensor.dtype == vt::DType::kF32);
  REQUIRE(logits.rows == 2);
  std::vector<uint16_t> tap_bits(12);
  std::vector<float> actual_logits(12);
  auto& backend = vt::GetBackend(queue.device);
  backend.Copy(queue, tap_bits.data(), tap.tensor.data, tap_bits.size() * 2);
  backend.Copy(queue, actual_logits.data(), logits.device_tensor.data,
               actual_logits.size() * sizeof(float));
  backend.Synchronize(queue);
  for (int t = 0; t < 3; ++t) {
    const int token = ids[t];
    float square = 0;
    for (int j = 0; j < 4; ++j)
      square += embedding[token * 4 + j] * embedding[token * 4 + j];
    const float inv = 1.0f / std::sqrt(square / 4.0f + 1e-6f);
    for (int j = 0; j < 4; ++j) {
      const float expected = vt::F16ToF32(vt::F32ToF16(
          embedding[token * 4 + j] * inv * (1.0f + gamma[j])));
      const float actual = vt::F16ToF32(tap_bits[t * 4 + j]);
      CHECK(std::abs(actual - expected) < 0.001f);
    }
  }
  for (int row = 0; row < 2; ++row) for (int vocab = 0; vocab < 6; ++vocab) {
    float expected = 0;
    const int selected = row == 0 ? 0 : 2;
    for (int j = 0; j < 4; ++j)
      expected += vt::F16ToF32(tap_bits[selected * 4 + j]) * head[vocab * 4 + j];
    CHECK(std::abs(actual_logits[row * 6 + vocab] - expected) < 0.001f);
  }
  vt::DestroyQueue(queue);
}

TEST_CASE("GPTQ4 two-layer XPU prefill dispatches every packed text projection") {
  vllm::HfConfig config;
  config.hidden_size = 128;
  config.intermediate_size = 128;
  config.vocab_size = 6;
  config.num_hidden_layers = 2;
  config.layer_types = {"linear_attention", "full_attention"};
  config.num_attention_heads = 2;
  config.num_key_value_heads = 1;
  config.head_dim = 64;
  config.rotary_dim = 32;
  config.rope_theta = 1000000.0;
  config.rms_norm_eps = 1e-6;
  config.linear_num_key_heads = 1;
  config.linear_num_value_heads = 2;
  config.linear_key_head_dim = 64;
  config.linear_value_head_dim = 64;
  config.linear_conv_kernel_dim = 4;

  vllm::Qwen3_5DenseWeights weights;
  weights.gptq4_checkpoint = true;
  weights.precision.activation = vt::DType::kF16;
  weights.precision.gdn_conv_state = vt::DType::kF16;
  weights.embed_tokens = ConstantHalf({6, 128}, 0.03125f);
  weights.final_norm = ConstantHalf({128}, 0.0f);
  weights.lm_head = ConstantHalf({6, 128}, 0.015625f, true);
  weights.layers.resize(2);
  for (auto& layer : weights.layers) {
    layer.input_layernorm = ConstantHalf({128}, 0.0f);
    layer.post_attention_layernorm = ConstantHalf({128}, 0.0f);
    layer.gptq4.mlp_gate_up = ConstantPacked(128, 256);
    layer.gptq4.mlp_down = ConstantPacked(128, 128);
  }
  auto& gdn = weights.layers[0];
  gdn.is_linear_attention = true;
  gdn.gptq4.gdn_qkvz = ConstantPacked(128, 384);
  gdn.gptq4.gdn_out = ConstantPacked(128, 128);
  gdn.gdn.in_proj_ba = ConstantHalf({4, 128}, 0.00390625f, true);
  gdn.gdn.conv1d_weight = ConstantHalf({256, 1, 4}, 0.125f);
  gdn.gdn.a_log = vllm::dense_loaders::MakeOwned(vt::DType::kF32, {2});
  gdn.gdn.dt_bias = vllm::dense_loaders::MakeOwned(vt::DType::kF32, {2});
  const float a_log[] = {0.0f, 0.0f}, dt_bias[] = {-2.0f, -2.0f};
  std::memcpy(gdn.gdn.a_log.bytes.data(), a_log, sizeof(a_log));
  std::memcpy(gdn.gdn.dt_bias.bytes.data(), dt_bias, sizeof(dt_bias));
  gdn.gdn.norm_weight = ConstantHalf({64}, 0.0f);
  auto& attn = weights.layers[1];
  attn.gptq4.attn_qkv = ConstantPacked(128, 384);
  attn.gptq4.attn_out = ConstantPacked(128, 128);
  attn.attn.q_norm = ConstantHalf({64}, 0.0f);
  attn.attn.k_norm = ConstantHalf({64}, 0.0f);

  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  auto& backend = vt::GetBackend(queue.device);
  constexpr int T = 64, block_size = 128;
  void* kv_data = vt::Alloc(queue.device, 2 * block_size * 64 * 2);
  void* ssm_data = vt::Alloc(queue.device, 2 * 64 * 64 * 4);
  void* conv_data = vt::Alloc(queue.device, 256 * 3 * 2);
  backend.Memset(queue, kv_data, 0, 2 * block_size * 64 * 2);
  backend.Memset(queue, ssm_data, 0, 2 * 64 * 64 * 4);
  backend.Memset(queue, conv_data, 0, 256 * 3 * 2);
  vllm::PagedKvCache kv;
  kv.data = kv_data;
  kv.dtype = vt::DType::kF16;
  kv.num_blocks = 1;
  kv.block_size = block_size;
  kv.num_kv_heads = 1;
  kv.head_size = 64;
  vllm::GdnStateCache state;
  state.ssm_state = vt::Tensor::Contiguous(
      ssm_data, vt::DType::kF32, queue.device, {1, 2, 64, 64});
  state.conv_state = vt::Tensor::Contiguous(
      conv_data, vt::DType::kF16, queue.device, {1, 256, 3});

  std::vector<int32_t> ids(T), positions(T);
  for (int i = 0; i < T; ++i) {
    ids[i] = 1 + i % 5;
    positions[i] = i;
  }
  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = T;
  am.query_start_loc = {0, T};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {T};
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = T;
  am.max_seq_len = T;
  am.block_table_num_cols = 1;
  am.block_table_tensor = {0};
  am.slot_mapping.assign(positions.begin(), positions.end());
  am.causal = true;
  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = T;
  gm.num_actual_tokens = T;
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, T};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, T};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  const auto conv = vllm::v1::ComputeCausalConv1dMetadata(
      *gm.non_spec_query_start_loc);
  gm.batch_ptr = conv.batch_ptr;
  gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;

  const auto before = vllm::dense_gptq4::GetDispatchCounts();
  vllm::Qwen3_5MTPHiddenStates tap;
  std::vector<vllm::PagedKvCache> kv_caches{kv};
  std::vector<vllm::GdnStateCache> gdn_caches{state};
  const std::vector<int32_t> logits_indices{T - 1};
  auto model = vllm::BorrowQwen3_5DenseLoadedModel(weights);
  vllm::ModelRegistry::Prepare(*model, config, queue);
  vllm::ModelForwardInput input{ids, positions, am, gm, kv_caches, gdn_caches,
                                config, queue, logits_indices};
  input.num_reqs = 1;
  input.hidden_tap = &tap;
  const auto logits = vllm::ModelRegistry::Forward(*model, input);
  REQUIRE(logits.on_device());
  REQUIRE(logits.rows == 1);
  REQUIRE(logits.vocab == 6);
  CHECK(tap.tensor.dtype == vt::DType::kF16);
  std::vector<float> values(6);
  backend.Copy(queue, values.data(), logits.device_tensor.data, values.size() * 4);
  backend.Synchronize(queue);
  for (float value : values) CHECK(std::isfinite(value));
  const auto after = vllm::dense_gptq4::GetDispatchCounts();
  using vllm::dense_gptq4::Projection;
  const auto calls = [&](Projection projection) {
    const size_t index = static_cast<size_t>(projection);
    return after.calls[index] - before.calls[index];
  };
  CHECK(calls(Projection::kGdnQkvz) == 1);
  CHECK(calls(Projection::kGdnBa) == 1);
  CHECK(calls(Projection::kGdnOut) == 1);
  CHECK(calls(Projection::kAttnQkv) == 1);
  CHECK(calls(Projection::kAttnOut) == 1);
  CHECK(calls(Projection::kMlpGateUp) == 2);
  CHECK(calls(Projection::kMlpDown) == 2);
  vt::Free(queue.device, conv_data);
  vt::Free(queue.device, ssm_data);
  vt::Free(queue.device, kv_data);
  vt::DestroyQueue(queue);
}

TEST_CASE("GPTQ4 real 64-layer text prefill and decode use the packed XPU path") {
  const char* checkpoint = std::getenv("VLLM_CPP_GPTQ4_CHECKPOINT_DIR");
  if (checkpoint == nullptr) {
    MESSAGE("Set VLLM_CPP_GPTQ4_CHECKPOINT_DIR for the pinned 64-layer text gate");
    return;
  }
  XpuQueueOwner owner;
  auto& queue = owner.queue;
  XpuCacheAllocations allocations{queue};
  const auto config = vllm::LoadHfConfig(std::string(checkpoint) + "/config.json");
  std::vector<vllm::SafetensorsFile> shards;
  for (int i = 1; i <= 5; ++i)
    shards.push_back(vllm::SafetensorsFile::Open(
        std::string(checkpoint) + "/model-0000" + std::to_string(i) +
        "-of-00005.safetensors"));
  auto model = vllm::ModelRegistry::Load(
      config, vllm::ModelSource::FromSafetensors(shards, &queue));
  vllm::ModelRegistry::Prepare(*model, config, queue);
  constexpr int block_size = 128, prefill = 64;
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  for (const std::string& type : config.layer_types) {
    if (type == "linear_attention") {
      const int64_t hv = config.linear_num_value_heads;
      const int64_t dv = config.linear_value_head_dim;
      const int64_t dk = config.linear_key_head_dim;
      const int64_t conv_dim = 2 * config.linear_num_key_heads * dk + hv * dv;
      const int64_t conv_len = config.linear_conv_kernel_dim - 1;
      vllm::GdnStateCache state;
      state.ssm_state = vt::Tensor::Contiguous(
          allocations.Zero(static_cast<size_t>(2 * hv * dv * dk) * 4),
          vt::DType::kF32, queue.device, {2, hv, dv, dk});
      state.conv_state = vt::Tensor::Contiguous(
          allocations.Zero(static_cast<size_t>(2 * conv_dim * conv_len) * 2),
          vt::DType::kF16, queue.device, {2, conv_dim, conv_len});
      gdn_state.push_back(state);
    } else {
      vllm::PagedKvCache cache;
      cache.data = allocations.Zero(static_cast<size_t>(
          4 * 2 * block_size * config.num_key_value_heads * config.head_dim) * 2);
      cache.dtype = vt::DType::kF16;
      cache.num_blocks = 4;
      cache.block_size = block_size;
      cache.num_kv_heads = config.num_key_value_heads;
      cache.head_size = config.head_dim;
      attn_kv.push_back(cache);
    }
  }
  REQUIRE(attn_kv.size() + gdn_state.size() == 64);

  const auto attention_meta = [](int query_len, int context) {
    vllm::v1::CommonAttentionMetadata meta;
    meta.num_reqs = 1;
    meta.num_actual_tokens = query_len;
    meta.query_start_loc = {0, query_len};
    meta.query_start_loc_cpu = meta.query_start_loc;
    meta.seq_lens = {context + query_len};
    meta.seq_lens_cpu = meta.seq_lens;
    meta.max_query_len = query_len;
    meta.max_seq_len = context + query_len;
    meta.block_table_num_cols = 1;
    meta.block_table_tensor = {0};
    for (int i = 0; i < query_len; ++i)
      meta.slot_mapping.push_back(context + i);
    meta.causal = true;
    return meta;
  };
  const auto gdn_meta = [](int query_len, bool initial) {
    vllm::v1::GDNAttentionMetadata meta;
    meta.num_actual_tokens = query_len;
    meta.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    meta.non_spec_query_start_loc = std::vector<int32_t>{0, query_len};
    if (initial) {
      meta.num_decodes = 1;
      meta.num_decode_tokens = query_len;
    } else {
      meta.num_prefills = 1;
      meta.num_prefill_tokens = query_len;
      meta.has_initial_state = std::vector<uint8_t>{0};
      meta.prefill_query_start_loc = std::vector<int32_t>{0, query_len};
      meta.prefill_state_indices = std::vector<int32_t>{0};
      meta.prefill_has_initial_state = std::vector<uint8_t>{0};
      const auto conv = vllm::v1::ComputeCausalConv1dMetadata(
          *meta.non_spec_query_start_loc);
      meta.batch_ptr = conv.batch_ptr;
      meta.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
    }
    return meta;
  };
  const auto read_logits = [&](const vllm::ForwardLogits& logits,
                                int expected_rows) {
    REQUIRE(logits.on_device());
    REQUIRE(logits.rows == expected_rows);
    REQUIRE(logits.vocab == config.vocab_size);
    std::vector<float> values(static_cast<size_t>(logits.rows * logits.vocab));
    vt::GetBackend(queue.device).Copy(queue, values.data(),
                                      logits.device_tensor.data, values.size() * 4);
    vt::GetBackend(queue.device).Synchronize(queue);
    for (float value : values) CHECK(std::isfinite(value));
    return values;
  };
  const auto compare_distribution = [&](const std::vector<float>& batched,
                                        int row, const std::vector<float>& isolated,
                                        const char* stage) {
    const size_t vocab = static_cast<size_t>(config.vocab_size);
    REQUIRE(isolated.size() == vocab);
    REQUIRE(batched.size() >= static_cast<size_t>(row + 1) * vocab);
    const float* actual = batched.data() + static_cast<size_t>(row) * vocab;
    const float* reference = isolated.data();
    float max_logit_difference = 0.0f;
    double actual_max = -INFINITY, reference_max = -INFINITY;
    for (size_t token = 0; token < vocab; ++token) {
      max_logit_difference = std::max(
          max_logit_difference, std::abs(actual[token] - reference[token]));
      actual_max = std::max(actual_max, static_cast<double>(actual[token]));
      reference_max = std::max(reference_max, static_cast<double>(reference[token]));
    }
    double actual_sum = 0.0, reference_sum = 0.0;
    for (size_t token = 0; token < vocab; ++token) {
      actual_sum += std::exp(static_cast<double>(actual[token]) - actual_max);
      reference_sum += std::exp(static_cast<double>(reference[token]) - reference_max);
    }
    const double actual_log_z = actual_max + std::log(actual_sum);
    const double reference_log_z = reference_max + std::log(reference_sum);
    double kl = 0.0, tv = 0.0;
    for (size_t token = 0; token < vocab; ++token) {
      const double log_p = static_cast<double>(reference[token]) - reference_log_z;
      const double log_q = static_cast<double>(actual[token]) - actual_log_z;
      const double p = std::exp(log_p), q = std::exp(log_q);
      kl += p * (log_p - log_q);
      tv += std::abs(p - q);
    }
    tv *= 0.5;
    MESSAGE("GPTQ4 B2 " << std::string(stage) << " request " << row
            << " max|logit diff| " << max_logit_difference
            << ", KL " << kl << ", TV " << tv);
    CHECK(kl <= 0.01);
    CHECK(tv <= 0.02);
  };
  std::vector<int32_t> ids(prefill), positions(prefill);
  for (int i = 0; i < prefill; ++i) {
    ids[i] = 100 + i % 11;
    positions[i] = i;
  }
  const std::vector<int32_t> prefill_indices{prefill - 1};
  const auto before = vllm::dense_gptq4::GetDispatchCounts();
  const auto prefill_am = attention_meta(prefill, 0);
  const auto prefill_gm = gdn_meta(prefill, false);
  vllm::ModelForwardInput prefill_input{
      ids, positions, prefill_am, prefill_gm, attn_kv, gdn_state,
      config, queue, prefill_indices};
  prefill_input.num_reqs = 1;
  const auto prefill_values = read_logits(
      vllm::ModelRegistry::Forward(*model, prefill_input), 1);

  const std::vector<int32_t> next_id{111}, next_pos{prefill};
  const std::vector<int32_t> decode_indices{0};
  const auto decode_am = attention_meta(1, prefill);
  const auto decode_gm = gdn_meta(1, true);
  vllm::ModelForwardInput decode_input{
      next_id, next_pos, decode_am, decode_gm, attn_kv, gdn_state,
      config, queue, decode_indices};
  decode_input.num_reqs = 1;
  decode_input.pure_decode = true;
  decode_input.uniform_query_len = 1;
  const auto decode_values = read_logits(
      vllm::ModelRegistry::Forward(*model, decode_input), 1);
  CHECK(prefill_values != decode_values);

  // Two fresh requests of different lengths must agree with the same requests
  // run in isolation, including their separate GDN state rows and KV blocks.
  const std::vector<int32_t> lengths{16, 8};
  std::vector<int32_t> batch_ids, batch_positions;
  for (int request = 0; request < 2; ++request)
    for (int token = 0; token < lengths[request]; ++token) {
      batch_ids.push_back(150 + request * 19 + token % 11);
      batch_positions.push_back(token);
    }
  auto batch_am = attention_meta(24, 0);
  batch_am.num_reqs = 2;
  batch_am.query_start_loc = {0, 16, 24};
  batch_am.query_start_loc_cpu = batch_am.query_start_loc;
  batch_am.seq_lens = {16, 8};
  batch_am.seq_lens_cpu = batch_am.seq_lens;
  batch_am.max_query_len = 16;
  batch_am.max_seq_len = 16;
  batch_am.block_table_tensor = {0, 1};
  batch_am.slot_mapping.clear();
  for (int token = 0; token < 16; ++token)
    batch_am.slot_mapping.push_back(token);
  for (int token = 0; token < 8; ++token)
    batch_am.slot_mapping.push_back(block_size + token);
  auto batch_gm = gdn_meta(24, false);
  batch_gm.num_prefills = 2;
  batch_gm.has_initial_state = std::vector<uint8_t>{0, 0};
  batch_gm.non_spec_state_indices_tensor = std::vector<int32_t>{0, 1};
  batch_gm.non_spec_query_start_loc = std::vector<int32_t>{0, 16, 24};
  batch_gm.prefill_query_start_loc = std::vector<int32_t>{0, 16, 24};
  batch_gm.prefill_state_indices = std::vector<int32_t>{0, 1};
  batch_gm.prefill_has_initial_state = std::vector<uint8_t>{0, 0};
  const auto batch_conv = vllm::v1::ComputeCausalConv1dMetadata(
      *batch_gm.non_spec_query_start_loc);
  batch_gm.batch_ptr = batch_conv.batch_ptr;
  batch_gm.token_chunk_offset_ptr = batch_conv.token_chunk_offset_ptr;
  const std::vector<int32_t> batch_indices{15, 23};
  vllm::ModelForwardInput batch_input{
      batch_ids, batch_positions, batch_am, batch_gm, attn_kv, gdn_state,
      config, queue, batch_indices};
  batch_input.num_reqs = 2;
  const auto batch_values = read_logits(
      vllm::ModelRegistry::Forward(*model, batch_input), 2);
  auto batch_decode_am = attention_meta(2, 0);
  batch_decode_am.num_reqs = 2;
  batch_decode_am.query_start_loc = {0, 1, 2};
  batch_decode_am.query_start_loc_cpu = batch_decode_am.query_start_loc;
  batch_decode_am.seq_lens = {17, 9};
  batch_decode_am.seq_lens_cpu = batch_decode_am.seq_lens;
  batch_decode_am.max_query_len = 1;
  batch_decode_am.max_seq_len = 17;
  batch_decode_am.block_table_tensor = {0, 1};
  batch_decode_am.slot_mapping = {16, block_size + 8};
  auto batch_decode_gm = gdn_meta(2, true);
  batch_decode_gm.num_decodes = 2;
  batch_decode_gm.non_spec_state_indices_tensor = std::vector<int32_t>{0, 1};
  batch_decode_gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1, 2};
  const std::vector<int32_t> batch_next_ids{173, 191};
  const std::vector<int32_t> batch_next_positions{16, 8};
  const std::vector<int32_t> batch_decode_indices{0, 1};
  vllm::ModelForwardInput batch_decode_input{
      batch_next_ids, batch_next_positions, batch_decode_am, batch_decode_gm,
      attn_kv, gdn_state, config, queue, batch_decode_indices};
  batch_decode_input.num_reqs = 2;
  batch_decode_input.pure_decode = true;
  batch_decode_input.uniform_query_len = 1;
  const auto batch_decode_values = read_logits(
      vllm::ModelRegistry::Forward(*model, batch_decode_input), 2);
  for (int request = 0; request < 2; ++request) {
    const int offset = request == 0 ? 0 : 16;
    const int length = lengths[request];
    std::vector<int32_t> solo_ids(batch_ids.begin() + offset,
                                  batch_ids.begin() + offset + length);
    std::vector<int32_t> solo_positions(batch_positions.begin() + offset,
                                        batch_positions.begin() + offset + length);
    auto solo_am = attention_meta(length, 0);
    solo_am.block_table_tensor = {request};
    for (auto& slot : solo_am.slot_mapping)
      slot += request * block_size;
    auto solo_gm = gdn_meta(length, false);
    solo_gm.non_spec_state_indices_tensor = std::vector<int32_t>{request};
    solo_gm.prefill_state_indices = std::vector<int32_t>{request};
    const std::vector<int32_t> solo_indices{length - 1};
    vllm::ModelForwardInput solo_input{
        solo_ids, solo_positions, solo_am, solo_gm, attn_kv, gdn_state,
        config, queue, solo_indices};
    solo_input.num_reqs = 1;
    const auto solo_values = read_logits(
        vllm::ModelRegistry::Forward(*model, solo_input), 1);
    compare_distribution(batch_values, request, solo_values, "prefill");
    const std::vector<int32_t> solo_next_ids{batch_next_ids[request]};
    const std::vector<int32_t> solo_next_positions{length};
    const std::vector<int32_t> solo_decode_indices{0};
    auto solo_decode_am = attention_meta(1, length);
    solo_decode_am.block_table_tensor = {request};
    solo_decode_am.slot_mapping[0] += request * block_size;
    auto solo_decode_gm = gdn_meta(1, true);
    solo_decode_gm.non_spec_state_indices_tensor = std::vector<int32_t>{request};
    vllm::ModelForwardInput solo_decode_input{
        solo_next_ids, solo_next_positions, solo_decode_am, solo_decode_gm,
        attn_kv, gdn_state, config, queue, solo_decode_indices};
    solo_decode_input.num_reqs = 1;
    solo_decode_input.pure_decode = true;
    solo_decode_input.uniform_query_len = 1;
    const auto solo_decode_values = read_logits(
        vllm::ModelRegistry::Forward(*model, solo_decode_input), 1);
    compare_distribution(batch_decode_values, request, solo_decode_values,
                         "decode");
  }
  const auto after = vllm::dense_gptq4::GetDispatchCounts();
  using vllm::dense_gptq4::Projection;
  const auto calls = [&](Projection projection) {
    const size_t index = static_cast<size_t>(projection);
    return after.calls[index] - before.calls[index];
  };
  const uint64_t gdn_layers = gdn_state.size();
  const uint64_t attn_layers = attn_kv.size();
  CHECK(calls(Projection::kGdnQkvz) == 8 * gdn_layers);
  CHECK(calls(Projection::kGdnBa) == 8 * gdn_layers);
  CHECK(calls(Projection::kGdnOut) == 8 * gdn_layers);
  CHECK(calls(Projection::kAttnQkv) == 8 * attn_layers);
  CHECK(calls(Projection::kAttnOut) == 8 * attn_layers);
  CHECK(calls(Projection::kMlpGateUp) == 512);
  CHECK(calls(Projection::kMlpDown) == 512);

  // Optional short timing screen. The first 256-token call warms its shape;
  // the measured call resets the same state slot and excludes load/JIT.
  if (std::getenv("VLLM_CPP_GPTQ4_SHORT_BENCH") != nullptr) {
    constexpr int prompt_tokens = 256, output_tokens = 8;
    std::vector<int32_t> bench_ids(prompt_tokens), bench_positions(prompt_tokens);
    for (int token = 0; token < prompt_tokens; ++token) {
      bench_ids[token] = 100 + token % 11;
      bench_positions[token] = token;
    }
    auto bench_am = attention_meta(prompt_tokens, 0);
    bench_am.block_table_num_cols = 2;
    bench_am.block_table_tensor = {0, 1};
    const auto bench_gm = gdn_meta(prompt_tokens, false);
    const std::vector<int32_t> bench_indices{prompt_tokens - 1};
    vllm::ModelForwardInput bench_input{
        bench_ids, bench_positions, bench_am, bench_gm, attn_kv, gdn_state,
        config, queue, bench_indices};
    bench_input.num_reqs = 1;
    auto& backend = vt::GetBackend(queue.device);
    auto run_prefill = [&] {
      const auto result = vllm::ModelRegistry::Forward(*model, bench_input);
      REQUIRE(result.on_device());
      backend.Synchronize(queue);
    };
    run_prefill();
    const auto prefill_start = std::chrono::steady_clock::now();
    run_prefill();
    const auto prefill_end = std::chrono::steady_clock::now();
    const double prefill_seconds =
        std::chrono::duration<double>(prefill_end - prefill_start).count();
    std::chrono::steady_clock::time_point first_decode_end, last_decode_end;
    for (int step = 0; step < output_tokens; ++step) {
      const std::vector<int32_t> token_id{300 + step};
      const std::vector<int32_t> position{prompt_tokens + step};
      auto decode_am = attention_meta(1, prompt_tokens + step);
      decode_am.block_table_num_cols = 3;
      decode_am.block_table_tensor = {0, 1, 2};
      const auto decode_gm = gdn_meta(1, true);
      const std::vector<int32_t> decode_index{0};
      vllm::ModelForwardInput step_input{
          token_id, position, decode_am, decode_gm, attn_kv, gdn_state,
          config, queue, decode_index};
      step_input.num_reqs = 1;
      step_input.pure_decode = true;
      step_input.uniform_query_len = 1;
      const auto result = vllm::ModelRegistry::Forward(*model, step_input);
      REQUIRE(result.on_device());
      backend.Synchronize(queue);
      const auto end = std::chrono::steady_clock::now();
      if (step == 0) first_decode_end = end;
      last_decode_end = end;
    }
    const double decode_seconds =
        std::chrono::duration<double>(last_decode_end - first_decode_end).count();
    MESSAGE("GPTQ4 short B1 timing: prompt " << prompt_tokens
            << ", output " << output_tokens
            << ", warm prefill 1, prefill compute " << prefill_seconds
            << " s (" << prompt_tokens / prefill_seconds << " tokens/s), "
            << "decode interval " << decode_seconds / (output_tokens - 1)
            << " s/token (" << (output_tokens - 1) / decode_seconds
            << " tokens/s); load/JIT excluded, eager FP16 KV");
  }
}

TEST_CASE("GPTQ4 attention QKV matches the captured Python post-load owner") {
  const char* checkpoint = std::getenv("VLLM_CPP_GPTQ4_CHECKPOINT_DIR");
  const char* oracle_dir = std::getenv("VLLM_CPP_GPTQ4_ORACLE_DIR");
  if (checkpoint == nullptr || oracle_dir == nullptr) {
    MESSAGE("Set VLLM_CPP_GPTQ4_CHECKPOINT_DIR and "
            "VLLM_CPP_GPTQ4_ORACLE_DIR for the pinned real-weight comparison");
    return;
  }
  const auto shard = vllm::SafetensorsFile::Open(
      std::string(checkpoint) + "/model-00002-of-00005.safetensors");
  const auto oracle = vllm::SafetensorsFile::Open(
      std::string(oracle_dir) + "/attention_qkv_weights.safetensors");
  const vllm::TensorResolver get = [&shard](const std::string& name)
      -> const vllm::StTensor& { return shard.Get(name); };
  const std::string base = "model.language_model.layers.3.self_attn.";
  const auto weight = vllm::LoadMergedGptq4Weight(
      get, {base + "q_proj", base + "k_proj", base + "v_proj"},
      5120, {12288, 1024, 1024});
  const auto& reference_packed = oracle.Get("qweight_nt_int32");
  const auto& reference_scales = oracle.Get("scales_f16");
  const auto& reference_zero = oracle.Get("effective_zero_point_i8");
  REQUIRE(reference_packed.nbytes == weight.qweight.bytes.size());
  REQUIRE(reference_scales.nbytes == weight.scales.bytes.size());
  REQUIRE(reference_zero.nbytes == weight.zero_point.bytes.size());
  CHECK(std::memcmp(reference_packed.data, weight.qweight.bytes.data(),
                    reference_packed.nbytes) == 0);
  CHECK(std::memcmp(reference_scales.data, weight.scales.bytes.data(),
                    reference_scales.nbytes) == 0);
  CHECK(std::memcmp(reference_zero.data, weight.zero_point.bytes.data(),
                    reference_zero.nbytes) == 0);
  // Independently decoded coefficients, including both sides of a group
  // boundary and both K/V shards of the Python merged QKV owner. The expected
  // values were read from the captured Python post-load words and FP16 scales.
  const auto coefficient = [&weight](int output, int inner) {
    const uint32_t word = Word(weight.qweight, output * 640 + inner / 8);
    const int code = static_cast<int>((word >> (4 * (inner % 8))) & 15);
    const float scale = vt::F16ToF32(
        Scale(weight.scales, (inner / 128) * 14336 + output));
    return (code - 8) * scale;
  };
  CHECK(coefficient(0, 0) == -0.01194000244140625f);
  CHECK(coefficient(5, 127) == 0.0056915283203125f);
  CHECK(coefficient(12288, 129) == 0.034942626953125f);
  CHECK(coefficient(13312, 5119) == 0.03049468994140625f);

  vllm::HfConfig config;
  config.hidden_size = 5120;
  config.intermediate_size = 17408;
  config.num_attention_heads = 24;
  config.num_key_value_heads = 4;
  config.head_dim = 256;
  config.linear_num_key_heads = 16;
  config.linear_key_head_dim = 128;
  config.linear_num_value_heads = 48;
  config.linear_value_head_dim = 128;
  config.layer_types = {"linear_attention", "linear_attention",
                        "linear_attention", "full_attention"};
  const auto has = [&shard](const std::string& name) {
    const auto& names = shard.Names();
    return std::find(names.begin(), names.end(), name) != names.end();
  };
  const auto attention = vllm::LoadQwen3_5DenseGptq4Projections(
      get, has, config, 3, "model.language_model.");
  CHECK(attention.gdn_qkvz.k == 0);
  CHECK(attention.attn_qkv.n == 14336);
  CHECK(attention.attn_out.k == 6144);
  CHECK(attention.mlp_gate_up.n == 34816);
  CHECK(attention.mlp_down.k == 17408);
  CHECK(std::memcmp(reference_packed.data,
                    attention.attn_qkv.qweight.bytes.data(),
                    reference_packed.nbytes) == 0);
  const auto gdn = vllm::LoadQwen3_5DenseGptq4Projections(
      get, has, config, 0, "model.language_model.");
  CHECK(gdn.attn_qkv.k == 0);
  CHECK(gdn.gdn_qkvz.n == 16384);
  CHECK(gdn.gdn_out.k == 6144);
  CHECK(gdn.mlp_gate_up.n == 34816);
  CheckAgainstPostLoad(
      gdn.mlp_gate_up,
      std::string(oracle_dir) + "/mlp_gate_up_weights.safetensors");
  CheckAgainstPostLoad(
      gdn.mlp_down,
      std::string(oracle_dir) + "/mlp_down_weights.safetensors");
  CheckAgainstPostLoad(
      gdn.gdn_qkvz,
      std::string(oracle_dir) + "/gdn_qkvz_weights.safetensors");
}

TEST_CASE("GPTQ4 pinned checkpoint validates every text projection") {
  const char* checkpoint = std::getenv("VLLM_CPP_GPTQ4_FULL_AUDIT_DIR");
  if (checkpoint == nullptr) {
    MESSAGE("Set VLLM_CPP_GPTQ4_FULL_AUDIT_DIR for the 400-matrix audit");
    return;
  }
  const auto config = vllm::LoadHfConfig(std::string(checkpoint) + "/config.json");
  REQUIRE(config.num_hidden_layers == 64);
  std::vector<vllm::SafetensorsFile> shards;
  for (int i = 1; i <= 5; ++i) {
    const std::string index = i < 10 ? "0000" + std::to_string(i)
                                     : "000" + std::to_string(i);
    shards.push_back(vllm::SafetensorsFile::Open(
        std::string(checkpoint) + "/model-" + index +
        "-of-00005.safetensors"));
  }
  std::map<std::string, const vllm::SafetensorsFile*> where;
  for (const auto& shard : shards)
    for (const auto& name : shard.Names())
      REQUIRE(where.emplace(name, &shard).second);
  const vllm::TensorResolver get = [&where](const std::string& name)
      -> const vllm::StTensor& { return where.at(name)->Get(name); };
  const auto has = [&where](const std::string& name) {
    return where.find(name) != where.end();
  };
  size_t matrices = 0;
  size_t packed_bytes = 0;
  for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
    const auto layer = vllm::LoadQwen3_5DenseGptq4Projections(
        get, has, config, i, "model.language_model.");
    const auto count = [&matrices, &packed_bytes](const vllm::Gptq4Weight& w,
                                                 int source_matrices) {
      if (w.k == 0) return;
      matrices += source_matrices;
      packed_bytes += w.qweight.bytes.size();
    };
    count(layer.gdn_qkvz, 2);
    count(layer.gdn_out, 1);
    count(layer.attn_qkv, 3);
    count(layer.attn_out, 1);
    count(layer.mlp_gate_up, 2);
    count(layer.mlp_down, 1);
  }
  CHECK(matrices == 400);
  CHECK(packed_bytes > 0);
}

TEST_CASE("GPTQ4 pinned checkpoint loads the complete text weight graph") {
  const char* checkpoint = std::getenv("VLLM_CPP_GPTQ4_FULL_LOAD_DIR");
  if (checkpoint == nullptr) {
    MESSAGE("Set VLLM_CPP_GPTQ4_FULL_LOAD_DIR for the complete text load");
    return;
  }
  const auto config = vllm::LoadHfConfig(std::string(checkpoint) + "/config.json");
  std::vector<vllm::SafetensorsFile> shards;
  for (int i = 1; i <= 5; ++i) {
    shards.push_back(vllm::SafetensorsFile::Open(
        std::string(checkpoint) + "/model-0000" + std::to_string(i) +
        "-of-00005.safetensors"));
  }
  const auto model = vllm::LoadQwen3_5Dense(shards, config);
  REQUIRE(model.gptq4_checkpoint);
  CHECK(config.torch_dtype == "float16");
  CHECK(config.dtype_source == "text.dtype");
  CHECK(model.precision.activation == vt::DType::kF16);
  REQUIRE(model.layers.size() == 64);
  CHECK(model.embed_tokens.dtype == vt::DType::kF16);
  CHECK(model.embed_tokens.shape[0] == config.vocab_size);
  CHECK(model.lm_head.dtype == vt::DType::kF16);
  CHECK(model.lm_head.nk);
  CHECK(model.layers[0].gdn.in_proj_ba.dtype == vt::DType::kF16);
  CHECK(model.layers[0].gdn.in_proj_ba.shape[0] == 96);
  CHECK(model.layers[3].attn.q_norm.dtype == vt::DType::kF16);
  size_t merged_owners = 0;
  for (const auto& layer : model.layers) {
    merged_owners += layer.gptq4.mlp_gate_up.k != 0;
    merged_owners += layer.gptq4.mlp_down.k != 0;
    merged_owners += layer.gptq4.gdn_qkvz.k != 0;
    merged_owners += layer.gptq4.gdn_out.k != 0;
    merged_owners += layer.gptq4.attn_qkv.k != 0;
    merged_owners += layer.gptq4.attn_out.k != 0;
    CHECK(layer.gptq4.ResidentBytes() == 0);
  }
  CHECK(merged_owners == 256);
}
