#include <doctest/doctest.h>

#include <algorithm>
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
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/v1/kv_cache_interface.h"
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
