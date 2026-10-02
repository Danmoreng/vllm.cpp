// MiMoV2 W1 gate test (MODEL-TEXT-mimo-v2, ISSUE-LOCAL-01M3F82S8ZYCTTDSKPFF5PGAH7).
//
// The W1 gate is: the model is discoverable in the registry, the config
// parses, and the KV-cache spec builds with the correct two-group geometry.
// This test constructs a minimal HfConfig matching the checkpoint's
// configuration_mimo_v2.py and exercises all three.

#include "vllm/model_executor/models/mimo_v2.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/v1/kv_cache_interface.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using vllm::HfConfig;
using vllm::ModelRegistry;
using vllm::ModelRegistration;
using vllm::MiMoV2Params;
using vllm::ParseMiMoV2Params;

namespace {

// Minimal config matching the MiMoV2.6-Flash checkpoint.
// Only the fields that ParseMiMoV2Params reads are included.
HfConfig MakeMiMoV2Config() {
  HfConfig config;
  config.model_type = "mimo_v2";
  config.architectures = {"MiMoV2ForCausalLM"};
  config.hidden_size = 4096;
  config.num_hidden_layers = 48;
  config.vocab_size = 152576;
  config.num_attention_heads = 64;

  nlohmann::json j;
  j["head_dim"] = 192;
  j["v_head_dim"] = 128;
  j["sliding_window"] = 128;
  j["partial_rotary_factor"] = 0.334;
  j["num_kv_heads_full"] = 4;
  j["num_kv_heads_swa"] = 8;
  j["rope_theta_full"] = 10000000.0;
  j["rope_theta_swa"] = 10000.0;
  j["add_swa_attention_sink_bias"] = true;
  j["attention_value_scale"] = 0.707;
  j["intermediate_size"] = 16384;
  j["tie_word_embeddings"] = false;
  j["rms_norm_eps"] = 1e-6;
  j["hidden_act"] = "silu";

  // 48-element hybrid pattern matching the actual checkpoint:
  // [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, ...]
  // Full-attention (0) at indices 0, 5, 11, 17, 23, 29, 35, 41, 47 → 9 layers
  std::vector<int> pattern(48, 1);
  pattern[0] = 0;
  for (int i = 5; i < 48; i += 6) pattern[i] = 0;
  j["hybrid_layer_pattern"] = pattern;

  // MoE: layer 0 dense, layers 1-47 MoE.
  j["n_routed_experts"] = 256;
  j["num_experts_per_tok"] = 8;
  j["moe_intermediate_size"] = 2048;
  j["moe_router_dtype"] = "bfloat16";
  j["n_shared_experts"] = false;

  std::vector<int> moe_freq(48, 1);
  moe_freq[0] = 0;
  j["moe_layer_freq"] = moe_freq;

  config.raw = j;
  return config;
}

}  // namespace

TEST_CASE("mimov2: architecture is discoverable in registry") {
  const auto registrations = ModelRegistry::Registrations();
  bool found = false;
  for (const auto& r : registrations) {
    if (r.architecture == "MiMoV2ForCausalLM") {
      found = true;
      REQUIRE(r.factory != nullptr);
      CHECK(r.factory->parse_config != nullptr);
      CHECK(r.factory->load_weights != nullptr);
      CHECK(r.factory->prepare != nullptr);
      CHECK(r.factory->forward != nullptr);
      CHECK(r.factory->make_kv_cache != nullptr);
      break;
    }
  }
  CHECK(found);
}

TEST_CASE("mimov2: Resolve returns the MiMoV2ForCausalLM registration") {
  const HfConfig config = MakeMiMoV2Config();
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  CHECK(reg.architecture == "MiMoV2ForCausalLM");
}

TEST_CASE("mimov2: ParseMiMoV2Params reads all fields correctly") {
  const HfConfig config = MakeMiMoV2Config();
  const MiMoV2Params p = ParseMiMoV2Params(config);

  CHECK(p.hidden_size == 4096);
  CHECK(p.num_hidden_layers == 48);
  CHECK(p.vocab_size == 152576);
  CHECK(p.num_attention_heads == 64);
  CHECK(p.head_dim == 192);
  CHECK(p.v_head_dim == 128);
  CHECK(p.sliding_window == 128);
  CHECK(p.rotary_dim == 64);  // int(192 * 0.334) = 64
  CHECK(p.num_kv_heads_full == 4);
  CHECK(p.num_kv_heads_swa == 8);
  CHECK(p.add_swa_attention_sink_bias == true);
  CHECK(p.attention_value_scale == 0.707);
  CHECK(p.hybrid_layer_pattern.size() == 48);
  CHECK(p.num_experts == 256);
  CHECK(p.num_experts_per_tok == 8);
  CHECK(p.moe_intermediate_size == 2048);
  CHECK(p.moe_layer_freq.size() == 48);
  CHECK(p.moe_layer_freq[0] == 0);
  CHECK(p.moe_layer_freq[1] == 1);
}

TEST_CASE("mimov2: hybrid_layer_pattern has 9 full-attn and 39 SWA layers") {
  const HfConfig config = MakeMiMoV2Config();
  const MiMoV2Params p = ParseMiMoV2Params(config);

  int full_count = 0;
  int swa_count = 0;
  for (int v : p.hybrid_layer_pattern) {
    if (v == 0)
      ++full_count;
    else
      ++swa_count;
  }
  CHECK(full_count == 9);   // indices 0,5,11,17,23,29,35,41,47
  CHECK(swa_count == 39);
}

TEST_CASE("mimov2: KV-cache spec builds with two groups") {
  const HfConfig config = MakeMiMoV2Config();
  const vllm::v1::KVCacheConfig kv =
      vllm::MakeMiMoV2KVCache(config, /*block_size=*/16, /*num_blocks=*/8);

  REQUIRE(kv.kv_cache_groups.size() == 2);

  // Group 0: full-attention layers (9 layers, 4 KV heads).
  CHECK(kv.kv_cache_groups[0].layer_names.size() == 9);
  const auto* full_spec = dynamic_cast<const vllm::v1::FullAttentionSpec*>(
      kv.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE(full_spec != nullptr);
  CHECK(full_spec->num_kv_heads == 4);
  CHECK(full_spec->head_size == 192);
  CHECK(full_spec->head_size_v == 128);

  // Group 1: SWA layers (39 layers, 8 KV heads, sliding_window=128).
  CHECK(kv.kv_cache_groups[1].layer_names.size() == 39);
  const auto* swa_spec = dynamic_cast<const vllm::v1::SlidingWindowSpec*>(
      kv.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(swa_spec != nullptr);
  CHECK(swa_spec->num_kv_heads == 8);
  CHECK(swa_spec->head_size == 192);
  CHECK(swa_spec->head_size_v == 128);
  CHECK(swa_spec->sliding_window == 128);
}

TEST_CASE("mimov2: KV-cache group layer names are correct") {
  const HfConfig config = MakeMiMoV2Config();
  const vllm::v1::KVCacheConfig kv =
      vllm::MakeMiMoV2KVCache(config, 16, 8);

  // Full-attention layers: 0, 5, 11, 17, 23, 29, 35, 41, 47 → 9 names.
  const auto& full_names = kv.kv_cache_groups[0].layer_names;
  CHECK(full_names[0] == "model.layers.0.self_attn");
  CHECK(full_names[1] == "model.layers.5.self_attn");
  CHECK(full_names[2] == "model.layers.11.self_attn");
  CHECK(full_names[8] == "model.layers.47.self_attn");

  // SWA layers: 1, 2, 3, 4, 6, 7, 8, ...
  const auto& swa_names = kv.kv_cache_groups[1].layer_names;
  CHECK(swa_names[0] == "model.layers.1.self_attn");
  CHECK(swa_names[1] == "model.layers.2.self_attn");
  CHECK(swa_names[4] == "model.layers.6.self_attn");
}

TEST_CASE("mimov2: parse_config rejects mismatched hybrid_layer_pattern length") {
  HfConfig config = MakeMiMoV2Config();
  // Truncate the pattern to 47 elements.
  config.raw["hybrid_layer_pattern"].erase(47);
  CHECK_THROWS_AS(ParseMiMoV2Params(config), std::runtime_error);
}

TEST_CASE("mimov2: parse_config rejects zero head_dim") {
  HfConfig config = MakeMiMoV2Config();
  config.raw["head_dim"] = 0;
  CHECK_THROWS_AS(ParseMiMoV2Params(config), std::runtime_error);
}

TEST_CASE("mimov2: parse_config rejects zero v_head_dim") {
  HfConfig config = MakeMiMoV2Config();
  config.raw["v_head_dim"] = 0;
  CHECK_THROWS_AS(ParseMiMoV2Params(config), std::runtime_error);
}
