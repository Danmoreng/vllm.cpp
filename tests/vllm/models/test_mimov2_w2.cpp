// MiMoV2 W2 gate test (MODEL-TEXT-mimo-v2, ISSUE-LOCAL-01M3F82S8ZYCTTDSKPFF5PGAH7).
//
// The W2 gate is: tensor enumeration produces the right names, accounting
// classifies a synthetic EXL3 checkpoint's names correctly (missing, duplicated,
// unaccounted all detected), and the weight loader struct compiles and links.
//
// We cannot load a real checkpoint in a unit test (the EXL3 checkpoint is
// 50+ GB), so the W2 test exercises the enumeration and accounting logic
// directly, plus checks that the MiMoV2Weights / MiMoV2Projection /
// MiMoV2LayerWeights / MiMoV2MoeWeights / MiMoV2DenseMlpWeights structs are
// usable.

#include "vllm/model_executor/models/mimo_v2.h"
#include "vllm/model_executor/models/mimo_v2_weights.h"
#include "vllm/transformers_utils/hf_config.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using vllm::MiMoV2Params;
using vllm::MiMoV2Weights;
using vllm::MiMoV2Projection;
using vllm::MiMoV2LayerWeights;
using vllm::MiMoV2MoeWeights;
using vllm::MiMoV2DenseMlpWeights;
using vllm::MiMoV2ExpertWeights;
using vllm::MiMoV2Accounting;
using vllm::MiMoV2Tensor;
using vllm::EnumerateMiMoV2Tensors;
using vllm::AccountMiMoV2Tensors;
using vllm::ParseMiMoV2Params;
using vllm::HfConfig;

namespace {

HfConfig MakeMiMoV2Config() {
  vllm::HfConfig config;
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

  std::vector<int> pattern(48, 1);
  pattern[0] = 0;
  for (int i = 5; i < 48; i += 6) pattern[i] = 0;
  j["hybrid_layer_pattern"] = pattern;

  j["num_experts"] = 256;
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

TEST_CASE("mimov2_w2: EnumerateMiMoV2Tensors counts all expected tensors") {
  const MiMoV2Params p = ParseMiMoV2Params(MakeMiMoV2Config());
  const auto tensors = EnumerateMiMoV2Tensors(p);

  CHECK(!tensors.empty());

  // Non-layer: embed_tokens, norm, lm_head = 3
  int non_layer = 0;
  for (const auto& t : tensors) {
    if (t.name.find("layers.") == std::string::npos) ++non_layer;
  }
  CHECK(non_layer == 3);

  // Each layer has:
  // - 2 layernorms (input_layernorm, post_attention_layernorm)
  // - 4 attention projections (q, k, v, o)
  // - sink_bias on 39 SWA layers
  // - layer 0: 3 dense MLP projections (gate, up, down)
  // - layers 1-47: 2 router tensors + 256 * 3 expert projections
  //
  // Total per layer:
  //   layer 0 (full + dense): 2 + 4 + 0 (full, no sink) + 3 = 9
  //   full + MoE (8 layers): 2 + 4 + 0 + 2 + 768 = 776
  //   SWA + MoE (39 layers): 2 + 4 + 1 + 2 + 768 = 777
  //
  // Total = 3 + 9 + 8*776 + 39*777 = 3 + 9 + 6208 + 30303 = 36523

  CHECK(tensors.size() == 36523);
}

TEST_CASE("mimov2_w2: EnumerateMiMoV2Tensors has correct names for layer 0") {
  const MiMoV2Params p = ParseMiMoV2Params(MakeMiMoV2Config());
  const auto tensors = EnumerateMiMoV2Tensors(p);

  // Find layer 0 tensors.
  std::vector<std::string> l0_names;
  for (const auto& t : tensors) {
    if (t.name.find("model.layers.0.") == 0) {
      l0_names.push_back(t.name);
    }
  }

  // Layer 0 is full-attention + dense MLP: no sink_bias.
  bool has_sink = false;
  for (const auto& n : l0_names) {
    if (n.find("attention_sink_bias") != std::string::npos) {
      has_sink = true;
      break;
    }
  }
  CHECK(!has_sink);

  // Layer 0 has dense MLP (gate_proj, up_proj, down_proj).
  bool has_gate_proj = false, has_up_proj = false, has_down_proj = false;
  for (const auto& n : l0_names) {
    if (n == "model.layers.0.mlp.gate_proj.weight") has_gate_proj = true;
    if (n == "model.layers.0.mlp.up_proj.weight") has_up_proj = true;
    if (n == "model.layers.0.mlp.down_proj.weight") has_down_proj = true;
  }
  CHECK(has_gate_proj);
  CHECK(has_up_proj);
  CHECK(has_down_proj);
}

TEST_CASE("mimov2_w2: EnumerateMiMoV2Tensors has sink_bias on SWA layers only") {
  const MiMoV2Params p = ParseMiMoV2Params(MakeMiMoV2Config());
  const auto tensors = EnumerateMiMoV2Tensors(p);

  // Count sink_bias tensors.
  int sink_count = 0;
  for (const auto& t : tensors) {
    if (t.name.find("attention_sink_bias") != std::string::npos) {
      ++sink_count;
      // Verify it's on a SWA layer (not layer 0, 5, 11, 17, 23, 29, 35, 41, 47).
      // Extract layer number.
      auto pos = t.name.find("model.layers.");
      auto dot = t.name.find('.', pos + 13);
      int layer = std::stoi(t.name.substr(pos + 13, dot - pos - 13));
      bool is_full = (layer == 0 || layer == 5 || layer == 11 || layer == 17 ||
                     layer == 23 || layer == 29 || layer == 35 ||
                     layer == 41 || layer == 47);
      CHECK(!is_full);
    }
  }
  CHECK(sink_count == 39);  // 39 SWA layers have sink_bias.
}

TEST_CASE("mimov2_w2: EnumerateMiMoV2Tensors has experts on MoE layers only") {
  const MiMoV2Params p = ParseMiMoV2Params(MakeMiMoV2Config());
  const auto tensors = EnumerateMiMoV2Tensors(p);

  // Layer 0 has NO expert tensors (it's dense).
  bool l0_has_experts = false;
  for (const auto& t : tensors) {
    if (t.name.find("model.layers.0.mlp.experts.") == 0) {
      l0_has_experts = true;
      break;
    }
  }
  CHECK(!l0_has_experts);

  // Layer 1 has 256 * 3 = 768 expert projections + 2 router tensors.
  int l1_expert_count = 0;
  for (const auto& t : tensors) {
    if (t.name.find("model.layers.1.mlp.experts.") == 0) {
      ++l1_expert_count;
    }
  }
  CHECK(l1_expert_count == 768);  // 256 experts * 3 projections (gate, up, down)
}

TEST_CASE("mimov2_w2: AccountMiMoV2Tensors detects missing tensors") {
  const MiMoV2Params p = ParseMiMoV2Params(MakeMiMoV2Config());
  const auto expected = EnumerateMiMoV2Tensors(p);

  // Provide all expected names except one.
  std::vector<std::string> present;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (i != 0) present.push_back(expected[i].name);
  }

  MiMoV2Accounting acc = AccountMiMoV2Tensors(p, present);
  CHECK(acc.missing.size() == 1);
  CHECK(acc.duplicated.empty());
  CHECK(acc.unaccounted.empty());
}

TEST_CASE("mimov2_w2: AccountMiMoV2Tensors detects unaccounted tensors") {
  const MiMoV2Params p = ParseMiMoV2Params(MakeMiMoV2Config());
  const auto expected = EnumerateMiMoV2Tensors(p);

  std::vector<std::string> present;
  for (const auto& t : expected) present.push_back(t.name);
  present.push_back("model.layers.0.self_attn.bogus.weight");

  MiMoV2Accounting acc = AccountMiMoV2Tensors(p, present);
  CHECK(acc.missing.empty());
  CHECK(acc.duplicated.empty());
  CHECK(acc.unaccounted.size() == 1);
}

TEST_CASE("mimov2_w2: AccountMiMoV2Tensors maps EXL3 suffixes to logical names") {
  const MiMoV2Params p = ParseMiMoV2Params(MakeMiMoV2Config());
  const auto expected = EnumerateMiMoV2Tensors(p);

  // Take the bf16 expected names and generate EXL3 on-disk names for all
  // linear weights. EXL3 on-disk has .trellis, .suh, .svh, .mul1 suffixes
  // instead of .weight.
  std::vector<std::string> present;
  for (const auto& t : expected) {
    if (t.name.size() > 7 && t.name.substr(t.name.size() - 7) == ".weight") {
      // Linear weight: strip .weight, add EXL3 suffixes.
      std::string base = t.name.substr(0, t.name.size() - 7);
      present.push_back(base + ".trellis");
      present.push_back(base + ".suh");
      present.push_back(base + ".svh");
      present.push_back(base + ".mul1");
    } else {
      // Non-linear (norms, router gate, sink_bias, e_score): kept as-is.
      present.push_back(t.name);
    }
  }

  MiMoV2Accounting acc = AccountMiMoV2Tensors(p, present);
  CHECK(acc.missing.empty());
  CHECK(acc.duplicated.empty());
  CHECK(acc.unaccounted.empty());
}

TEST_CASE("mimov2_w2: AccountMiMoV2Tensors skips MTP and multimodal prefixes") {
  const MiMoV2Params p = ParseMiMoV2Params(MakeMiMoV2Config());
  const auto expected = EnumerateMiMoV2Tensors(p);

  std::vector<std::string> present;
  for (const auto& t : expected) present.push_back(t.name);
  // Add tensors that should be silently skipped.
  present.push_back("model.mtp.layers.0.self_attn.q_proj.weight");
  present.push_back("model.vision_tower.vision.blocks.0.weight");
  present.push_back("audio_tower.audio.blocks.0.weight");

  MiMoV2Accounting acc = AccountMiMoV2Tensors(p, present);
  CHECK(acc.missing.empty());
  CHECK(acc.duplicated.empty());
  CHECK(acc.unaccounted.empty());
}

TEST_CASE("mimov2_w2: MiMoV2Weights struct is default-constructible") {
  MiMoV2Weights w;
  CHECK(w.params.hidden_size == 0);
  CHECK(w.layers.empty());
  CHECK(w.is_exl3 == false);
}

TEST_CASE("mimov2_w2: MiMoV2LayerWeights has correct flags") {
  MiMoV2LayerWeights lw;
  CHECK(lw.is_full_attention == false);
  CHECK(lw.is_moe == false);
  CHECK(lw.attn.has_sink_bias == false);
}
