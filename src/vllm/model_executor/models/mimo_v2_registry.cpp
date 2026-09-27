// MiMoV2 (`MiMoV2ForCausalLM`) registry TU — the additive self-registration
// seam for the MiMoV2 bring-up (MODEL-TEXT-mimo-v2, W1). Follows the
// deepseek_v4_registry.cpp / dots3_note_registry.cpp seam exactly: a NEW
// translation unit with ONE REGISTER_VLLM_MODEL line and ZERO edit to any
// shared array. It owns the arch entry points: the config hook (config-descent
// validation), the KV-cache spec (two groups: full-attention + sliding-window),
// and stub load/prepare/forward functions.
//
// SCOPE HONESTY: registering this arch makes it RESOLVE + parse config +
// build the hybrid KV-cache spec. The weight loader (W2) and forward pass
// (W3) are stubs that VT_CHECK(false, ...) — so the W1 gate is: the model is
// discoverable, the config parses, the KV-cache spec builds.
//
// The KV-cache has two groups:
//   1. FullAttentionSpec for full-attention layers (every 6th: 0,6,12,...,42):
//      4 KV heads, head_size=192, head_size_v=128
//   2. SlidingWindowSpec for SWA layers (the remaining 40):
//      8 KV heads, head_size=192, head_size_v=128, sliding_window=128
//
// The v_head_dim != head_dim pattern is new: both specs accept head_size_v
// as an optional parameter that defaults to head_size.

#include "vllm/model_executor/models/mimo_v2.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits carrier
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vllm {

// ---- Config field accessors ----
// HfConfig.raw is nlohmann::json; we read fields directly with defaults
// matching the checkpoint's configuration_mimo_v2.py.

namespace {

int64_t GetInt(const nlohmann::json& j, const char* key, int64_t def = 0) {
  if (const auto it = j.find(key); it != j.end() && it->is_number_integer())
    return it->get<int64_t>();
  if (const auto it = j.find(key); it != j.end() && it->is_number_float())
    return static_cast<int64_t>(it->get<double>());
  return def;
}

double GetDouble(const nlohmann::json& j, const char* key, double def = 0.0) {
  if (const auto it = j.find(key); it != j.end() && it->is_number())
    return it->get<double>();
  return def;
}

bool GetBool(const nlohmann::json& j, const char* key, bool def = false) {
  if (const auto it = j.find(key); it != j.end() && it->is_boolean())
    return it->get<bool>();
  return def;
}

std::string GetString(const nlohmann::json& j, const char* key,
                      const char* def = "") {
  if (const auto it = j.find(key); it != j.end() && it->is_string())
    return it->get<std::string>();
  return def;
}

std::vector<int> GetIntArray(const nlohmann::json& j, const char* key) {
  std::vector<int> out;
  if (const auto it = j.find(key); it != j.end() && it->is_array()) {
    for (const auto& el : *it) {
      if (el.is_number_integer())
        out.push_back(el.get<int>());
      else if (el.is_number_float())
        out.push_back(static_cast<int>(el.get<double>()));
    }
  }
  return out;
}

}  // namespace

MiMoV2Params ParseMiMoV2Params(const HfConfig& config) {
  MiMoV2Params p;
  const auto& raw = config.raw;

  p.hidden_size = config.hidden_size;
  p.num_hidden_layers = config.num_hidden_layers;
  p.vocab_size = config.vocab_size;
  p.num_attention_heads = config.num_attention_heads;
  p.intermediate_size = GetInt(raw, "intermediate_size");
  p.tie_word_embeddings = GetBool(raw, "tie_word_embeddings", false);

  // Attention geometry
  p.head_dim = GetInt(raw, "head_dim", 0);
  p.v_head_dim = GetInt(raw, "v_head_dim", 0);
  p.sliding_window = GetInt(raw, "sliding_window", 128);

  // partial_rotary_factor: stored as a double (0.334). rotary_dim is
  // int(head_dim * partial_rotary_factor) = int(192 * 0.334) = 64.
  p.partial_rotary_factor_num = static_cast<int64_t>(
      GetDouble(raw, "partial_rotary_factor", 0.0) * 1000.0);
  p.rotary_dim = GetInt(raw, "rotary_dim", 0);
  if (p.rotary_dim == 0 && p.head_dim > 0) {
    double prf = GetDouble(raw, "partial_rotary_factor", 1.0);
    p.rotary_dim = static_cast<int64_t>(p.head_dim * prf);
  }

  // Per-layer KV head counts
  p.num_kv_heads_full = GetInt(raw, "num_kv_heads_full", 0);
  p.num_kv_heads_swa = GetInt(raw, "num_kv_heads_swa", 0);

  // RoPE
  p.rope_theta_full = GetDouble(raw, "rope_theta_full", 0.0);
  p.rope_theta_swa = GetDouble(raw, "rope_theta_swa", 0.0);

  // Attention sink bias
  p.add_swa_attention_sink_bias =
      GetBool(raw, "add_swa_attention_sink_bias", false);
  p.attention_value_scale = GetDouble(raw, "attention_value_scale", 1.0);

  // Hybrid layer pattern
  p.hybrid_layer_pattern = GetIntArray(raw, "hybrid_layer_pattern");

  // MoE
  p.num_experts = GetInt(raw, "num_experts", 0);
  p.num_experts_per_tok = GetInt(raw, "num_experts_per_tok", 0);
  p.moe_intermediate_size = GetInt(raw, "moe_intermediate_size", 0);
  p.moe_router_dtype = GetString(raw, "moe_router_dtype", "bfloat16");
  p.n_shared_experts = GetBool(raw, "n_shared_experts", false);
  p.moe_layer_freq = GetIntArray(raw, "moe_layer_freq");

  // Norm
  p.rms_norm_eps = GetDouble(raw, "rms_norm_eps", 1e-6);
  p.hidden_act = GetString(raw, "hidden_act", "silu");

  // ---- Validation ----
  VT_CHECK(p.hidden_size > 0, "mimo_v2: hidden_size must be positive");
  VT_CHECK(p.num_hidden_layers > 0, "mimo_v2: num_hidden_layers must be positive");
  VT_CHECK(p.vocab_size > 0, "mimo_v2: vocab_size must be positive");
  VT_CHECK(p.num_attention_heads > 0,
           "mimo_v2: num_attention_heads must be positive");
  VT_CHECK(p.head_dim > 0, "mimo_v2: head_dim must be positive");
  VT_CHECK(p.v_head_dim > 0, "mimo_v2: v_head_dim must be positive");
  VT_CHECK(p.sliding_window > 0, "mimo_v2: sliding_window must be positive");
  VT_CHECK(p.num_kv_heads_full > 0,
           "mimo_v2: num_kv_heads_full must be positive");
  VT_CHECK(p.num_kv_heads_swa > 0,
           "mimo_v2: num_kv_heads_swa must be positive");
  VT_CHECK(!p.hybrid_layer_pattern.empty(),
           "mimo_v2: hybrid_layer_pattern must not be empty");
  VT_CHECK(static_cast<int64_t>(p.hybrid_layer_pattern.size()) ==
              p.num_hidden_layers,
           "mimo_v2: hybrid_layer_pattern length must equal num_hidden_layers");

  return p;
}

void ParseMiMoV2Config(const HfConfig& config) {
  // The resolve IS the validation: it throws on everything unrepresentable.
  (void)ParseMiMoV2Params(config);
}

// ---- KV-cache spec ----
v1::KVCacheConfig MakeMiMoV2KVCache(const HfConfig& config, int block_size,
                                     int num_blocks) {
  const MiMoV2Params p = ParseMiMoV2Params(config);
  const vt::DType dtype = v1::ResolveKvCacheDType();

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;

  // Two KV-cache groups: full-attention layers and SWA layers.
  // Full-attention layers: 4 KV heads, head_size=192, head_size_v=128
  // SWA layers: 8 KV heads, head_size=192, head_size_v=128, sliding_window=128
  std::vector<std::string> full_layer_names;
  std::vector<std::string> swa_layer_names;
  for (int i = 0; i < static_cast<int>(p.hybrid_layer_pattern.size()); ++i) {
    const std::string name =
        "model.layers." + std::to_string(i) + ".self_attn";
    if (p.hybrid_layer_pattern[i] == 0) {
      full_layer_names.push_back(name);
    } else {
      swa_layer_names.push_back(name);
    }
  }

  if (!full_layer_names.empty()) {
    kv.kv_cache_groups.emplace_back(
        std::move(full_layer_names),
        std::make_shared<v1::FullAttentionSpec>(
            block_size, static_cast<int>(p.num_kv_heads_full),
            static_cast<int>(p.head_dim), dtype,
            /*head_size_v=*/static_cast<int>(p.v_head_dim)));
  }

  if (!swa_layer_names.empty()) {
    kv.kv_cache_groups.emplace_back(
        std::move(swa_layer_names),
        std::make_shared<v1::SlidingWindowSpec>(
            block_size, static_cast<int>(p.num_kv_heads_swa),
            static_cast<int>(p.head_dim), dtype,
            static_cast<int>(p.sliding_window),
            /*head_size_v=*/static_cast<int>(p.v_head_dim)));
  }

  return kv;
}

// ---- Stub Load/Prepare/Forward (W2-W3 will implement) ----
std::unique_ptr<LoadedModel> LoadMiMoV2ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  (void)registration;
  // The config still has to be VALID to get a truthful refusal.
  ParseMiMoV2Config(config);
  (void)source;
  VT_CHECK(false,
           "MiMoV2ForCausalLM: the weight loader is not ported. W1 makes "
           "this architecture RESOLVE and makes its config.json PARSE and "
           "VALIDATE; it loads no weights. The loader is OWED to W2. "
           "Row MODEL-TEXT-mimo-v2, spec .agents/specs/mimov2.md, "
           "issue ISSUE-LOCAL-01M3F82S8ZYCTTDSKPFF5PGAH7.");
}

void PrepareMiMoV2ForCausalLM(LoadedModel& model, const HfConfig& config,
                                vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
  // Unreachable while the loader refuses. Still a refusal rather than a no-op.
  VT_CHECK(false,
           "MiMoV2ForCausalLM: prepare is not ported (no weights can be "
           "loaded yet -- W2 owes the loader). Row MODEL-TEXT-mimo-v2, "
           "issue ISSUE-LOCAL-01M3F82S8ZYCTTDSKPFF5PGAH7.");
}

ForwardLogits ForwardMiMoV2ForCausalLM(LoadedModel& model,
                                       const ModelForwardInput& input) {
  (void)model;
  (void)input;
  VT_CHECK(false,
           "MiMoV2ForCausalLM: the forward pass is not ported. W3 owes "
           "the hybrid attention (sink_bias, v_head_dim != head_dim, "
           "partial RoPE) and the MoE routing. Row MODEL-TEXT-mimo-v2, "
           "issue ISSUE-LOCAL-01M3F82S8ZYCTTDSKPFF5PGAH7.");
}

// ---- ModelInfo ----
inline constexpr ModelInfo kMimoV2Info{
    .is_text_generation_model = true,
    .is_hybrid = true,
    .supports_multimodal = false,
};

// ---- ModelFactory ----
inline constexpr ModelFactory kMimoV2Factory{
    .parse_config = &ParseMiMoV2Config,
    .load_weights = &LoadMiMoV2ForCausalLM,
    .prepare = &PrepareMiMoV2ForCausalLM,
    .forward = &ForwardMiMoV2ForCausalLM,
    .make_kv_cache = &MakeMiMoV2KVCache,
    .is_dense_model = false,
    .consumes_multi_kv = true,
};

REGISTER_VLLM_MODEL(mimo_v2, "MiMoV2ForCausalLM", kMimoV2Factory,
                    kMimoV2Info)

}  // namespace vllm
