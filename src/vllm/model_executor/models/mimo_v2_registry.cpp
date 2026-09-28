// MiMoV2 (`MiMoV2ForCausalLM`) registry TU — the additive self-registration
// seam for the MiMoV2 bring-up (MODEL-TEXT-mimo-v2, W1+W2+W3). Follows the
// deepseek_v4_registry.cpp / dots3_note_registry.cpp seam exactly: a NEW
// translation unit with ONE REGISTER_VLLM_MODEL line and ZERO edit to any
// shared array. It owns the arch entry points: the config hook (config-descent
// validation), the KV-cache spec (two groups: full-attention + sliding-window),
// the weight loader (W2), and the forward pass (W3, delegated to mimo_v2.cpp).
//
// The KV-cache has two groups:
//   1. FullAttentionSpec for full-attention layers (0,5,11,17,23,29,35,41,47):
//      4 KV heads, head_size=192, head_size_v=128
//   2. SlidingWindowSpec for SWA layers (the remaining 39):
//      8 KV heads, head_size=192, head_size_v=128, sliding_window=128
//
// The v_head_dim != head_dim pattern is new: both specs accept head_size_v
// as an optional parameter that defaults to head_size.

#include "vllm/model_executor/models/mimo_v2.h"
#include "vllm/model_executor/models/mimo_v2_weights.h"
#include "vllm/model_executor/models/dense_attn_block.h"  // StepInputs, BuildStepInputs
#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf
#include "vllm/model_executor/models/host_token_ids.h"  // ResolveHostTokenIds
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, PagedKvCache
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
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

  // Per-layer KV head counts. The checkpoint config uses "num_key_value_heads"
  // for full-attention layers and "swa_num_key_value_heads" for SWA layers.
  p.num_kv_heads_full = GetInt(raw, "num_kv_heads_full", 0);
  if (p.num_kv_heads_full == 0)
    p.num_kv_heads_full = GetInt(raw, "num_key_value_heads", 0);
  p.num_kv_heads_swa = GetInt(raw, "num_kv_heads_swa", 0);
  if (p.num_kv_heads_swa == 0)
    p.num_kv_heads_swa = GetInt(raw, "swa_num_key_value_heads", 0);

  // RoPE: "rope_theta" for full-attention, "swa_rope_theta" for SWA.
  p.rope_theta_full = GetDouble(raw, "rope_theta_full", 0.0);
  if (p.rope_theta_full == 0.0)
    p.rope_theta_full = GetDouble(raw, "rope_theta", 0.0);
  p.rope_theta_swa = GetDouble(raw, "rope_theta_swa", 0.0);

  // Attention sink bias
  p.add_swa_attention_sink_bias =
      GetBool(raw, "add_swa_attention_sink_bias", false);
  p.attention_value_scale = GetDouble(raw, "attention_value_scale", 1.0);

  // Hybrid layer pattern
  p.hybrid_layer_pattern = GetIntArray(raw, "hybrid_layer_pattern");

  // MoE
  p.num_experts = GetInt(raw, "n_routed_experts", 0);
  p.num_experts_per_tok = GetInt(raw, "num_experts_per_tok", 0);
  p.moe_intermediate_size = GetInt(raw, "moe_intermediate_size", 0);
  p.moe_router_dtype = GetString(raw, "moe_router_dtype", "bfloat16");
  p.n_shared_experts = GetBool(raw, "n_shared_experts", false);
  p.n_group = GetInt(raw, "n_group", 1);
  p.topk_group = GetInt(raw, "topk_group", 1);
  p.norm_topk_prob = GetBool(raw, "norm_topk_prob", true);
  // routed_scaling_factor is None in the config → 1.0
  if (auto it = raw.find("routed_scaling_factor"); it != raw.end()) {
    if (it->is_number()) {
      p.routed_scaling_factor = it->get<double>();
    } else {
      p.routed_scaling_factor = 1.0;
    }
  } else {
    p.routed_scaling_factor = 1.0;
  }
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
  // Pattern: [0,1,1,1,1,0,...] — full attention (0) at 0,5,11,17,23,29,35,41,47
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

// ---- LoadedModel subclass ----

class MiMoV2LoadedModel final : public LoadedModel {
 public:
  MiMoV2LoadedModel(const ModelRegistration& registration,
                    MiMoV2Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const MiMoV2Weights& weights() const { return weights_; }

 private:
  MiMoV2Weights weights_;
};

// ---- Load / Prepare / Forward ----

std::unique_ptr<LoadedModel> LoadMiMoV2ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    throw std::runtime_error(
        "MiMoV2ForCausalLM: GGUF is not supported for this architecture. "
        "Row MODEL-TEXT-mimo-v2, spec .agents/specs/mimov2.md, "
        "issue ISSUE-LOCAL-01M3F82S8ZYCTTDSKPFF5PGAH7.");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<MiMoV2LoadedModel>(
      registration, LoadMiMoV2Weights(*source.safetensors, config));
}

void PrepareMiMoV2ForCausalLM(LoadedModel& model, const HfConfig& config,
                                vt::Queue& queue) {
  // Following Dots3Note: prepare is a no-op. Materialization happens inside
  // load_weights; device upload is lazy via ResidentWeight.
  (void)model;
  (void)config;
  (void)queue;
}

// ---- Forward (delegated to mimo_v2.cpp) ----

// ForwardMiMoV2Device is defined in mimo_v2.cpp.
ForwardLogits ForwardMiMoV2Device(
    const std::vector<int32_t>& token_ids,
    const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const MiMoV2Weights& weights,
    const MultiKvCacheIndex* multi_kv,
    vt::Queue& queue,
    const std::vector<int32_t>& logits_indices);

ForwardLogits ForwardMiMoV2ForCausalLM(LoadedModel& model,
                                        const ModelForwardInput& input) {
  auto& mv = ModelAs<MiMoV2LoadedModel>(model, "MiMoV2ForCausalLM");
  const MiMoV2Weights& weights = mv.weights();

  // Resolve host token ids (handles async device-token-id path).
  std::vector<int32_t> device_ids;
  const std::vector<int32_t>& ids =
      ResolveHostTokenIds(input, &device_ids, "MiMoV2ForCausalLM");

  VT_CHECK(input.multi_kv != nullptr,
           "MiMoV2ForCausalLM: requires multi_kv (hybrid KV cache topology)");

  return ForwardMiMoV2Device(
      ids, input.positions, input.attn_meta, input.attn_kv, weights,
      input.multi_kv, input.queue, input.logits_indices);
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
