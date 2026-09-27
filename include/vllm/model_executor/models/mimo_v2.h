// MiMoV2 — typed config params parsed from HfConfig.
//
// (MODEL-TEXT-mimo-v2 W1) The MiMoV2 architecture is a hybrid full/sliding-
// window attention MoE model with per-layer KV geometry split: full-attention
// layers have 4 KV heads with head_dim=192 and v_head_dim=128, SWA layers have
// 8 KV heads with the same head_dim/v_head_dim. The v_head_dim != head_dim
// pattern is new to this codebase — FullAttentionSpec and SlidingWindowSpec
// both accept head_size_v as an optional parameter.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {

// Hybrid layer pattern: 0 = full attention, 1 = sliding-window attention.
// The checkpoint carries a 48-element list; every 6th layer (indices 0,6,12,
// ...,42) is full attention, the rest are SWA with sliding_window=128.
struct MiMoV2Params {
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t vocab_size = 0;
  int64_t num_attention_heads = 0;
  int64_t intermediate_size = 0;  // dense layer 0 only

  // Attention geometry
  int64_t head_dim = 0;       // 192 — Q/K head dim
  int64_t v_head_dim = 0;     // 128 — V head dim (NEW: != head_dim)
  int64_t partial_rotary_factor_num = 0;  // 334 (from 0.334)
  int64_t rotary_dim = 0;     // int(192 * 0.334) = 64
  int64_t sliding_window = 0; // 128

  // Per-layer KV geometry: full-attention layers have 4 KV heads, SWA layers
  // have 8 KV heads. Both use head_dim=192 for K, v_head_dim=128 for V.
  int64_t num_kv_heads_full = 0;   // 4
  int64_t num_kv_heads_swa = 0;    // 8

  // RoPE: different theta per layer type
  double rope_theta_full = 0.0;   // 10e6
  double rope_theta_swa = 0.0;    // 10e3

  // Attention sink bias (SWA layers only): [num_attention_heads] = [64]
  bool add_swa_attention_sink_bias = false;
  double attention_value_scale = 0.0;  // 0.707 — multiplied into V pre-cache

  // Hybrid layer pattern: 0 = full, 1 = SWA. Length == num_hidden_layers.
  std::vector<int> hybrid_layer_pattern;

  // MoE
  int64_t num_experts = 0;              // 256
  int64_t num_experts_per_tok = 0;      // 8
  int64_t moe_intermediate_size = 0;   // 2048
  std::string moe_router_dtype;         // "bfloat16" (but gate forces fp32)
  bool n_shared_experts = false;       // false — no shared expert

  // Layer 0 is dense, layers 1..N-1 are MoE
  // moe_layer_freq is a per-layer list: 0 = dense, 1 = MoE
  std::vector<int> moe_layer_freq;

  // Norm
  double rms_norm_eps = 0.0;   // 1e-6
  std::string hidden_act;      // "silu"

  bool tie_word_embeddings = false;
};

// Parses and validates all MiMoV2 config fields. Throws on missing required
// fields or invalid values.
MiMoV2Params ParseMiMoV2Params(const HfConfig& config);

// Validates config (delegates to ParseMiMoV2Params, void return).
void ParseMiMoV2Config(const HfConfig& config);

// Builds the hybrid KV-cache spec: two groups (full-attention + SWA).
v1::KVCacheConfig MakeMiMoV2KVCache(const HfConfig& config, int block_size,
                                     int num_blocks);

}  // namespace vllm
