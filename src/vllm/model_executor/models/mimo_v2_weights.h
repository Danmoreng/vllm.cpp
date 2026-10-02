// MiMoV2 — weight structs and loader declarations (private header).
//
// (MODEL-TEXT-mimo-v2 W2) The weight structures mirror the checkpoint's
// tensor layout: separate q_proj/k_proj/v_proj (NOT fused qkv_proj, despite
// config saying attention_projection_layout: "fused_qkv"), per-layer KV
// geometry split (full-attn: 4 KV heads, SWA: 8 KV heads), v_head_dim=128
// != head_dim=192, attention_sink_bias on SWA layers only, and layer-0 dense
// vs layers-1..47 MoE.
//
// Both bf16 and EXL3 paths are handled: each linear weight has an OwnedTensor
// (bf16) and an Exl3Weight (EXL3) field. IsExl3() decides which is populated.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/mimo_v2.h"  // MiMoV2Params
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor, Exl3Weight, TensorResolver
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// One attention projection. Holds both bf16 and EXL3 representations; only one
// is populated. `IsExl3()` checks the EXL3 field.
struct MiMoV2Projection {
  OwnedTensor bf16;       // [in, out] transposed (Matmul-B layout)
  Exl3Weight exl3;        // trellis + suh + svh + codebook

  bool IsExl3() const { return !exl3.trellis.Empty(); }
};

// Per-layer attention weights.
struct MiMoV2AttnWeights {
  MiMoV2Projection q_proj;
  MiMoV2Projection k_proj;
  MiMoV2Projection v_proj;
  MiMoV2Projection o_proj;

  // SWA layers only: [num_attention_heads] = [64], F32 per-head scalar.
  // Absent on full-attention layers (add_full_attention_sink_bias=false).
  OwnedTensor sink_bias;  // F32 [num_attention_heads] or empty
  bool has_sink_bias = false;
};

// Dense MLP (layer 0 only, intermediate_size=16384).
struct MiMoV2DenseMlpWeights {
  MiMoV2Projection gate_proj;
  MiMoV2Projection up_proj;
  MiMoV2Projection down_proj;
};

// One MoE expert (moe_intermediate_size=2048).
struct MiMoV2ExpertWeights {
  MiMoV2Projection gate_proj;
  MiMoV2Projection up_proj;
  MiMoV2Projection down_proj;
};

// MoE block (layers 1..47).
struct MiMoV2MoeWeights {
  OwnedTensor router_gate;             // BF16 [num_experts, hidden_size]
  OwnedTensor e_score_correction_bias;  // F32 [num_experts]
  std::vector<MiMoV2ExpertWeights> experts;  // [n_routed_experts]
};

// One backbone layer. `is_full_attention` selects the KV geometry. `is_moe`
// selects dense vs MoE MLP.
struct MiMoV2LayerWeights {
  bool is_full_attention = false;
  bool is_moe = false;

  OwnedTensor input_layernorm;          // BF16 [hidden_size]
  OwnedTensor post_attention_layernorm;  // BF16 [hidden_size]

  MiMoV2AttnWeights attn;

  // Only one of these is populated per layer.
  MiMoV2DenseMlpWeights dense_mlp;   // layer 0
  MiMoV2MoeWeights moe;             // layers 1..47
};

// Top-level weight container.
struct MiMoV2Weights {
  MiMoV2Params params;

  OwnedTensor embed_tokens;  // BF16 [vocab_size, hidden_size]
  OwnedTensor final_norm;    // BF16 [hidden_size]
  MiMoV2Projection lm_head;   // BF16 or EXL3

  std::vector<MiMoV2LayerWeights> layers;

  bool is_exl3 = false;  // set during load
};

// Accounting result: every on-disk tensor must be classified.
struct MiMoV2Accounting {
  std::vector<std::string> missing;
  std::vector<std::string> duplicated;
  std::vector<std::string> unaccounted;
  int64_t language = 0;
};

// A named expected tensor, tagged with its consumer for diagnostics.
struct MiMoV2Tensor {
  std::string name;
  std::string consumer;
};

// Enumerates every expected tensor name (bf16 ".weight" form). EXL3 on-disk
// names are mapped to these logical names during accounting by stripping the
// EXL3 suffix and appending ".weight".
std::vector<MiMoV2Tensor> EnumerateMiMoV2Tensors(const MiMoV2Params& p);

// Classifies every on-disk name against the expected set. EXL3 suffixes
// (.trellis, .suh, .svh, .mul1, .mcg) are stripped and ".weight" appended
// before lookup, so an EXL3 checkpoint accounts against the same name set.
MiMoV2Accounting AccountMiMoV2Tensors(
    const MiMoV2Params& p,
    const std::vector<std::string>& present);

// Loads all weights from safetensors shards. Performs full accounting before
// materialization. Throws on missing/duplicated/unaccounted tensors.
MiMoV2Weights LoadMiMoV2Weights(const std::vector<SafetensorsFile>& shards,
                                const HfConfig& config);

}  // namespace vllm
