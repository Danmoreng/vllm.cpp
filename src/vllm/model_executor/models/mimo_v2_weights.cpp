// MiMoV2 — weight loader (MODEL-TEXT-mimo-v2 W2).
//
// Loads every tensor from safetensors shards, performs full accounting (every
// on-disk name must be classified as expected, duplicated, or unaccounted), then
// materializes weights into the MiMoV2Weights tree.
//
// Two paths are handled:
//  - bf16: linear weights stored as "{name}.weight" BF16 tensors.
//  - EXL3: linear weights stored as separate .trellis/.suh/.svh/.mul1 (or .mcg)
//    tensors, detected via IsExl3Projection.
//
// On-disk attention names use separate q_proj/k_proj/v_proj (NOT fused
// qkv_proj), despite config saying attention_projection_layout: "fused_qkv".

#include "vllm/model_executor/models/mimo_v2_weights.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "vllm/model_executor/layers/quantization/exl3_checkpoint.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"

namespace vllm {

using dense_loaders::IsExl3Projection;
using dense_loaders::LoadExl3;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadBf16Direct;
using dense_loaders::MakeOwned;

namespace {

// ── EXL3 suffix handling ────────────────────────────────────────────────────

// Strips EXL3 suffixes from a tensor name, returning the logical base name
// and a bool indicating whether any suffix was stripped.
struct StrippedName {
  std::string base;
  bool is_exl3_suffix = false;
};

StrippedName StripExl3Suffix(const std::string& name) {
  static const std::array<const char*, 5> kSuffixes = {
      ".trellis", ".suh", ".svh", ".mul1", ".mcg"};
  for (const char* suf : kSuffixes) {
    size_t pos = name.rfind(suf);
    if (pos != std::string::npos && pos + strlen(suf) == name.size()) {
      return {name.substr(0, pos), true};
    }
  }
  return {name, false};
}

// ── Tensor index helper ─────────────────────────────────────────────────────

// Builds a name → StTensor* index from all shards, detecting duplicates.
struct TensorIndex {
  std::unordered_map<std::string, const StTensor*> by_name;
  std::vector<std::string> duplicates;
};

TensorIndex BuildTensorIndex(const std::vector<SafetensorsFile>& shards) {
  TensorIndex idx;
  for (const auto& shard : shards) {
    for (const std::string& name : shard.Names()) {
      auto [it, inserted] = idx.by_name.try_emplace(name, &shard.Get(name));
      if (!inserted) {
        idx.duplicates.push_back(name);
      }
    }
  }
  return idx;
}

// TensorResolver adaptor: looks up by name, throws if missing.
const StTensor& ResolveTensor(const TensorIndex& idx,
                                const std::string& name) {
  auto it = idx.by_name.find(name);
  if (it == idx.by_name.end()) {
    throw std::runtime_error("mimo_v2: missing tensor: " + name);
  }
  return *it->second;
}

// has-tensor predicate for IsExl3Projection / LoadExl3.
struct HasTensor {
  const TensorIndex& idx;
  bool operator()(const std::string& name) const {
    return idx.by_name.count(name) > 0;
  }
};

// TensorResolver lambda wrapper.
struct GetTensor {
  const TensorIndex& idx;
  // TensorResolver lambda wrapper. Returns a reference (TensorResolver signature).
  const StTensor& operator()(const std::string& name) const {
    return ResolveTensor(idx, name);
  }
};

// ── F32 vector loader (same as dots3_note_device.cpp:266) ──────────────────

OwnedTensor LoadF32VectorImpl(const TensorIndex& idx,
                               const std::string& name) {
  const StTensor& t = ResolveTensor(idx, name);
  if (t.shape.size() != 1) {
    throw std::runtime_error("mimo_v2: expected 1-D F32 vector: " + name);
  }
  OwnedTensor o = MakeOwned(vt::DType::kF32, t.shape);
  if (t.nbytes != o.bytes.size()) {
    throw std::runtime_error("mimo_v2: byte-size mismatch for F32 vector: " + name);
  }
  std::memcpy(o.bytes.data(), t.data, t.nbytes);
  return o;
}

// ── Layer-kind helpers ─────────────────────────────────────────────────────

bool IsFullAttentionLayer(const MiMoV2Params& p, int64_t layer) {
  if (layer < 0 || layer >= static_cast<int64_t>(p.hybrid_layer_pattern.size())) {
    return false;
  }
  return p.hybrid_layer_pattern[static_cast<size_t>(layer)] == 0;
}

bool IsMoeLayer(const MiMoV2Params& p, int64_t layer) {
  if (layer < 0 || layer >= static_cast<int64_t>(p.moe_layer_freq.size())) {
    return false;
  }
  return p.moe_layer_freq[static_cast<size_t>(layer)] == 1;
}

[[maybe_unused]]
int64_t NumKvHeads(const MiMoV2Params& p, int64_t layer) {
  // Full-attention layers have 4 KV heads; SWA layers have 8.
  return IsFullAttentionLayer(p, layer) ? p.num_kv_heads_full : p.num_kv_heads_swa;
}

// ── Per-projection loader ──────────────────────────────────────────────────

MiMoV2Projection LoadProjection(const TensorIndex& idx,
                                 const std::string& base) {
  MiMoV2Projection proj;
  HasTensor has{idx};
  GetTensor get{idx};

  if (IsExl3Projection(has, base)) {
    proj.exl3 = LoadExl3(get, has, base);
  } else {
    proj.bf16 = LoadBf16Transposed(get, base + ".weight");
  }
  return proj;
}

}  // namespace

// ── Tensor enumeration ──────────────────────────────────────────────────────

std::vector<MiMoV2Tensor> EnumerateMiMoV2Tensors(const MiMoV2Params& p) {
  std::vector<MiMoV2Tensor> out;

  auto add = [&](const std::string& name, const std::string& consumer) {
    out.push_back({name, consumer});
  };

  // Non-layer
  add("model.embed_tokens.weight", "embed");
  add("model.norm.weight", "final_norm");
  add("lm_head.weight", "lm_head");

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string lp = "model.layers." + std::to_string(l) + ".";
    const bool full = IsFullAttentionLayer(p, l);
    const bool moe = IsMoeLayer(p, l);

    // Layernorms
    add(lp + "input_layernorm.weight", "layernorm");
    add(lp + "post_attention_layernorm.weight", "layernorm");

    // Attention projections (separate q/k/v, not fused)
    add(lp + "self_attn.q_proj.weight", "attn");
    add(lp + "self_attn.k_proj.weight", "attn");
    add(lp + "self_attn.v_proj.weight", "attn");
    add(lp + "self_attn.o_proj.weight", "attn");

    // Sink bias on SWA layers only
    if (!full && p.add_swa_attention_sink_bias) {
      add(lp + "self_attn.attention_sink_bias", "sink_bias");
    }

    // MLP
    if (moe) {
      // Router gate
      add(lp + "mlp.gate.weight", "router");
      add(lp + "mlp.gate.e_score_correction_bias", "e_score");
      // Experts
      for (int64_t e = 0; e < p.num_experts; ++e) {
        const std::string ep =
            lp + "mlp.experts." + std::to_string(e) + ".";
        add(ep + "gate_proj.weight", "expert");
        add(ep + "up_proj.weight", "expert");
        add(ep + "down_proj.weight", "expert");
      }
    } else {
      // Dense MLP (layer 0)
      add(lp + "mlp.gate_proj.weight", "dense_mlp");
      add(lp + "mlp.up_proj.weight", "dense_mlp");
      add(lp + "mlp.down_proj.weight", "dense_mlp");
    }
  }

  return out;
}

// ── Accounting ──────────────────────────────────────────────────────────────

MiMoV2Accounting AccountMiMoV2Tensors(
    const MiMoV2Params& p,
    const std::vector<std::string>& present) {
  MiMoV2Accounting acc;

  // Build expected name set.
  std::unordered_set<std::string> expected;
  std::unordered_map<std::string, std::string> consumers;
  for (const auto& t : EnumerateMiMoV2Tensors(p)) {
    expected.insert(t.name);
    consumers[t.name] = t.consumer;
  }

  // For each present name, strip EXL3 suffix if present and append ".weight"
  // to normalize to the logical name. Then classify.
  //
  // EXL3 linear weights appear as 3-4 on-disk tensors (.trellis, .suh, .svh,
  // and optionally .mul1/.mcg). These are sibling components of ONE logical
  // weight, NOT duplicates of each other. A real duplicate is the same logical
  // weight present in BOTH bf16 (.weight) and EXL3 (suffix) form — a mixed
  // checkpoint that should be refused.
  std::unordered_map<std::string, bool> seen;  // logical name → is_exl3
  for (const auto& raw : present) {
    // Skip known-irrelevant prefixes (vision_config, audio_config tensors
    // from the multimodal checkpoint — out of scope for this text-only row).
    if (raw.rfind("model.vision_tower.", 0) == 0 ||
        raw.rfind("model.audio_tower.", 0) == 0 ||
        raw.rfind("vision_", 0) == 0 ||
        raw.rfind("audio_", 0) == 0 ||
        raw.rfind("mtp.", 0) == 0 ||
        raw.rfind("model.mtp.", 0) == 0) {
      continue;
    }

    StrippedName sn = StripExl3Suffix(raw);
    std::string logical = sn.base;
    // EXL3 linear weights: the base name (after suffix strip) maps to the
    // logical "{base}.weight" name. Non-linear tensors (norms, router gate,
    // sink_bias, e_score) have no suffix and map directly.
    if (sn.is_exl3_suffix) {
      logical = sn.base + ".weight";
    }

    if (expected.count(logical) == 0) {
      acc.unaccounted.push_back(raw);
      continue;
    }
    auto it = seen.find(logical);
    if (it != seen.end()) {
      // Same logical name already seen. If both are EXL3 siblings, that's
      // fine (trellis + suh + svh + mul1 for the same weight). If one is bf16
      // and the other EXL3, it's a mixed-checkpoint error.
      if (it->second != sn.is_exl3_suffix) {
        acc.duplicated.push_back(raw);
      }
      // Same-path siblings are not duplicates.
      continue;
    }
    seen[logical] = sn.is_exl3_suffix;
    ++acc.language;
  }

  // Missing = expected - seen.
  for (const auto& name : expected) {
    if (seen.count(name) == 0) {
      acc.missing.push_back(name);
    }
  }

  return acc;
}

// ── Weight loader ───────────────────────────────────────────────────────────

MiMoV2Weights LoadMiMoV2Weights(const std::vector<SafetensorsFile>& shards,
                                const HfConfig& config) {
  MiMoV2Params p = ParseMiMoV2Params(config);

  // Build the tensor index.
  TensorIndex idx = BuildTensorIndex(shards);

  // Collect present names.
  std::vector<std::string> present;
  present.reserve(idx.by_name.size());
  for (const auto& [name, _] : idx.by_name) {
    present.push_back(name);
  }

  // Account.
  MiMoV2Accounting acc = AccountMiMoV2Tensors(p, present);

  // VT_CHECK: missing, duplicated, unaccounted.
  if (!acc.missing.empty()) {
    std::ostringstream oss;
    oss << "mimo_v2: missing tensors (" << acc.missing.size() << "):";
    for (const auto& n : acc.missing) {
      oss << "\n  " << n;
    }
    VT_CHECK(false, oss.str());
  }
  if (!acc.duplicated.empty()) {
    std::ostringstream oss;
    oss << "mimo_v2: duplicated tensors (" << acc.duplicated.size() << "):";
    for (const auto& n : acc.duplicated) {
      oss << "\n  " << n;
    }
    VT_CHECK(false, oss.str());
  }
  if (!acc.unaccounted.empty()) {
    std::ostringstream oss;
    oss << "mimo_v2: unaccounted tensors (" << acc.unaccounted.size() << "):";
    for (const auto& n : acc.unaccounted) {
      oss << "\n  " << n;
    }
    VT_CHECK(false, oss.str());
  }

  // Determine if this is an EXL3 checkpoint.
  const bool is_exl3 = IsExl3Checkpoint(config);

  // Materialize.
  MiMoV2Weights w;
  w.params = p;
  w.is_exl3 = is_exl3;

  GetTensor get{idx};

  // Non-layer
  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  w.lm_head = LoadProjection(idx, "lm_head");

  // Layers
  w.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    MiMoV2LayerWeights& lw = w.layers[static_cast<size_t>(l)];
    const std::string lp = "model.layers." + std::to_string(l) + ".";
    const bool full = IsFullAttentionLayer(p, l);
    const bool moe = IsMoeLayer(p, l);

    lw.is_full_attention = full;
    lw.is_moe = moe;

    // Layernorms
    lw.input_layernorm = LoadBf16Direct(get, lp + "input_layernorm.weight");
    lw.post_attention_layernorm =
        LoadBf16Direct(get, lp + "post_attention_layernorm.weight");

    // Attention
    lw.attn.q_proj = LoadProjection(idx, lp + "self_attn.q_proj");
    lw.attn.k_proj = LoadProjection(idx, lp + "self_attn.k_proj");
    lw.attn.v_proj = LoadProjection(idx, lp + "self_attn.v_proj");
    lw.attn.o_proj = LoadProjection(idx, lp + "self_attn.o_proj");

    // Sink bias on SWA layers only
    if (!full && p.add_swa_attention_sink_bias) {
      lw.attn.has_sink_bias = true;
      lw.attn.sink_bias =
          LoadF32VectorImpl(idx, lp + "self_attn.attention_sink_bias");
    }

    // MLP
    if (moe) {
      // Router gate: BF16 [num_experts, hidden_size]
      lw.moe.router_gate = LoadBf16Direct(get, lp + "mlp.gate.weight");
      // e_score_correction_bias: F32 [num_experts]
      lw.moe.e_score_correction_bias =
          LoadF32VectorImpl(idx, lp + "mlp.gate.e_score_correction_bias");

      // Experts
      lw.moe.experts.resize(static_cast<size_t>(p.num_experts));
      for (int64_t e = 0; e < p.num_experts; ++e) {
        const std::string ep =
            lp + "mlp.experts." + std::to_string(e) + ".";
        MiMoV2ExpertWeights& ex = lw.moe.experts[static_cast<size_t>(e)];
        ex.gate_proj = LoadProjection(idx, ep + "gate_proj");
        ex.up_proj = LoadProjection(idx, ep + "up_proj");
        ex.down_proj = LoadProjection(idx, ep + "down_proj");
      }
    } else {
      // Dense MLP (layer 0)
      lw.dense_mlp.gate_proj = LoadProjection(idx, lp + "mlp.gate_proj");
      lw.dense_mlp.up_proj = LoadProjection(idx, lp + "mlp.up_proj");
      lw.dense_mlp.down_proj = LoadProjection(idx, lp + "mlp.down_proj");
    }
  }

  return w;
}

}  // namespace vllm
