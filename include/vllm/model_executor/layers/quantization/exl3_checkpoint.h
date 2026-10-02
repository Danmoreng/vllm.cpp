// EXL3 checkpoint detection — the shared, model-agnostic predicate.
//
// WHY THIS FILE EXISTS (QUANT-EXL3-GENERALISE W1). `IsExl3Checkpoint` lived
// in an anonymous namespace inside `deepseek_v4_weights.cpp`, so no other
// model loader could call it. That made the EXL3 path a DSV4-private arm —
// the parallel-path shape AGENTS.md forbids. This header lifts the predicate
// to a shared seam: any model loader can `#include` it and ask "is this an
// EXL3 checkpoint?" without depending on DSV4.
//
// The check is `quantization_config.quant_method == "exl3"` only. It does NOT
// gate on `version` or `codebook` — those are loader-specific invariants.
// DeepSeek-V4's loader keeps its own `version == "rank-sliced-deepseek-v4-v1"`
// rejection because that is a DSV4 artifact format, not an EXL3 one. A general
// EXL3 checkpoint (like MiMo-V2) carries `version: "1.5.1"` and
// `codebook: "mul1"` and is a valid EXL3 checkpoint.
#pragma once

#include <string>

#include <nlohmann/json.hpp>

#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// Returns a pointer to the `quantization_config` sub-object inside the
// checkpoint's config.json, or nullptr if absent or not an object.
// Exposed so model loaders that need to read quantization_config fields
// (version, codebook, bits, head_bits, …) share one access path.
inline const nlohmann::json* Exl3QuantConfig(const HfConfig& config) {
  const auto it = config.raw.find("quantization_config");
  if (it == config.raw.end() || it->is_null() || !it->is_object()) return nullptr;
  return &(*it);
}

// True when the checkpoint declares `quantization_config.quant_method == "exl3"`.
// This is the only model-agnostic EXL3 gate; version/codebook/bits are checked
// by the model-specific loader.
inline bool IsExl3Checkpoint(const HfConfig& config) {
  const nlohmann::json* qc = Exl3QuantConfig(config);
  if (qc == nullptr) return false;
  const auto it = qc->find("quant_method");
  if (it == qc->end() || !it->is_string()) return false;
  return it->get<std::string>() == "exl3";
}

}  // namespace vllm
