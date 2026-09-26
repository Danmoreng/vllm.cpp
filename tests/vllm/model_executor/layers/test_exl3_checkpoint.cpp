// IsExl3Checkpoint — the shared, model-agnostic EXL3 checkpoint predicate.
//
// Before QUANT-EXL3-GENERALISE W1, `IsExl3Checkpoint` lived in an anonymous
// namespace inside `deepseek_v4_weights.cpp` — no other model loader could
// call it. This suite gates the shared header `exl3_checkpoint.h`: DSV4 and
// MiMoV2 EXL3 checkpoints both return true; non-EXL3 checkpoints return false.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/quantization/exl3_checkpoint.h"
#include "vllm/transformers_utils/hf_config.h"

namespace {

vllm::HfConfig MakeConfig(const std::string& json_str) {
  vllm::HfConfig c;
  c.raw = nlohmann::json::parse(json_str);
  return c;
}

}  // namespace

TEST_CASE("IsExl3Checkpoint: a DeepSeek-V4 EXL3 config is detected") {
  // Mirrors the DSV4 EXL3 checkpoint: version "rank-sliced-deepseek-v4-v1",
  // codebook "mcg", whole-number bits.
  const auto c = MakeConfig(R"({
    "model_type": "deepseek_v4",
    "architectures": ["DeepseekV4ForCausalLM"],
    "quantization_config": {
      "quant_method": "exl3",
      "version": "rank-sliced-deepseek-v4-v1",
      "codebook": "mcg",
      "bits": 3.0
    }
  })");
  CHECK(vllm::IsExl3Checkpoint(c));
}

TEST_CASE("IsExl3Checkpoint: a MiMoV2 EXL3 config is detected") {
  // Mirrors the MiMo-V2.6-Flash-RL-EXL3 checkpoint: version "1.5.1",
  // codebook "mul1", fractional bits 2.2, head_bits 6.
  const auto c = MakeConfig(R"({
    "model_type": "mimo_v2",
    "architectures": ["MiMoV2ForCausalLM"],
    "quantization_config": {
      "quant_method": "exl3",
      "version": "1.5.1",
      "codebook": "mul1",
      "bits": 2.2,
      "head_bits": 6
    }
  })");
  CHECK(vllm::IsExl3Checkpoint(c));
}

TEST_CASE("IsExl3Checkpoint: a config without quantization_config returns false") {
  const auto c = MakeConfig(R"({
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096
  })");
  CHECK_FALSE(vllm::IsExl3Checkpoint(c));
}

TEST_CASE("IsExl3Checkpoint: a non-exl3 quant_method returns false") {
  const auto c = MakeConfig(R"({
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "quantization_config": {
      "quant_method": "fp8",
      "activation_scheme": "dynamic"
    }
  })");
  CHECK_FALSE(vllm::IsExl3Checkpoint(c));
}

TEST_CASE("IsExl3Checkpoint: quantization_config present but no quant_method returns false") {
  const auto c = MakeConfig(R"({
    "model_type": "llama",
    "quantization_config": {
      "bits": 4
    }
  })");
  CHECK_FALSE(vllm::IsExl3Checkpoint(c));
}

TEST_CASE("IsExl3Checkpoint: quantization_config: null returns false") {
  const auto c = MakeConfig(R"({
    "model_type": "llama",
    "quantization_config": null
  })");
  CHECK_FALSE(vllm::IsExl3Checkpoint(c));
}

TEST_CASE("Exl3QuantConfig: returns the quantization_config sub-object") {
  const auto c = MakeConfig(R"({
    "model_type": "mimo_v2",
    "quantization_config": {
      "quant_method": "exl3",
      "codebook": "mul1"
    }
  })");
  const nlohmann::json* qc = vllm::Exl3QuantConfig(c);
  REQUIRE(qc != nullptr);
  CHECK(qc->at("quant_method") == "exl3");
  CHECK(qc->at("codebook") == "mul1");
}

TEST_CASE("Exl3QuantConfig: returns nullptr when absent") {
  const auto c = MakeConfig(R"({"model_type":"llama"})");
  CHECK(vllm::Exl3QuantConfig(c) == nullptr);
}
