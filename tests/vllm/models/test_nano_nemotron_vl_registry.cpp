// NemotronH_Nano_Omni_Reasoning_V3 REACHABILITY: an image enters through the
// production entry points and changes the tokens.
//
// A tiny but structurally complete checkpoint is written to a temp file: the
// NemotronH hybrid language tower under `language_model.` (the same tiny
// schedule test_nemotron_h_paged_forward.cpp gates), a real-geometry
// `vit_small_patch16_224` RADIO tower, `mlp1`, and the tensors the loader must
// DEFER by name (the input conditioner, the video embedder, an audio tensor).
// It is loaded through `ModelRegistry::Load` under the released architecture
// name, the image is processed by `NanoNemotronVLPrepareInputs` (the function
// the chat seam calls), and a greedy decode runs through `GPUModelRunner`,
// which reaches `encode_mm`, `embed_mm` and the paged NemotronH forward.
//
// The reference is computed independently of the runner: the tower, the
// shuffle and the projector on a SECOND load of the same bytes, the merged
// embeddings spliced by hand, and the NemotronH HOST forward over them. The
// runner's tokens must equal the reference's, and must DIFFER from the tokens
// the same prompt gives with the image rows left as `<image>` embeddings.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "vllm/config/multimodal.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/nano_nemotron_vl.h"
#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/model_executor/models/nemotron_h_forward.h"
#include "vllm/model_executor/models/nemotron_h_loader.h"
#include "vllm/model_executor/models/radio.h"
#include "vllm/multimodal/inputs.h"
#include "vllm/multimodal/nano_nemotron_vl_processor.h"
#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using vllm::HfConfig;
using vllm::ModelRegistry;
using vllm::ModelSource;
using vllm::NemotronHBlock;
using vllm::SafetensorsFile;
using vllm::SamplingParams;
using vllm::v1::CachedRequestData;
using vllm::v1::GPUModelRunner;
using vllm::v1::KVCacheConfig;
using vllm::v1::NewRequestData;
using vllm::v1::SchedulerOutput;

namespace {

// The tiny language tower (test_nemotron_h_paged_forward.cpp's).
constexpr int kHidden = 24;
constexpr int kVocab = 32;
constexpr int kAttnHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 6;
constexpr int kMambaHeads = 4;
constexpr int kMambaHeadDim = 6;
constexpr int kNGroups = 2;
constexpr int kStateSize = 8;
constexpr int kConvKernel = 4;
constexpr int kChunkSize = 8;
constexpr int kRoutedExperts = 8;
constexpr int kExpertsPerTok = 3;
constexpr int kMoeInter = 10;
constexpr int kSharedInter = 12;
constexpr int kMambaInter = kMambaHeads * kMambaHeadDim;
constexpr int kConvDim = kMambaInter + 2 * kNGroups * kStateSize;
constexpr int kInProjOut = kMambaInter + kConvDim + kMambaHeads;
constexpr int kQDim = kAttnHeads * kHeadDim;
constexpr int kKvDim = kKvHeads * kHeadDim;
constexpr int kBlockSize = 16;
constexpr int kNumBlocks = 16;
constexpr int kMaxModelLen = 128;

// The vision side: vit_small_patch16_224 (configs/radio.py:13), CPE table
// 4x4 (cpe_max_size 64), 2 teachers -> 2 CLS + 2 registers.
constexpr int kVitH = 384, kVitL = 12, kVitI = 1536, kPatch = 16;
constexpr int kPosGrid = 4, kSkip = 4;
constexpr int kProjHidden = 32;

// The special tokens, inside the tiny vocabulary.
constexpr int32_t kImgContext = 10, kImgStart = 11, kImgEnd = 12;

struct Fx {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string bytes;
};

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[static_cast<size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

int64_t NumEl(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

float Synth(uint32_t& r, float scale) {
  r = r * 1664525u + 1013904223u;
  const float u = static_cast<float>(r >> 8) / static_cast<float>(1u << 24);
  return (u - 0.5f) * scale;
}

std::string Bf16Bytes(size_t n, int seed, float scale, float offset = 0.0f) {
  std::string s(n * 2, '\0');
  uint32_t r = static_cast<uint32_t>(seed) * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    const uint16_t bf = vt::F32ToBF16(offset + Synth(r, scale));
    s[i * 2] = static_cast<char>(bf & 0xff);
    s[i * 2 + 1] = static_cast<char>((bf >> 8) & 0xff);
  }
  return s;
}

std::string F32Bytes(size_t n, int seed, float scale) {
  std::string s(n * 4, '\0');
  uint32_t r = static_cast<uint32_t>(seed) * 2246822519u + 1u;
  for (size_t i = 0; i < n; ++i) {
    const float f = Synth(r, scale);
    std::memcpy(&s[i * 4], &f, 4);
  }
  return s;
}

Fx Bf16(const std::string& n, std::vector<int64_t> sh, int seed, float scale = 0.5f,
        float offset = 0.0f) {
  return {n, "BF16", sh, Bf16Bytes(static_cast<size_t>(NumEl(sh)), seed, scale, offset)};
}
Fx F32(const std::string& n, std::vector<int64_t> sh, int seed, float scale = 0.5f) {
  return {n, "F32", sh, F32Bytes(static_cast<size_t>(NumEl(sh)), seed, scale)};
}

std::string BuildSt(const std::vector<Fx>& ts) {
  nlohmann::json hdr = nlohmann::json::object();
  std::string data;
  for (const Fx& t : ts) {
    const size_t start = data.size();
    data += t.bytes;
    hdr[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {start, data.size()}}};
  }
  const std::string header = hdr.dump();
  return U64Le(header.size()) + header + data;
}

class TempFile {
 public:
  explicit TempFile(const std::string& bytes, const char* ext = ".safetensors") {
    static int c = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("nano_nemotron_vl_" + std::to_string(::getpid()) + "_" + std::to_string(c++) + ext))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

const std::vector<NemotronHBlock>& Schedule() {
  static const std::vector<NemotronHBlock> s{NemotronHBlock::kMamba, NemotronHBlock::kMoe,
                                             NemotronHBlock::kAttention, NemotronHBlock::kMamba,
                                             NemotronHBlock::kMoe};
  return s;
}

std::vector<Fx> BuildTensors(bool with_stray = false) {
  std::vector<Fx> v;
  int s = 1;
  const std::string lm = "language_model.";
  v.push_back(Bf16(lm + "backbone.embeddings.weight", {kVocab, kHidden}, s++));
  v.push_back(Bf16(lm + "backbone.norm_f.weight", {kHidden}, s++, 0.8f));
  v.push_back(Bf16(lm + "lm_head.weight", {kVocab, kHidden}, s++, 0.25f));
  for (size_t l = 0; l < Schedule().size(); ++l) {
    const std::string p = lm + "backbone.layers." + std::to_string(l) + ".";
    const std::string m = p + "mixer.";
    v.push_back(Bf16(p + "norm.weight", {kHidden}, s++, 0.9f));
    switch (Schedule()[l]) {
      case NemotronHBlock::kMamba:
        v.push_back(Bf16(m + "in_proj.weight", {kInProjOut, kHidden}, s++, 0.3f));
        v.push_back(Bf16(m + "out_proj.weight", {kHidden, kMambaInter}, s++, 0.3f));
        v.push_back(Bf16(m + "conv1d.weight", {kConvDim, 1, kConvKernel}, s++, 0.4f));
        v.push_back(Bf16(m + "conv1d.bias", {kConvDim}, s++, 0.2f));
        v.push_back(F32(m + "A_log", {kMambaHeads}, s++, 0.6f));
        v.push_back(F32(m + "D", {kMambaHeads}, s++, 0.6f));
        v.push_back(F32(m + "dt_bias", {kMambaHeads}, s++, 0.3f));
        v.push_back(Bf16(m + "norm.weight", {kMambaInter}, s++, 0.7f));
        break;
      case NemotronHBlock::kAttention:
        v.push_back(Bf16(m + "q_proj.weight", {kQDim, kHidden}, s++, 1.5f));
        v.push_back(Bf16(m + "k_proj.weight", {kKvDim, kHidden}, s++, 1.5f));
        v.push_back(Bf16(m + "v_proj.weight", {kKvDim, kHidden}, s++, 1.5f));
        v.push_back(Bf16(m + "o_proj.weight", {kHidden, kQDim}, s++, 0.3f));
        break;
      case NemotronHBlock::kMoe:
        v.push_back(F32(m + "gate.weight", {kRoutedExperts, kHidden}, s++, 0.35f));
        v.push_back(F32(m + "gate.e_score_correction_bias", {kRoutedExperts}, s++, 0.4f));
        for (int e = 0; e < kRoutedExperts; ++e) {
          const std::string ex = m + "experts." + std::to_string(e) + ".";
          v.push_back(Bf16(ex + "up_proj.weight", {kMoeInter, kHidden}, s++, 0.3f));
          v.push_back(Bf16(ex + "down_proj.weight", {kHidden, kMoeInter}, s++, 0.3f));
        }
        v.push_back(Bf16(m + "shared_experts.up_proj.weight", {kSharedInter, kHidden}, s++, 0.3f));
        v.push_back(Bf16(m + "shared_experts.down_proj.weight", {kHidden, kSharedInter}, s++, 0.3f));
        break;
      case NemotronHBlock::kMlp:
        break;
    }
  }
  const std::string pg = "vision_model.radio_model.model.patch_generator.";
  v.push_back(Bf16(pg + "embedder.weight", {kVitH, 3 * kPatch * kPatch}, s++, 0.08f));
  v.push_back(Bf16(pg + "pos_embed", {1, kPosGrid * kPosGrid, kVitH}, s++, 0.5f));
  v.push_back(Bf16(pg + "cls_token.token", {kSkip, kVitH}, s++, 0.5f));
  v.push_back(Bf16(pg + "video_embedder.weight", {kVitH, 2 * 3 * kPatch * kPatch}, s++, 0.1f));
  v.push_back(F32("vision_model.radio_model.input_conditioner.norm_mean", {3, 1, 1}, s++));
  v.push_back(F32("vision_model.radio_model.input_conditioner.norm_std", {3, 1, 1}, s++));
  for (int l = 0; l < kVitL; ++l) {
    const std::string b = "vision_model.radio_model.model.blocks." + std::to_string(l) + ".";
    v.push_back(Bf16(b + "norm1.weight", {kVitH}, s++, 0.2f, 1.0f));
    v.push_back(Bf16(b + "norm1.bias", {kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "attn.qkv.weight", {3 * kVitH, kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "attn.qkv.bias", {3 * kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "attn.proj.weight", {kVitH, kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "attn.proj.bias", {kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "norm2.weight", {kVitH}, s++, 0.2f, 1.0f));
    v.push_back(Bf16(b + "norm2.bias", {kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "mlp.fc1.weight", {kVitI, kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "mlp.fc1.bias", {kVitI}, s++, 0.1f));
    v.push_back(Bf16(b + "mlp.fc2.weight", {kVitH, kVitI}, s++, 0.05f));
    v.push_back(Bf16(b + "mlp.fc2.bias", {kVitH}, s++, 0.1f));
  }
  v.push_back(Bf16("mlp1.0.weight", {4 * kVitH}, s++, 0.2f, 1.0f));
  v.push_back(Bf16("mlp1.1.weight", {kProjHidden, 4 * kVitH}, s++, 0.1f));
  // Large enough that the image rows move the residual stream decisively.
  v.push_back(Bf16("mlp1.3.weight", {kHidden, kProjHidden}, s++, 4.0f));
  v.push_back(Bf16("sound_projection.linear.weight", {kHidden, 8}, s++));
  if (with_stray) v.push_back(Bf16("mystery_head.weight", {4}, s++));
  return v;
}

nlohmann::json TinyConfig() {
  nlohmann::json llm;
  llm["architectures"] = nlohmann::json::array({"NemotronHForCausalLM"});
  llm["model_type"] = "nemotron_h";
  llm["dtype"] = "bfloat16";
  nlohmann::json blocks = nlohmann::json::array();
  for (NemotronHBlock b : Schedule()) {
    blocks.push_back(b == NemotronHBlock::kMamba ? "mamba"
                     : b == NemotronHBlock::kMoe ? "moe"
                                                 : "attention");
  }
  llm["layers_block_type"] = blocks;
  llm["num_hidden_layers"] = static_cast<int>(Schedule().size());
  llm["hidden_size"] = kHidden;
  llm["vocab_size"] = kVocab;
  llm["max_position_embeddings"] = kMaxModelLen;
  llm["layer_norm_epsilon"] = 1e-5;
  llm["tie_word_embeddings"] = false;
  llm["num_attention_heads"] = kAttnHeads;
  llm["num_key_value_heads"] = kKvHeads;
  llm["head_dim"] = kHeadDim;
  llm["attention_bias"] = false;
  llm["mamba_num_heads"] = kMambaHeads;
  llm["mamba_head_dim"] = kMambaHeadDim;
  llm["n_groups"] = kNGroups;
  llm["ssm_state_size"] = kStateSize;
  llm["conv_kernel"] = kConvKernel;
  llm["chunk_size"] = kChunkSize;
  llm["expand"] = 2;
  llm["mamba_hidden_act"] = "silu";
  llm["mamba_ssm_cache_dtype"] = "float32";
  llm["use_conv_bias"] = true;
  llm["mamba_proj_bias"] = false;
  llm["n_routed_experts"] = kRoutedExperts;
  llm["num_experts_per_tok"] = kExpertsPerTok;
  llm["moe_intermediate_size"] = kMoeInter;
  llm["n_shared_experts"] = 1;
  llm["moe_shared_expert_intermediate_size"] = kSharedInter;
  llm["n_group"] = 1;
  llm["topk_group"] = 1;
  llm["routed_scaling_factor"] = 2.5;
  llm["norm_topk_prob"] = true;
  llm["mlp_hidden_act"] = "relu2";
  llm["mlp_bias"] = false;

  nlohmann::json vision;
  vision["patch_size"] = kPatch;
  vision["preferred_resolution"] = {32, 32};
  vision["video_temporal_patch_size"] = 2;
  vision["args"] = {{"model", "vit_small_patch16_224"},
                    {"min_num_patches", 4},
                    {"max_num_patches", 16},
                    {"register_multiple", 4},
                    {"cpe_max_size", 64},
                    {"cls_token_per_teacher", true},
                    {"teachers", nlohmann::json::array({{{"name", "a"}}, {{"name", "b"}}})}};

  nlohmann::json j;
  j["architectures"] = nlohmann::json::array({"NemotronH_Nano_Omni_Reasoning_V3"});
  j["model_type"] = "NemotronH_Nano_Omni_Reasoning_V3";
  j["torch_dtype"] = "bfloat16";
  j["downsample_ratio"] = 0.5;
  j["ps_version"] = "v2";
  j["patch_size"] = kPatch;
  j["vit_hidden_size"] = kVitH;
  j["projector_hidden_size"] = kProjHidden;
  j["norm_mean"] = {0.48145466, 0.4578275, 0.40821073};
  j["norm_std"] = {0.26862954, 0.26130258, 0.27577711};
  j["llm_config"] = llm;
  j["vision_config"] = vision;
  j["sound_config"] = nlohmann::json::object();
  return j;
}

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

struct Fixture {
  std::unique_ptr<TempFile> st;
  std::unique_ptr<TempFile> cfg_json;
  std::vector<SafetensorsFile> shards;
  HfConfig cfg;
  std::unique_ptr<vllm::LoadedModel> model;

  Fixture() {
    st = std::make_unique<TempFile>(BuildSt(BuildTensors()));
    cfg_json = std::make_unique<TempFile>(TinyConfig().dump(2), ".json");
    shards.push_back(SafetensorsFile::Open(st->path()));
    cfg = vllm::LoadHfConfig(cfg_json->path());
    model = ModelRegistry::Load(cfg, ModelSource::FromSafetensors(shards));
  }
};

std::vector<uint8_t> Image(int64_t h, int64_t w) {
  std::vector<uint8_t> px(static_cast<size_t>(h * w * 3));
  uint32_t r = 77;
  for (auto& v : px) {
    r = r * 1664525u + 1013904223u;
    v = static_cast<uint8_t>(r >> 24);
  }
  return px;
}

vllm::multimodal::MultiModalInputs Prepare(const Fixture& fx, const std::vector<uint8_t>& rgb,
                                           int64_t h, int64_t w,
                                           const std::vector<int32_t>& prompt) {
  vllm::NanoNemotronVLParams p = vllm::ParseNanoNemotronVLParams(fx.cfg);
  p.processor.max_model_len = kMaxModelLen;
  vllm::multimodal::NanoNemotronVLTokenIds ids{kImgStart, kImgContext, kImgEnd};
  const int64_t text_len = static_cast<int64_t>(prompt.size()) - 1;
  return vllm::multimodal::NanoNemotronVLPrepareInputs(prompt, text_len,
                                                       {{rgb.data(), h, w}}, p.processor, ids,
                                                       "tiny-omni");
}

SamplingParams Greedy() {
  SamplingParams sp;
  sp.temperature = 0.0;
  sp.PostInit();
  return sp;
}

// Prefill + decode through the runner; the image item is encoded on the first
// step because the scheduler names it in `scheduled_encoder_inputs`.
std::vector<int32_t> RunnerGreedy(Fixture& fx, const vllm::multimodal::MultiModalInputs& mm,
                                  int steps) {
  const vllm::ModelRegistration& reg = fx.model->registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(fx.cfg, kBlockSize, kNumBlocks);
  vt::Queue q = Q();
  GPUModelRunner runner(fx.cfg, *fx.model, kv, q, /*max_num_reqs=*/2, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);
  const std::string id = "img";
  NewRequestData nr;
  nr.req_id = id;
  nr.prompt_token_ids = mm.prompt_token_ids;
  nr.sampling_params = Greedy();
  nr.block_ids = {std::vector<int>{0, 1, 2}, std::vector<int>{0}};
  nr.num_computed_tokens = 0;
  nr.prefill_token_ids = mm.prompt_token_ids;
  nr.mm_features = mm.mm_features;
  SchedulerOutput so;
  so.scheduled_cached_reqs = CachedRequestData::make_empty();
  so.scheduled_new_reqs.push_back(nr);
  so.num_scheduled_tokens[id] = static_cast<int>(mm.prompt_token_ids.size());
  so.total_num_scheduled_tokens = static_cast<int>(mm.prompt_token_ids.size());
  if (!mm.mm_features.empty()) so.scheduled_encoder_inputs[id] = {0};
  CHECK_FALSE(runner.execute_model(so).has_value());
  vllm::v1::ModelRunnerOutput m1 = runner.sample_tokens(std::nullopt);
  REQUIRE(m1.sampled_token_ids.size() == 1);
  std::vector<int32_t> out{m1.sampled_token_ids[0][0]};
  int computed = static_cast<int>(mm.prompt_token_ids.size());
  int outputs = 1;
  for (int s = 1; s < steps; ++s) {
    SchedulerOutput sd;
    CachedRequestData cached;
    cached.req_ids = {id};
    cached.num_computed_tokens.push_back(computed);
    cached.num_output_tokens.push_back(outputs);
    cached.new_block_ids.emplace_back(std::nullopt);
    sd.scheduled_cached_reqs = std::move(cached);
    sd.num_scheduled_tokens[id] = 1;
    sd.total_num_scheduled_tokens = 1;
    CHECK_FALSE(runner.execute_model(sd).has_value());
    vllm::v1::ModelRunnerOutput md = runner.sample_tokens(std::nullopt);
    REQUIRE(md.sampled_token_ids.size() == 1);
    out.push_back(md.sampled_token_ids[0][0]);
    ++computed;
    ++outputs;
  }
  return out;
}

int32_t Argmax(const std::vector<float>& logits, size_t row, size_t vocab) {
  const float* r = logits.data() + row * vocab;
  return static_cast<int32_t>(std::max_element(r, r + vocab) - r);
}

}  // namespace

TEST_CASE("nano-nemotron-vl: the released architecture names resolve through the registry") {
  const auto archs = ModelRegistry::SupportedArchs();
  CHECK(std::find(archs.begin(), archs.end(), "NemotronH_Nano_Omni_Reasoning_V3") != archs.end());
  CHECK(std::find(archs.begin(), archs.end(), "NemotronH_Nano_VL_V2") != archs.end());
}

TEST_CASE("nano-nemotron-vl: the loader accounts for every shipped tensor") {
  Fixture fx;
  REQUIRE(fx.model != nullptr);
  // A stray tensor nothing names is REFUSED, never dropped.
  TempFile stray(BuildSt(BuildTensors(/*with_stray=*/true)));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(stray.path()));
  bool refused = false;
  try {
    (void)ModelRegistry::Load(fx.cfg, ModelSource::FromSafetensors(shards));
  } catch (const std::runtime_error& e) {
    refused = std::string(e.what()).find("mystery_head.weight") != std::string::npos;
  }
  CHECK(refused);
}

TEST_CASE("nano-nemotron-vl: an image reaches encode_mm, embed_mm and the paged forward") {
  Fixture fx;
  const int64_t h = 40, w = 56;
  const std::vector<uint8_t> rgb = Image(h, w);
  const std::vector<int32_t> prompt = {3, 7, kImgContext, 5, 9};
  const vllm::multimodal::MultiModalInputs mm = Prepare(fx, rgb, h, w, prompt);
  REQUIRE(mm.mm_features.size() == 1);
  const auto& feat = mm.mm_features[0];
  // 40x56 -> a 4x4 patch grid -> 4 rows after the shuffle.
  CHECK(feat.data->image_grid_thw[1] == 4);
  CHECK(feat.data->image_grid_thw[2] == 4);
  REQUIRE(feat.length == 4);
  CHECK(mm.prompt_token_ids ==
        std::vector<int32_t>{3, 7, kImgStart, kImgContext, kImgContext, kImgContext,
                             kImgContext, kImgEnd, 5, 9});
  CHECK(feat.offset == 3);

  constexpr int kSteps = 4;
  const std::vector<int32_t> got = RunnerGreedy(fx, mm, kSteps);

  // ── the independent reference ──
  const vllm::NanoNemotronVLParams p = vllm::ParseNanoNemotronVLParams(fx.cfg);
  const vllm::NanoNemotronVLVisionLoad vis = vllm::LoadNanoNemotronVLVisionWeights(fx.shards, p);
  vt::Backend* cpu = vt::TryGetBackend(vt::DeviceType::kCPU);
  REQUIRE(cpu != nullptr);
  const vllm::multimodal::RadioVisionTower tower(vis.radio, p.radio, *cpu);
  const vllm::multimodal::NanoNemotronVLProjector proj(vis.projector, p.projector, *cpu);
  vllm::multimodal::RadioImage im;
  im.patches = feat.data->pixel_values_f32;
  im.grid_h = 4;
  im.grid_w = 4;
  std::vector<float> f = tower.Forward({im});
  for (float& v : f) v = vt::BF16ToF32(vt::F32ToBF16(v));
  const std::vector<float> rows =
      proj.Forward(vllm::multimodal::NanoNemotronVLPixelShuffle(f, 4, 4, kVitH, 2), 4);

  const HfConfig text_cfg = vllm::NanoNemotronVLTextConfig(fx.cfg);
  const vllm::NemotronHParams tp = vllm::ParseNemotronHParams(text_cfg);
  vllm::NemotronHLoadReport rep;
  const vllm::NemotronHHostWeights host = vllm::LoadNemotronHHostWeights(
      fx.shards, tp, vllm::ResolveNemotronHModelDType(text_cfg), &rep, "language_model.");
  CHECK(rep.materialized == rep.in_index);

  auto embed_row = [&](int32_t id) {
    std::vector<float> r(kHidden);
    const auto* tab = reinterpret_cast<const uint16_t*>(host.embeddings.bytes.data());
    for (int c = 0; c < kHidden; ++c) r[static_cast<size_t>(c)] = vt::BF16ToF32(tab[id * kHidden + c]);
    return r;
  };
  std::vector<int32_t> seq = mm.prompt_token_ids;
  std::vector<float> embeds;
  for (size_t t = 0; t < seq.size(); ++t) {
    const int64_t k = static_cast<int64_t>(t) - feat.offset;
    if (k >= 0 && k < feat.length) {
      for (int c = 0; c < kHidden; ++c)
        embeds.push_back(vt::BF16ToF32(vt::F32ToBF16(rows[static_cast<size_t>(k * kHidden + c)])));
    } else {
      const auto r = embed_row(seq[t]);
      embeds.insert(embeds.end(), r.begin(), r.end());
    }
  }
  vt::Queue q = Q();
  std::vector<int32_t> want;
  for (int s = 0; s < kSteps; ++s) {
    const auto logits = vllm::NemotronHForward(host, tp, seq, {}, q, nullptr, &embeds);
    const int32_t tok = Argmax(logits, seq.size() - 1, kVocab);
    want.push_back(tok);
    seq.push_back(tok);
    const auto r = embed_row(tok);
    embeds.insert(embeds.end(), r.begin(), r.end());
  }
  CHECK(got == want);

  // The image MATTERS: the same ids with the `<image>` rows embedded as text
  // decode differently. A forward that ignored inputs_embeds would pass the
  // comparison above only if this one failed.
  const std::vector<float> prompt_embeds(
      embeds.begin(),
      embeds.begin() + static_cast<std::ptrdiff_t>(mm.prompt_token_ids.size() * kHidden));
  const auto text_logits = vllm::NemotronHForward(host, tp, mm.prompt_token_ids, {}, q);
  const auto img_logits =
      vllm::NemotronHForward(host, tp, mm.prompt_token_ids, {}, q, nullptr, &prompt_embeds);
  double diff = 0;
  const size_t last = (mm.prompt_token_ids.size() - 1) * kVocab;
  for (int v = 0; v < kVocab; ++v)
    diff = std::max(diff, std::abs(static_cast<double>(text_logits[last + v]) - img_logits[last + v]));
  MESSAGE("max |logit(image) - logit(text-only)| at the last prompt position: " << diff);
  CHECK(diff > 1e-2);
  // And the decoded TOKENS differ, so the runner equality above cannot be met
  // by a runner that dropped the image rows.
  std::vector<int32_t> text_seq = mm.prompt_token_ids;
  std::vector<int32_t> want_text;
  for (int s = 0; s < kSteps; ++s) {
    const auto logits = vllm::NemotronHForward(host, tp, text_seq, {}, q);
    const int32_t tok = Argmax(logits, text_seq.size() - 1, kVocab);
    want_text.push_back(tok);
    text_seq.push_back(tok);
  }
  MESSAGE("image decode " << want[0] << " " << want[1] << " " << want[2] << " " << want[3]
                          << " | text-only decode " << want_text[0] << " " << want_text[1]
                          << " " << want_text[2] << " " << want_text[3]);
  CHECK(want != want_text);
}

// ─── upstream tests/models/multimodal/test_nano_nemotron_vl.py @ e126687a9a ──
// The three `load_weights` cases, adapted from a mocked module to a real load:
// the harness here has no module tree to monkeypatch, so each case loads a
// tiny checkpoint and asserts the same observable outcome. The two
// `_extract_audio_from_videos` cases are NOT ported: audio is refused by name.

// test_nano_nemotron_vl_skips_multimodal_weights_in_text_only_mode (:77-96)
TEST_CASE("nano-nemotron-vl (upstream): text-only mode skips the multimodal weights") {
  Fixture fx;
  vllm::MultiModalConfig text_only;
  text_only.language_model_only = true;
  ModelSource src = ModelSource::FromSafetensors(fx.shards);
  src.multimodal = &text_only;
  auto model = ModelRegistry::Load(fx.cfg, src);
  REQUIRE(model != nullptr);
  vllm::multimodal::MultiModalFeatureSpec item;
  item.modality = "image";
  item.length = 4;
  item.data = std::make_shared<vllm::multimodal::ImageKwargs>();
  item.data->num_patches = 16;
  vt::Queue q = Q();
  bool refused = false;
  try {
    (void)ModelRegistry::EncodeMm(*model, fx.cfg, q, item);
  } catch (const std::exception& e) {
    refused = std::string(e.what()).find("loaded text-only") != std::string::npos;
  }
  CHECK(refused);
}

// test_nano_nemotron_vl_loads_vision_weights_without_sound_encoder (:99-121)
// and test_nano_nemotron_vl_requires_sound_encoder_for_sound_weights (:124-136)
TEST_CASE("nano-nemotron-vl (upstream): vision loads without a sound encoder; sound weights need one") {
  nlohmann::json no_sound = TinyConfig();
  no_sound.erase("sound_config");
  TempFile cfg_json(no_sound.dump(2), ".json");
  const HfConfig cfg = vllm::LoadHfConfig(cfg_json.path());
  // Without the sound tensor: loads.
  std::vector<Fx> ts = BuildTensors();
  ts.erase(std::remove_if(ts.begin(), ts.end(),
                          [](const Fx& f) { return f.name.rfind("sound", 0) == 0; }),
           ts.end());
  {
    TempFile st(BuildSt(ts));
    std::vector<SafetensorsFile> shards;
    shards.push_back(SafetensorsFile::Open(st.path()));
    CHECK(ModelRegistry::Load(cfg, ModelSource::FromSafetensors(shards)) != nullptr);
  }
  // With it: refused, because no sound encoder exists to take it.
  {
    TempFile st(BuildSt(BuildTensors()));
    std::vector<SafetensorsFile> shards;
    shards.push_back(SafetensorsFile::Open(st.path()));
    bool refused = false;
    try {
      (void)ModelRegistry::Load(cfg, ModelSource::FromSafetensors(shards));
    } catch (const std::runtime_error& e) {
      refused = std::string(e.what()).find("no `sound_config`") != std::string::npos;
    }
    CHECK(refused);
  }
}

TEST_CASE("nano-nemotron-vl: audio and video items are refused by name at encode") {
  Fixture fx;
  vllm::multimodal::MultiModalFeatureSpec item;
  item.modality = "audio";
  item.length = 4;
  vt::Queue q = Q();
  bool refused = false;
  try {
    (void)ModelRegistry::EncodeMm(*fx.model, fx.cfg, q, item);
  } catch (const std::exception& e) {
    refused = std::string(e.what()).find("modality 'audio' is not ported") != std::string::npos;
  }
  CHECK(refused);
}

// ─── STRUCTURAL gate on a released index (env-gated, headers only) ──────────
// NANO_NEMOTRON_VL_INDEX_JSON names a model.safetensors.index.json and
// NANO_NEMOTRON_VL_CONFIG_JSON the config.json beside it. Every tensor the index
// ships must be claimed: the language tower by NemotronH's own enumeration
// under `language_model.`, the vision tower and mlp1 by name, and the rest
// deferred by name (input conditioner, video embedder, audio). The verdict is
// printed for each arm so a run records which checkpoint layout this tree can
// load and which it refuses.
TEST_CASE("nano-nemotron-vl STRUCTURAL: a released index is fully accounted") {
  const char* index_path = std::getenv("NANO_NEMOTRON_VL_INDEX_JSON");
  const char* config_path = std::getenv("NANO_NEMOTRON_VL_CONFIG_JSON");
  if (index_path == nullptr || config_path == nullptr) {
    MESSAGE("SKIP: NANO_NEMOTRON_VL_INDEX_JSON / NANO_NEMOTRON_VL_CONFIG_JSON unset");
    return;
  }
  nlohmann::json index;
  {
    std::ifstream f(index_path);
    REQUIRE(f.good());
    f >> index;
  }
  const HfConfig cfg = vllm::LoadHfConfig(config_path);
  const vllm::NanoNemotronVLParams p = vllm::ParseNanoNemotronVLParams(cfg);
  const HfConfig text_cfg = vllm::NanoNemotronVLTextConfig(cfg);
  const vllm::NemotronHParams tp = vllm::ParseNemotronHParams(text_cfg);
  std::set<std::string> enumerated;
  for (const auto& t : vllm::EnumerateNemotronHTensors(tp)) enumerated.insert(t.name);

  std::set<std::string> vision_expected;
  const std::string pg = "vision_model.radio_model.model.patch_generator.";
  for (const char* n : {"embedder.weight", "pos_embed", "cls_token.token"}) vision_expected.insert(pg + n);
  for (int64_t l = 0; l < p.radio.num_hidden_layers; ++l) {
    const std::string b = "vision_model.radio_model.model.blocks." + std::to_string(l) + ".";
    for (const char* n : {"norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias",
                          "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias",
                          "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"})
      vision_expected.insert(b + n);
  }
  for (const char* n : {"mlp1.0.weight", "mlp1.1.weight", "mlp1.3.weight"}) vision_expected.insert(n);

  int64_t lm = 0, lm_claimed = 0, vis = 0, deferred = 0;
  std::vector<std::string> unclaimed;
  std::set<std::string> lm_seen;
  for (const auto& [name, shard] : index.at("weight_map").items()) {
    (void)shard;
    if (name.rfind("language_model.", 0) == 0) {
      ++lm;
      const std::string stripped = name.substr(std::string("language_model.").size());
      lm_seen.insert(stripped);
      if (enumerated.count(stripped) != 0) ++lm_claimed; else unclaimed.push_back(name);
    } else if (vision_expected.count(name) != 0) {
      ++vis;
    } else if (name == pg + "video_embedder.weight" ||
               name.rfind("vision_model.radio_model.input_conditioner.", 0) == 0 ||
               name.rfind("sound_encoder.", 0) == 0 || name.rfind("sound_projection.", 0) == 0) {
      ++deferred;
    } else {
      unclaimed.push_back(name);
    }
  }
  std::vector<std::string> missing;
  for (const std::string& n : enumerated)
    if (lm_seen.count(n) == 0 && n.rfind("mtp.", 0) != 0) missing.push_back(n);
  MESSAGE("language tower: " << lm << " shipped, " << lm_claimed << " claimed by the NemotronH "
                             << "enumeration, " << missing.size() << " enumerated but not shipped");
  MESSAGE("vision + mlp1: " << vis << " of " << vision_expected.size() << " claimed; " << deferred
                            << " deferred by name; " << unclaimed.size() << " unclaimed");
  for (size_t i = 0; i < std::min<size_t>(unclaimed.size(), 5); ++i)
    MESSAGE("  unclaimed: " << unclaimed[i]);
  for (size_t i = 0; i < std::min<size_t>(missing.size(), 5); ++i)
    MESSAGE("  enumerated but not shipped: " << missing[i]);
  CHECK(vis == static_cast<int64_t>(vision_expected.size()));
  CHECK(unclaimed.empty());
  CHECK(missing.empty());
}
