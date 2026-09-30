// MODEL-NIMBLE: the Nimble decision lane (.agents/specs/nimble.md).
//
// Three kinds of evidence, in order of how much they rest on the reference:
//   1. nimble_goldens.inc is produced by RUNNING the model author's prompt code
//      (parallel_schema.py via compiler.py) and openjev's scoring.answer
//      (scripts/gen-nimble-goldens.py). The prompt and answer cases compare
//      against it byte for byte and to 1e-12.
//   2. The request refusals are the reference's own error cases.
//   3. NimbleDecide, the seam /v1/systemone and vllm_decide both call, runs
//      over a synthetic Qwen3.5 dense model and a byte-level tokenizer, and its
//      answers must equal an INDEPENDENT recomputation from ForwardDense's
//      full-logits last row at the candidate ids.
// With VLLM_CPP_NIMBLE_TOKENIZER_DIR pointing at the checkpoint's tokenizer,
// the prompts must also tokenize to the reference's exact ids.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "nimble_goldens.inc"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/nimble_inference.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace {

using ojson = nlohmann::ordered_json;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5DenseWeights;
using vllm::tok::Tokenizer;
using vt::DType;

const ojson& Goldens() {
  static const ojson g = ojson::parse(kNimbleGoldens);
  return g;
}

// ── A byte-level tokenizer with the four Qwen chat tokens ──────────────────
// 256 byte tokens (id = byte value) and no merges, so every character is one
// token and "A" is id 65. The chat tokens are added tokens, as in Qwen3.5.

std::string JsonEscape(std::string_view s) {
  std::string out;
  for (unsigned char c : s) {
    if (c == '\\') {
      out += "\\\\";
    } else if (c == '"') {
      out += "\\\"";
    } else if (c < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else {
      out += static_cast<char>(c);
    }
  }
  return out;
}

std::string CodepointToUtf8(uint32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return out;
}

const Tokenizer& ByteTokenizer() {
  static const Tokenizer tok = [] {
    std::string vocab;
    for (int b = 0; b < 256; ++b) {
      if (b > 0) vocab += ',';
      vocab += "\"" +
               JsonEscape(CodepointToUtf8(
                   vllm::tok::ByteToUnicode(static_cast<uint8_t>(b)))) +
               "\":" + std::to_string(b);
    }
    const char* added[4][2] = {{"<|im_start|>", "true"},
                               {"<|im_end|>", "true"},
                               {"<think>", "false"},
                               {"</think>", "false"}};
    std::string added_json;
    for (int i = 0; i < 4; ++i) {
      vocab += ",\"" + std::string(added[i][0]) + "\":" + std::to_string(256 + i);
      if (i > 0) added_json += ',';
      added_json += "{\"id\":" + std::to_string(256 + i) + ",\"content\":\"" +
                    added[i][0] +
                    "\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,"
                    "\"normalized\":false,\"special\":" +
                    added[i][1] + "}";
    }
    std::string json = R"({"version":"1.0","truncation":null,"padding":null,)";
    json += "\"added_tokens\":[" + added_json + "],";
    json += R"("normalizer":null,)";
    // The Qwen split regex the loader requires; with no merges it does not
    // change the ids, every byte is still one token.
    const std::string regex =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|)"
        R"([^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|)"
        R"( ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|)"
        R"(\s*[\r\n]+|\s+(?!\S)|\s+)";
    json += R"("pre_tokenizer":{"type":"Sequence","pretokenizers":[)";
    json += "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"" + JsonEscape(regex) +
            "\"},\"behavior\":\"Isolated\",\"invert\":false},";
    json += R"({"type":"ByteLevel","add_prefix_space":false,)"
            R"("trim_offsets":false,"use_regex":false}]},)";
    json += R"("post_processor":{"type":"ByteLevel","add_prefix_space":false,)"
            R"("trim_offsets":false,"use_regex":false},)";
    json += R"("decoder":{"type":"ByteLevel","add_prefix_space":false,)"
            R"("trim_offsets":false,"use_regex":false},)";
    json += R"("model":{"type":"BPE","dropout":null,"unk_token":null,)"
            R"("continuing_subword_prefix":null,"end_of_word_suffix":null,)"
            R"("fuse_unk":false,"byte_fallback":false,"ignore_merges":false,)";
    json += "\"vocab\":{" + vocab + "},\"merges\":[]}}";
    return Tokenizer::FromHfJsonBytes(json, "nimble_test");
  }();
  return tok;
}

// ── A small synthetic Qwen3.5 dense model (GDN hybrid) ─────────────────────
// Same generator and shapes as test_qwen3_5_dense_vision.cpp, with the vocab
// widened to the byte tokenizer's 260 ids.

uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}

OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"NimbleModel"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;  // [LA, LA, LA, FA]
  c.vocab_size = 260;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.intermediate_size = 16;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 4;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 4096;
  c.rope_parameters.mrope_interleaved = true;
  c.rope_parameters.mrope_section = {1, 1, 0};
  return c;
}

Qwen3_5DenseWeights MakeWeights(const HfConfig& c) {
  Qwen3_5DenseWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size, I = c.intermediate_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3_5DenseLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.mlp.gate_proj = MakeOwned(DType::kBF16, {H, I}, s + 500);
    lw.mlp.up_proj = MakeOwned(DType::kBF16, {H, I}, s + 600);
    lw.mlp.down_proj = MakeOwned(DType::kBF16, {I, H}, s + 700);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

std::string RefusalOf(const ojson& body) {
  try {
    (void)vllm::nimble::CompileRequest(body);
  } catch (const vllm::nimble::RequestError& e) {
    return e.what();
  }
  return "";
}

ojson Body(const char* text) { return ojson::parse(text); }

}  // namespace

// ── 1. Reference goldens ────────────────────────────────────────────────────

TEST_CASE("nimble.prompts.match_the_reference_prepare_prompts") {
  for (const ojson& c : Goldens()["cases"]) {
    const vllm::nimble::Request r = vllm::nimble::CompileRequest(c["body"]);
    const std::vector<std::string> prompts = vllm::nimble::BuildPrompts(r);
    REQUIRE(prompts.size() == c["prompts"].size());
    for (size_t i = 0; i < prompts.size(); ++i) {
      CHECK(prompts[i] == c["prompts"][i].get<std::string>());
    }
  }
}

TEST_CASE("nimble.answers.match_openjev_scoring") {
  for (const ojson& c : Goldens()["cases"]) {
    const vllm::nimble::Request r = vllm::nimble::CompileRequest(c["body"]);
    for (const ojson& a : c["answers"]) {
      const std::string name = a["field"].get<std::string>();
      const vllm::nimble::Field* field = nullptr;
      for (const auto& f : r.fields)
        if (f.name == name) field = &f;
      REQUIRE(field != nullptr);
      const ojson got = vllm::nimble::AnswerFromLogits(
          *field, a["logits"].get<std::vector<double>>(),
          a["temperature"].get<double>());
      const ojson& want = a["expected"];
      INFO(name << " T=" << a["temperature"].get<double>());
      // Same keys in the same order as the pydantic model.
      std::vector<std::string> got_keys, want_keys;
      for (auto it = got.begin(); it != got.end(); ++it) got_keys.push_back(it.key());
      for (auto it = want.begin(); it != want.end(); ++it) want_keys.push_back(it.key());
      CHECK(got_keys == want_keys);
      for (auto it = want.begin(); it != want.end(); ++it) {
        const ojson& w = it.value();
        const ojson& g = got[it.key()];
        if (w.is_number_float()) {
          CHECK(g.get<double>() == doctest::Approx(w.get<double>()).epsilon(1e-12));
        } else if (w.is_object() && !w.empty() && w.begin()->is_number()) {
          for (auto p = w.begin(); p != w.end(); ++p) {
            CHECK(g[p.key()].get<double>() ==
                  doctest::Approx(p.value().get<double>()).epsilon(1e-12));
          }
        } else {
          CHECK(g == w);
        }
      }
    }
  }
}

// ── 2. The reference's refusals ─────────────────────────────────────────────

TEST_CASE("nimble.request.refuses_what_the_reference_refuses") {
  CHECK(RefusalOf(Body(R"({"questions":{"a":{"type":"noul","instructions":"x"}}})"))
            .find("state") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"  ","questions":{"a":{"type":"noul","instructions":"x"}}})"))
            .find("Context must be a nonempty string") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{}})")).find("questions") !=
        std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{"a":{"type":"rank","instructions":"x"}}})"))
            .find("unknown type") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{"a":{"type":"noul","instructions":" "}}})"))
            .find("nonempty description") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{"a":{"type":"noul","instructions":"x","extra":1}}})"))
            .find("unsupported key 'extra'") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{"a":{"type":"noul","instructions":"x","criteria":{"maybe":"m"}}}})"))
            .find("unsupported key 'maybe'") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{"a":{"type":"choice","instructions":"x","criteria":{"only":null}}}})"))
            .find("2-64") != std::string::npos);
  CHECK(RefusalOf(Body(R"({"state":"s","questions":{"a":{"type":"score","instructions":"x","criteria":["low",2]}}})"))
            .find("strings") != std::string::npos);
  // 27 choices: the reference switches prompt contracts; this engine refuses
  // by name rather than approximating the extended arm.
  ojson wide = Body(R"({"state":"s","questions":{"a":{"type":"choice","instructions":"x","criteria":{}}}})");
  for (int i = 0; i < 27; ++i) wide["questions"]["a"]["criteria"]["k" + std::to_string(i)] = nullptr;
  const std::string msg = RefusalOf(wide);
  CHECK(msg.find("27 choices") != std::string::npos);
  CHECK(msg.find("extended_schema.py") != std::string::npos);
  // 26 is the widest accepted field.
  wide["questions"]["a"]["criteria"].erase("k26");
  CHECK(RefusalOf(wide).empty());
}

// ── 3. Reachability through NimbleDecide ────────────────────────────────────

TEST_CASE("nimble.decide.reads_the_last_row_candidate_logits_per_field") {
  const HfConfig config = MakeConfig();
  const double temperature = 2.179078721266035;
  auto model = vllm::MakeNimbleLoadedModel(MakeWeights(config), config,
                                           temperature, 8192, Q());
  const Tokenizer& tok = ByteTokenizer();
  const ojson body = Goldens()["cases"][0]["body"];

  const vllm::NimbleResponse got = vllm::NimbleDecide(*model, tok, body);

  // Independent recomputation: the FULL-logits ForwardDense, its last row, the
  // byte ids of "A", "B", "C" (65, 66, 67), then the openjev answer.
  const vllm::nimble::Request req = vllm::nimble::CompileRequest(body);
  const std::vector<std::string> prompts = vllm::nimble::BuildPrompts(req);
  const Qwen3_5DenseWeights weights = MakeWeights(config);
  vt::Queue q = Q();
  int64_t tokens = 0;
  std::vector<double> first_logits;
  REQUIRE(got.answers.size() == req.fields.size());
  for (size_t i = 0; i < req.fields.size(); ++i) {
    const auto& f = req.fields[i];
    const std::vector<int32_t> ids = tok.Encode(prompts[i]);
    tokens += static_cast<int64_t>(ids.size());
    std::vector<int32_t> pos(ids.size());
    std::iota(pos.begin(), pos.end(), 0);
    const std::vector<float> all = vllm::Qwen3_5DenseModel::ForwardDense(
        ids, pos, weights, config, q);
    const size_t last = (ids.size() - 1) * static_cast<size_t>(config.vocab_size);
    std::vector<double> cand;
    for (size_t k = 0; k < f.keys.size(); ++k) cand.push_back(all[last + 65 + k]);
    if (i == 0) first_logits = cand;
    const ojson want = vllm::nimble::AnswerFromLogits(f, cand, temperature);
    const ojson& g = got.answers[f.name];
    INFO(f.name);
    REQUIRE(g["type"] == want["type"]);
    if (f.type == "noul") {
      CHECK(g["noul"].get<double>() == doctest::Approx(want["noul"].get<double>()).epsilon(1e-5));
    } else {
      for (const auto& key : f.keys) {
        CHECK(g["probabilities"][key].get<double>() ==
              doctest::Approx(want["probabilities"][key].get<double>()).epsilon(1e-5));
      }
      CHECK(g["confidence"].get<double>() ==
            doctest::Approx(want["confidence"].get<double>()).epsilon(1e-5));
    }
  }
  CHECK(got.input_tokens == tokens);
  // The answer is not the uniform prior, so the logits actually moved it.
  CHECK(std::abs(first_logits[0] - first_logits[1]) > 1e-6);

  // The engine's own temperature is the one applied: T=1 must differ.
  auto t1 = vllm::MakeNimbleLoadedModel(MakeWeights(config), config, 1.0, 8192, Q());
  const vllm::NimbleResponse got1 = vllm::NimbleDecide(*t1, tok, body);
  CHECK(std::abs(got1.answers["refund"]["noul"].get<double>() -
                 got.answers["refund"]["noul"].get<double>()) > 1e-6);
}

TEST_CASE("nimble.decide.refuses_an_overlong_prompt_without_truncating") {
  const HfConfig config = MakeConfig();
  auto model = vllm::MakeNimbleLoadedModel(MakeWeights(config), config, 1.0, 100, Q());
  const ojson body = Goldens()["cases"][0]["body"];
  std::string msg;
  try {
    (void)vllm::NimbleDecide(*model, ByteTokenizer(), body);
  } catch (const vllm::nimble::RequestError& e) {
    msg = e.what();
  }
  CHECK(msg.find("limit is 100. Nothing was truncated.") != std::string::npos);
}

TEST_CASE("nimble.decide.is_registered_as_NimbleModel") {
  const auto& reg = vllm::RegistrationFor("NimbleModel");
  CHECK(reg.info.is_pooling_model);
  CHECK_FALSE(reg.info.is_text_generation_model);
}

// ── 4. Real tokenizer (opt-in) ──────────────────────────────────────────────

TEST_CASE("nimble.tokens.match_the_reference_ids_with_the_checkpoint_tokenizer") {
  const char* dir = std::getenv("VLLM_CPP_NIMBLE_TOKENIZER_DIR");
  if (dir == nullptr) {
    MESSAGE("skipped: set VLLM_CPP_NIMBLE_TOKENIZER_DIR to the tokenizer of "
            "bespokelabs/Bespoke-Nimble-9B @ bd792f44");
    return;
  }
  std::ifstream in(std::filesystem::path(dir) / "tokenizer.json", std::ios::binary);
  REQUIRE(in.good());
  std::stringstream ss;
  ss << in.rdbuf();
  const Tokenizer tok = Tokenizer::FromHfJsonBytes(ss.str(), dir);
  for (const ojson& c : Goldens()["cases"]) {
    const std::vector<std::string> prompts =
        vllm::nimble::BuildPrompts(vllm::nimble::CompileRequest(c["body"]));
    for (size_t i = 0; i < prompts.size(); ++i) {
      CHECK(tok.Encode(prompts[i]) == c["ids"][i].get<std::vector<int32_t>>());
      std::vector<int32_t> cand;
      for (size_t k = 0; k < c["candidate_ids"][i].size(); ++k) {
        const std::vector<int32_t> letter =
            tok.Encode(std::string(1, static_cast<char>('A' + k)));
        REQUIRE(letter.size() == 1);
        cand.push_back(letter[0]);
      }
      CHECK(cand == c["candidate_ids"][i].get<std::vector<int32_t>>());
    }
  }
}
