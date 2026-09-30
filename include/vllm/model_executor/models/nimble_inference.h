// Nimble decision model (MODEL-NIMBLE): the ONE library seam behind both
// POST /v1/systemone and vllm_decide for a "NimbleModel" engine.
//
// Nimble (bespokelabs/Bespoke-Nimble-9B) is a LoRA on the Qwen3.5-9B dense
// backbone, merged at convert time by scripts/convert-nimble.py. It answers
// each schema field with one forward: the last-position logits of the
// candidate letter tokens, then softmax(logits / T). Nothing is sampled.
//
// The seam is request-level, not per-question, because every field's prompt
// embeds the WHOLE schema (parallel_schema.py prepare_prompts). References:
//   prompt   bespokelabs/Bespoke-Nimble-9B @ bd792f44 parallel_schema.py
//   mapping  bespokelabsai/nimble @ 62076b4f nimble/serving/compiler.py
//   answer   ekzhang/openjev-sglang @ 7f84bedc src/openjev/scoring.py
// See .agents/specs/nimble.md.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

namespace vllm {

class LoadedModel;
namespace tok {
class Tokenizer;
}

namespace nimble {

// parallel_schema.py serves at most 26 one-letter codes. Wider fields switch
// to extended_schema.py, which this port does not implement yet.
inline constexpr int kMaxChoices = 26;
// openjev defaults.py MAX_QUESTIONS and MAX_ANSWERS.
inline constexpr int kMaxQuestions = 64;
inline constexpr int kMaxAnswers = 64;
// schema_config.json max_length; the converter records the checkpoint's own.
inline constexpr int64_t kDefaultMaxLength = 8192;

// parallel_schema.py SYSTEM_PROMPT, verbatim.
extern const char* const kSystemPrompt;

// A refusal of the request itself (HTTP 400, VLLM_ERR_INVALID_ARGUMENT).
class RequestError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

// One schema field, compiled from one Jev question (compiler.py).
struct Field {
  std::string name;         // the question id
  std::string type;         // "noul" | "choice" | "score"
  std::string description;  // serialize(instructions)
  // The schema values in candidate order: false/true for noul, the criteria
  // keys for choice, "0".."L-1" for score.
  std::vector<nlohmann::ordered_json> values;
  // One description per value (compiler.py always supplies one).
  std::vector<std::string> value_descriptions;
  // Answer keys, in the same order: "false"/"true", criteria keys, "0"..
  std::vector<std::string> keys;
};

struct Request {
  std::string model;    // empty when the request names none
  std::string context;  // serialize(state)
  std::vector<Field> fields;
};

// Python json.dumps(ensure_ascii=False) with the default ", " / ": "
// separators, then "<" and ">" replaced by < / > (safe_json).
std::string SafeJson(const nlohmann::ordered_json& value);

// compiler.py serialize(): a string as-is, anything else json.dumps.
std::string Serialize(const nlohmann::ordered_json& value);

// Validate and compile a /v1/systemone body. Throws RequestError.
Request CompileRequest(const nlohmann::ordered_json& body);

// The rendered chat prompt for every field, in field order.
std::vector<std::string> BuildPrompts(const Request& request);

// The openjev answer for one field from its candidate logits.
nlohmann::ordered_json AnswerFromLogits(const Field& field,
                                        const std::vector<double>& logits,
                                        double temperature);

// openjev confidence(): clamp(1 - H(p) / ln(n), 0, 1).
double EntropyConfidence(const std::vector<double>& probabilities);

}  // namespace nimble

// The answers and token accounting for one request.
struct NimbleResponse {
  nlohmann::ordered_json answers = nlohmann::ordered_json::object();
  int64_t input_tokens = 0;
};

// Run a whole /v1/systemone request on a NimbleModel engine. Throws
// nimble::RequestError for a request the reference refuses, and
// std::runtime_error for an engine fault.
NimbleResponse NimbleDecide(const LoadedModel& model,
                            const tok::Tokenizer& tokenizer,
                            const nlohmann::ordered_json& body);

// Wrap already-built weights as a NimbleModel (synthetic tests), the same
// shape as MakeQwen3_5DenseLoadedModel. `queue` is the device the forward runs
// on; a loaded engine gets it from the factory's prepare callback instead.
std::unique_ptr<LoadedModel> MakeNimbleLoadedModel(Qwen3_5DenseWeights weights,
                                                   HfConfig config,
                                                   double temperature,
                                                   int64_t max_length,
                                                   vt::Queue queue);

}  // namespace vllm
