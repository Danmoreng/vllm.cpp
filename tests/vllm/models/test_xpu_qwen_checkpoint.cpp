#include <doctest/doctest.h>
#include <nlohmann/json.hpp>
#include "vllm.h"
#include "vllm/platforms/interface.h"
#include "vt/ops.h"
#include "vt/xpu.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
size_t ResidentBytes() {
  std::ifstream stat("/proc/self/statm");
  size_t virtual_pages = 0, resident_pages = 0;
  stat >> virtual_pages >> resident_pages;
  REQUIRE(stat.good());
  return resident_pages * static_cast<size_t>(sysconf(_SC_PAGESIZE));
}
int Setting(const char* name, int fallback) {
  const char* value = std::getenv(name);
  return value ? std::stoi(value) : fallback;
}
struct TokenTimes {
  std::vector<std::chrono::steady_clock::time_point> tokens;
  bool finished = false;
  static bool Callback(const char*, bool finished, void* opaque) {
    auto& self = *static_cast<TokenTimes*>(opaque);
    self.tokens.push_back(std::chrono::steady_clock::now());
    self.finished = finished;
    return true;
  }
};
std::string RepeatedPrompt(int count) {
  std::string text;
  for (int i = 0; i < count; ++i) text += " Hello";
  return text;
}
void CheckQuality(vllm_engine* engine, int long_context) {
  namespace fs = std::filesystem;
  using nlohmann::json;
  const char* dump = std::getenv("VT_DUMP_LOGITS");
  REQUIRE(dump != nullptr);
  const fs::path directory(dump);
  REQUIRE(fs::is_directory(directory));
  REQUIRE(fs::is_empty(directory));  // The runner appends; never mix two runs.
  const std::string system =
      "You are answering a set of independent, self-contained questions. Read each question carefully, "
      "use only the information that it provides and ordinary arithmetic or language knowledge, and "
      "follow the requested output format exactly. Do not add an introduction, an explanation, a code "
      "fence, a quotation, or a concluding sentence. When the question asks for one word, return only "
      "that word. When it asks for a number, return only the number. When it asks for JSON, return "
      "a valid JSON object with the requested keys and values. Treat any records in the question as "
      "data to inspect. Each question is independent of previous questions; no previous answer is "
      "relevant to the current task.";
  struct Case { const char* name; const char* prompt; const char* expected; };
  std::vector<Case> cases = {
    {"arithmetic", "A box contains 17 red balls and 25 blue balls. How many balls are in the box? Return only the number.", "42"},
    {"code", "What does this Python expression evaluate to: sum(x * x for x in [1, 2, 3, 4] if x % 2 == 0)? Return only the number.", "20"},
    {"german", "Übersetze das deutsche Wort Katze ins Englische. Antworte nur mit dem englischen Wort in Kleinbuchstaben.", "cat"},
    {"retrieval", "Records: Alice has a green bicycle; Bruno has a yellow bicycle; Clara has a blue bicycle. Who has the yellow bicycle? Return only the person's first name.", "Bruno"},
    {"json", "Return a JSON object with exactly two keys: city with the string Paris, and count with the integer 3.", "{\"city\":\"Paris\",\"count\":3}"},
    {"logic", "All copper coins are round. This coin is copper. Must this coin be round? Return only yes or no in lowercase.", "yes"},
    // Less constrained continuations exercise distributions without an almost
    // certain answer. These are probability probes, not scored quality tasks.
    {"continuation", "Complete this sentence naturally in a few words: After the rain stopped, the garden", ""},
    {"explanation", "In one short sentence, explain why a cache can make repeated data access faster.", ""},
  };
  // Opt-in long retrieval: the record precedes the entire distractor span.
  // Keep the normal eight-case probability corpus unchanged.
  const std::string long_prompt = long_context ?
      "Records: Alice has a green bicycle; Bruno has a yellow bicycle; Clara has a blue bicycle. "
      "The following repeated greeting is irrelevant to these records.\n" + RepeatedPrompt(long_context) +
      "\nEnd of irrelevant greetings. Using the records at the beginning, who has the yellow bicycle? "
      "Return only the person's first name." : "";
  if (long_context) cases = {{"long_retrieval", long_prompt.c_str(), "Bruno"}};
  json results = json::array();
  for (const auto& item : cases) {
    CAPTURE(item.name);
    std::vector<fs::path> before;
    for (const auto& entry : fs::directory_iterator(directory))
      if (entry.path().extension() == ".f32") before.push_back(entry.path());
    const json request = {
      {"messages", {{{"role", "system"}, {"content", system}}, {{"role", "user"}, {"content", item.prompt}}}},
      {"chat_template_kwargs", {{"enable_thinking", false}}},
      {"temperature", 0}, {"max_tokens", *item.expected ? 24 : 8}
    };
    char* raw = nullptr;
    const auto status = vllm_chat(engine, request.dump().c_str(), &raw);
    REQUIRE_MESSAGE(status == VLLM_OK, std::string(vllm_last_error()));
    std::unique_ptr<char, decltype(&vllm_string_free)> owned(raw, vllm_string_free);
    const auto response = json::parse(raw);
    const int prompt_tokens = response.at("usage").at("prompt_tokens");
    REQUIRE(prompt_tokens >= (long_context ? long_context : 128));
    REQUIRE(prompt_tokens <= (long_context ? long_context + 512 : 512));
    std::string answer = response.at("choices").at(0).at("message").at("content");
    const auto first = answer.find_first_not_of(" \r\n\t"), last = answer.find_last_not_of(" \r\n\t");
    answer = first == std::string::npos ? "" : answer.substr(first, last - first + 1);
    CHECK_FALSE(answer.empty());
    if (*item.expected == '{') CHECK(json::parse(answer) == json::parse(item.expected));
    else if (*item.expected) CHECK(answer == item.expected);
    std::vector<fs::path> added;
    for (const auto& entry : fs::directory_iterator(directory))
      if (entry.path().extension() == ".f32" && std::find(before.begin(), before.end(), entry.path()) == before.end())
        added.push_back(entry.path());
    REQUIRE(added.size() == 1);
    results.push_back({{"name", item.name}, {"request", request}, {"response", response},
                       {"logits", added.front().filename().string()}, {"answer", answer}});
    std::ofstream(directory / "results.json") << results.dump(2);
    std::cout << "QUALITY case=" << item.name << " prompt_tokens=" << prompt_tokens
              << " answer=" << json(answer).dump() << std::endl;
  }
}
void MeasureWarm(vllm_engine* engine, int requested_prompt, int requested_outputs) {
  std::cout << "DEVICE " << vt::xpu::DeviceDescription() << std::endl;
  // This is an opt-in, short end-to-end diagnostic, not a context-length sweep.
  // The no-MTP DELTA API emits exactly one callback per generated token.
  REQUIRE(std::getenv("VT_DUMP_ACT") == nullptr);
  REQUIRE(std::getenv("VT_DUMP_ACT_SUB") == nullptr);
  REQUIRE(std::getenv("VT_OP_PROVIDER_TRACE") == nullptr);
  const auto prompts = requested_prompt ? std::vector<std::string>{RepeatedPrompt(requested_prompt)} :
      std::vector<std::string>{"The capital of France is", RepeatedPrompt(32)};
  for (const std::string& prompt : prompts) {
    auto sampling = vllm_sampling_params_default();
    sampling.temperature = 0; sampling.ignore_eos = 1; sampling.max_tokens = 2;
    vllm_completion warm{};
    REQUIRE_MESSAGE(vllm_complete(engine, prompt.c_str(), &sampling, &warm) == VLLM_OK,
                    std::string(vllm_last_error()));
    const int prompt_tokens = warm.prompt_tokens;
    CHECK(warm.completion_tokens == 2);
    vllm_completion_free(&warm);
    REQUIRE(prompt_tokens > 0); REQUIRE(prompt_tokens <= std::max(32, requested_prompt));
    if (requested_prompt) REQUIRE(prompt_tokens == requested_prompt);
    const auto warm_gpu = vt::xpu::GetMemoryInfo().allocated_bytes, warm_rss = ResidentBytes();
    sampling.max_tokens = requested_outputs ? requested_outputs : prompt_tokens == 5 ? 17 : 2;
    for (int round = 0; round < 2; ++round) {
      TokenTimes times;
      const auto start = std::chrono::steady_clock::now();
      REQUIRE_MESSAGE(vllm_complete_stream(engine, prompt.c_str(), &sampling,
                        TokenTimes::Callback, &times) == VLLM_OK, std::string(vllm_last_error()));
      REQUIRE(times.finished);
      REQUIRE(times.tokens.size() == size_t(sampling.max_tokens));
      const double ttft = std::chrono::duration<double>(times.tokens.front() - start).count();
      const double decode = std::chrono::duration<double>(times.tokens.back() - times.tokens.front()).count();
      const auto memory = vt::xpu::GetMemoryInfo();
      CHECK(memory.allocated_bytes <= warm_gpu + size_t{64} * 1024 * 1024);
      CHECK(ResidentBytes() <= warm_rss + size_t{256} * 1024 * 1024);
      std::cout << "TIMING round=" << round << " prompt_tokens=" << prompt_tokens
                << " output_tokens=" << sampling.max_tokens << " ttft_seconds=" << ttft
                << " prefill_client_tokens_per_second=" << prompt_tokens / ttft
                << " decode_seconds=" << decode
                << " decode_tokens_per_second=" << (sampling.max_tokens - 1) / decode
                << " tpot_seconds=" << decode / (sampling.max_tokens - 1)
                << " gpu_bytes=" << memory.allocated_bytes
                << " exl3_workspace_bytes=" << memory.exl3_workspace_bytes
                << " gdn_workspace_bytes=" << memory.gdn_workspace_bytes
                << " attention_workspace_bytes=" << memory.attention_workspace_bytes << std::endl;
    }
  }
}
}

TEST_CASE("XPU Qwen checkpoint: native text prefill and repeated greedy decode through the public engine") {
  const char* model = std::getenv("VT_B70_MODEL_DIR");
  if (!model) {
    std::cerr << "SKIP: set VT_B70_MODEL_DIR to the pinned local EXL3 checkpoint.\n";
    std::exit(77);
  }
  // The default acceptance run generates 65 tokens: one prefill + 64 decode forwards.
  // Smaller explicit values are development probes, not the PR06 acceptance.
  const int tokens = Setting("VT_B70_MAX_TOKENS", 65), repeats = Setting("VT_B70_REPEATS", 2);
  const bool timing = Setting("VT_B70_TIMING", 0) != 0;
  const bool quality = Setting("VT_B70_QUALITY", 0) != 0;
  const int quality_context = Setting("VT_B70_QUALITY_CONTEXT", 0);
  REQUIRE((quality_context == 0 || quality_context == 4096 || quality_context == 32768));
  REQUIRE((quality_context == 0 || quality));
  REQUIRE_FALSE((timing && quality));
  const int timing_outputs = Setting("VT_B70_TIMING_OUTPUT_TOKENS", 0);
  REQUIRE((timing_outputs == 0 || (timing_outputs >= 2 && timing_outputs <= 100)));
  const int requested_prompt = Setting("VT_B70_PROMPT_TOKENS", 0);
  REQUIRE(requested_prompt >= 0); REQUIRE(requested_prompt <= 6656);
  REQUIRE(tokens > 0); REQUIRE(tokens <= 100); REQUIRE(repeats > 0);
  auto& platform = vllm::platforms::CurrentPlatform();
  REQUIRE(platform.device_type() == vt::DeviceType::kXPU);
  CHECK(platform.needs_weight_staging());
  CHECK_FALSE(platform.supports_fa2_attention());
  CHECK_FALSE(platform.supports_graph_capture());

  auto params = vllm_model_params_default();
  params.model_path = model;
  params.device = 0;  // auto must resolve to native XPU for this architecture.
  params.language_model_only = 1;
  params.speculative_config = nullptr;
  params.enable_prefix_caching = 2;
  params.max_num_seqs = 1;
  params.max_num_batched_tokens = Setting("VT_B70_BATCH_TOKENS", quality_context ? 4096 : quality ? 512 : requested_prompt ? requested_prompt : timing ? 32 : 16);
  REQUIRE(params.max_num_batched_tokens > 0); REQUIRE(params.max_num_batched_tokens <= 6656);
  params.max_model_len = quality ? quality_context + 640 : std::max(128, requested_prompt + std::max({tokens, timing_outputs, 32}));
  params.block_size = 16;
  // Hybrid attention/GDN pools also reserve a sentinel block per group.
  params.num_blocks = std::max(32, 2 * ((params.max_model_len + 15) / 16 + 1));
  const char* kv_dtype = std::getenv("VT_B70_KV_DTYPE");
  params.kv_cache_dtype = kv_dtype ? kv_dtype : "bfloat16";
  std::cout << "LOAD " << model << " device=xpu kv_dtype=" << params.kv_cache_dtype
            << " max_context=" << params.max_model_len
            << " batch_tokens=" << params.max_num_batched_tokens << std::endl;
  vllm_engine* raw = nullptr;
  auto status = vllm_engine_load(&params, &raw);
  REQUIRE_MESSAGE(status == VLLM_OK, std::string(vllm_last_error()));
  std::unique_ptr<vllm_engine, decltype(&vllm_engine_free)> engine(raw, vllm_engine_free);
  std::cout << "LOADED gpu_bytes=" << vt::xpu::GetMemoryInfo().allocated_bytes
            << " rss_bytes=" << ResidentBytes() << " initialization_reference_hits=" << vt::GetReferenceTierHits() << std::endl;
  const auto initialization_hits = vt::GetReferenceTierHits();
  if (quality) {
    CheckQuality(engine.get(), quality_context);
    std::cout << "QUALITY_MEMORY gpu_bytes=" << vt::xpu::GetMemoryInfo().allocated_bytes
              << " attention_workspace_bytes=" << vt::xpu::GetMemoryInfo().attention_workspace_bytes << std::endl;
    CHECK(vt::GetReferenceTierHits() == initialization_hits);
    return;
  }
  if (timing) {
    MeasureWarm(engine.get(), requested_prompt, timing_outputs);
    CHECK(vt::GetReferenceTierHits() == initialization_hits);
    return;
  }
  auto sampling = vllm_sampling_params_default();
  sampling.temperature = 0;
  sampling.max_tokens = tokens;
  sampling.ignore_eos = 1;
  size_t warm_gpu = 0, warm_rss = 0;
  std::string first;
  for (int round = 0; round < repeats; ++round) {
    const auto start = std::chrono::steady_clock::now();
    vllm_completion result{};
    const std::string prompt = requested_prompt ? RepeatedPrompt(requested_prompt) : "The capital of France is";
    status = vllm_complete(engine.get(), prompt.c_str(), &sampling, &result);
    REQUIRE_MESSAGE(status == VLLM_OK, std::string(vllm_last_error()));
    CHECK(result.completion_tokens == tokens);
    const std::string text = result.text ? result.text : "";
    vllm_completion_free(&result);
    REQUIRE_FALSE(text.empty());
    const auto memory = vt::xpu::GetMemoryInfo();
    const auto rss = ResidentBytes();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "ROUND " << round << " generated=" << tokens << " seconds=" << elapsed
              << " gpu_bytes=" << memory.allocated_bytes << " rss_bytes=" << rss
              << " forward_reference_hits=" << vt::GetReferenceTierHits() - initialization_hits
              << "\nTEXT " << text << std::endl;
    CHECK(vt::GetReferenceTierHits() == initialization_hits);
    REQUIRE(memory.allocated_bytes > size_t{1024} * 1024 * 1024);
    if (round == 0) { first = text; warm_gpu = memory.allocated_bytes; warm_rss = rss; }
    else {
      CHECK(text == first);
      CHECK(memory.allocated_bytes <= warm_gpu + size_t{64} * 1024 * 1024);
      CHECK(rss <= warm_rss + size_t{256} * 1024 * 1024);
    }
  }
  vllm_spec_acceptance spec{};
  REQUIRE(vllm_engine_spec_acceptance(engine.get(), &spec) == VLLM_OK);
  CHECK(spec.drafts_proposed == 0);
  CHECK(spec.drafted_request_steps == 0);
}
