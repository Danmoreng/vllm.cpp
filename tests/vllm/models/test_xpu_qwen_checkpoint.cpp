#include <doctest/doctest.h>
#include <nlohmann/json.hpp>
#include <blake3.h>
#include "vllm.h"
#include "vllm/platforms/interface.h"
#include "vllm/entrypoints/chat_template.h"
#include "vllm/tokenizer/tokenizer.h"
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
#include <array>
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
  std::vector<size_t> submission_counts;
  bool profile_counts = false;
  bool finished = false;
  static bool Callback(const char*, bool finished, void* opaque) {
    auto& self = *static_cast<TokenTimes*>(opaque);
    self.tokens.push_back(std::chrono::steady_clock::now());
    if (self.profile_counts) self.submission_counts.push_back(vt::xpu::PendingProfileEventCount());
    self.finished = finished;
    return true;
  }
};
std::string RepeatedPrompt(int count) {
  std::string text;
  for (int i = 0; i < count; ++i) text += " Hello";
  return text;
}
void CheckBatch(vllm_engine* engine) {
  struct Result {
    std::string text;
    int chunks = 0, cancel_after = 0;
    static bool Callback(const char* text, bool, void* opaque) {
      auto& r = *static_cast<Result*>(opaque); r.text += text ? text : ""; ++r.chunks;
      return !r.cancel_after || r.chunks < r.cancel_after;
    }
  };
  std::array<std::string, 4> prompts, expected;
  std::array<vllm_sampling_params, 4> settings;
  const char* tails[] = {" The capital of France is", " The capital of Germany is", " The capital of Italy is", " The capital of Spain is"};
  for (int i = 0; i < 4; ++i) {
    prompts[i] = RepeatedPrompt(128 + 32 * i) + tails[i];
    settings[i] = vllm_sampling_params_default();
    settings[i].temperature = i == 0 ? 0 : .7f;
    settings[i].has_seed = 1; settings[i].seed = 381 + i;
    settings[i].top_k = 20; settings[i].top_p = .9f; settings[i].min_p = .05f;
    settings[i].repetition_penalty = 1.05f;
    // Leave enough decode steps to warm and capture both slots at B=4, then
    // exercise condensation into the smaller buckets as requests complete.
    settings[i].ignore_eos = 1;
    settings[i].max_tokens = 3 + i + (Setting("VT_XPU_GRAPH", 0) == 1 ? 6 : 0);
    vllm_completion result{};
    REQUIRE_MESSAGE(vllm_complete(engine, prompts[i].c_str(), &settings[i], &result) == VLLM_OK, std::string(vllm_last_error()));
    CHECK(result.completion_tokens == settings[i].max_tokens);
    expected[i] = result.text ? result.text : ""; vllm_completion_free(&result);
  }
  const auto baseline = vt::xpu::GetMemoryInfo().allocated_bytes;
  using Request = std::unique_ptr<vllm_request, decltype(&vllm_request_free)>;
  std::array<Result, 4> results;
  std::vector<Request> requests;
  // Submit longest first so short requests finish and the batch condenses.
  for (int i : {3, 0, 2, 1}) {
    vllm_request* raw = nullptr;
    REQUIRE_MESSAGE(vllm_request_submit(engine, prompts[i].c_str(), &settings[i], Result::Callback, &results[i], &raw) == VLLM_OK, std::string(vllm_last_error()));
    requests.emplace_back(raw, vllm_request_free);
  }
  for (auto& request : requests) REQUIRE_MESSAGE(vllm_request_wait(request.get()) == VLLM_OK, std::string(vllm_request_error(request.get())));
  for (int i = 0; i < 4; ++i) {
    CAPTURE(i);
    CHECK(results[i].text == expected[i]);
    std::cout << "BATCH case=" << i << " answer=" << nlohmann::json(results[i].text).dump() << std::endl;
  }
  requests.clear();
  if (Setting("VT_XPU_GRAPH", 0) == 1) {
    // Condensation can cross B=2 for only one step. A second, sustained pair
    // proves that this bucket also captures both slots and stays within the
    // six-executable bound for buckets 1/2/4.
    std::array<Result, 4> pair;
    for (int i : {3, 1}) {
      vllm_request* raw = nullptr;
      REQUIRE(vllm_request_submit(engine, prompts[i].c_str(), &settings[i], Result::Callback, &pair[i], &raw) == VLLM_OK);
      requests.emplace_back(raw, vllm_request_free);
    }
    for (auto& request : requests) REQUIRE_MESSAGE(vllm_request_wait(request.get()) == VLLM_OK,
                                                   std::string(vllm_request_error(request.get())));
    for (int i : {3, 1}) CHECK(pair[i].text == expected[i]);
    requests.clear();
    CHECK(vt::xpu::GetMemoryInfo().graph_count == 6);
  }
  Result cancelled; cancelled.cancel_after = 2;
  auto cancel_settings = settings[3]; cancel_settings.max_tokens = 16;
  vllm_request* raw = nullptr;
  REQUIRE(vllm_request_submit(engine, prompts[3].c_str(), &cancel_settings, Result::Callback, &cancelled, &raw) == VLLM_OK);
  Request request(raw, vllm_request_free);
  REQUIRE(vllm_request_wait(raw) == VLLM_OK); CHECK(cancelled.chunks >= 2); request.reset();
  // Reuse after cancellation must start from its own clean recurrent state.
  vllm_completion resumed{};
  REQUIRE_MESSAGE(vllm_complete(engine, prompts[3].c_str(), &settings[3], &resumed) == VLLM_OK, std::string(vllm_last_error()));
  CHECK(std::string(resumed.text ? resumed.text : "") == expected[3]); vllm_completion_free(&resumed);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes <= baseline + 64 * 1024 * 1024);
}
void CheckPrefix(vllm_engine* engine) {
  const std::array<std::string, 2> prompts = {
      RepeatedPrompt(256) + " The capital of France is",
      RepeatedPrompt(256) + " The capital of Germany is"};
  std::array<std::string, 2> reference;
  auto sampling = vllm_sampling_params_default();
  sampling.temperature = 0; sampling.max_tokens = 5; sampling.ignore_eos = 1;
  size_t baseline = 0;
  for (int step = 0; step < 5; ++step) {
    const int which = step % 2;
    vllm_completion result{};
    REQUIRE_MESSAGE(vllm_complete(engine, prompts[which].c_str(), &sampling, &result) == VLLM_OK, std::string(vllm_last_error()));
    REQUIRE(result.completion_tokens == 5);
    const std::string answer = result.text ? result.text : "";
    if (step < 2) reference[which] = answer; else CHECK(answer == reference[which]);
    REQUIRE_FALSE(answer.empty()); vllm_completion_free(&result);
    const auto memory = vt::xpu::GetMemoryInfo();
    if (!step) baseline = memory.allocated_bytes;
    CHECK(memory.allocated_bytes <= baseline + 64 * 1024 * 1024);
    std::cout << "PREFIX case=" << which << " answer=" << nlohmann::json(answer).dump()
              << " gpu_bytes=" << memory.allocated_bytes << " tracked_peak_gpu_bytes=" << memory.peak_allocated_bytes << std::endl;
  }
}
std::vector<std::string> PrefixHashes(const std::vector<int32_t>& prompt,
                                      const std::vector<int32_t>& forced) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  const auto append = [&](int32_t token) {
    const uint32_t value = static_cast<uint32_t>(token);
    const unsigned char bytes[4] = {static_cast<unsigned char>(value),
        static_cast<unsigned char>(value >> 8), static_cast<unsigned char>(value >> 16),
        static_cast<unsigned char>(value >> 24)};
    blake3_hasher_update(&hasher, bytes, sizeof(bytes));
  };
  for (int32_t token : prompt) append(token);
  std::vector<std::string> hashes;
  hashes.reserve(forced.size());
  constexpr char digits[] = "0123456789abcdef";
  for (int32_t token : forced) {
    unsigned char digest[32];
    blake3_hasher_finalize(&hasher, digest, sizeof(digest));
    std::string hex;
    hex.reserve(64);
    for (unsigned char byte : digest) { hex += digits[byte >> 4]; hex += digits[byte & 15]; }
    hashes.push_back(std::move(hex));
    append(token);
  }
  return hashes;
}
struct ForcedTokens {
  const std::vector<int32_t>* ids;
  bool invalid = false;
  static void Callback(const int32_t* prefix, int32_t count, float* logits,
                       int32_t vocab, void* opaque) {
    auto& self = *static_cast<ForcedTokens*>(opaque);
    if (count < 0 || count >= static_cast<int32_t>(self.ids->size()) ||
        (*self.ids)[static_cast<size_t>(count)] < 0 ||
        (*self.ids)[static_cast<size_t>(count)] >= vocab) {
      self.invalid = true;
      return;
    }
    for (int32_t i = 0; i < count; ++i) {
      if (prefix[i] != (*self.ids)[static_cast<size_t>(i)]) {
        self.invalid = true;
        return;
      }
    }
    std::fill_n(logits, vocab, -1.0e30f);
    logits[(*self.ids)[static_cast<size_t>(count)]] = 0.0f;
  }
};
constexpr const char* kQualitySystem =
    "You are answering a set of independent, self-contained questions. Read each question carefully, "
    "use only the information that it provides and ordinary arithmetic or language knowledge, and "
    "follow the requested output format exactly. Do not add an introduction, an explanation, a code "
    "fence, a quotation, or a concluding sentence. When the question asks for one word, return only "
    "that word. When it asks for a number, return only the number. When it asks for JSON, return "
    "a valid JSON object with the requested keys and values. Treat any records in the question as "
    "data to inspect. Each question is independent of previous questions; no previous answer is "
    "relevant to the current task.";
constexpr std::array<std::pair<const char*, const char*>, 24> kTeacherCases = {{
    {"arithmetic", "A box contains 17 red balls and 25 blue balls. How many balls are in the box? Return only the number."},
    {"code", "What does this Python expression evaluate to: sum(x * x for x in [1, 2, 3, 4] if x % 2 == 0)? Return only the number."},
    {"german", "Übersetze das deutsche Wort Katze ins Englische. Antworte nur mit dem englischen Wort in Kleinbuchstaben."},
    {"retrieval", "Records: Alice has a green bicycle; Bruno has a yellow bicycle; Clara has a blue bicycle. Who has the yellow bicycle? Return only the person's first name."},
    {"json", "Return a JSON object with exactly two keys: city with the string Paris, and count with the integer 3."},
    {"logic", "All copper coins are round. This coin is copper. Must this coin be round? Return only yes or no in lowercase."},
    {"continuation", "Complete this sentence naturally in a few words: After the rain stopped, the garden"},
    {"explanation", "In one short sentence, explain why a cache can make repeated data access faster."},
    {"heldout_de_01", "Fasse in einem Satz zusammen: Der Zug kam pünktlich, aber der Anschlussbus fiel aus."},
    {"heldout_de_02", "Setze den Satz fort: Im stillen Museum betrachtete sie"},
    {"heldout_de_03", "Welche Zahl ist größer: 37 oder 73? Antworte mit der Zahl."},
    {"heldout_de_04", "Ordne die Wörter alphabetisch: Birne, Apfel, Kirsche."},
    {"heldout_en_01", "Write one sentence explaining why a metal spoon feels colder than a wooden spoon."},
    {"heldout_en_02", "Continue the story: The library door opened just as"},
    {"heldout_en_03", "Which of these is a mammal: trout, dolphin, or sparrow?"},
    {"heldout_en_04", "Rewrite in past tense: The gardener waters the roses."},
    {"heldout_code_01", "In Python, write a function that returns the sum of even integers in a list."},
    {"heldout_code_02", "What does JavaScript Array.prototype.map return? Answer in one sentence."},
    {"heldout_code_03", "Find the bug: for (int i = 0; i <= n; ++i) a[i] = 0; when a has n elements."},
    {"heldout_code_04", "Write a SQL query selecting names from users where age is at least 18."},
    {"heldout_json_01", "Return JSON with keys ok (true) and count (7)."},
    {"heldout_json_02", "Return a JSON array containing the integers 2, 3, and 5."},
    {"heldout_retrieval_01", "Records: Mina owns the red cup; Noah owns the blue cup; Iris owns the green cup. Who owns the blue cup?"},
    {"heldout_retrieval_02", "Memo: Monday is design review, Tuesday is testing, Wednesday is deployment. Which day is testing?"},
}};
void CheckTeacherForced(vllm_engine* engine, const std::string& model, bool candidate) {
  namespace fs = std::filesystem;
  using nlohmann::json;
  const char* dump = std::getenv("VT_DUMP_LOGITS");
  REQUIRE(dump != nullptr);
  const fs::path directory(dump);
  REQUIRE(fs::is_directory(directory));
  REQUIRE(fs::is_empty(directory));
  const int count = Setting("VT_B70_TEACHER_CASES", 24);
  const int steps = Setting("VT_B70_TEACHER_STEPS", 32);
  REQUIRE(count >= 1); REQUIRE(count <= 24);
  REQUIRE(steps >= 1); REQUIRE(steps <= 32);
  std::ifstream source(fs::path(model) / "SOURCE-REVISION.txt");
  REQUIRE(source.good());
  std::string revision, line;
  while (std::getline(source, line))
    if (line.starts_with("Revision: ")) revision = line.substr(10);
  REQUIRE(revision == "19441ac874c4018295da848e250f23511361cda4");
  const auto tokenizer = vllm::tok::Tokenizer::FromHfJson((fs::path(model) / "tokenizer.json").string());
  const auto template_text = vllm::entrypoints::LoadChatTemplateFromConfig(
      (fs::path(model) / "tokenizer_config.json").string());
  const auto bos = tokenizer.BosId() >= 0 ? tokenizer.Decode({tokenizer.BosId()}) : std::string();
  const auto eos = tokenizer.EosId() >= 0 ? tokenizer.Decode({tokenizer.EosId()}) : std::string();
  const auto render_prompt = vllm::entrypoints::MakeChatTemplatePromptFn(template_text, bos, eos);
  const nlohmann::ordered_json template_kwargs = {{"enable_thinking", false}};
  json reference = json::array();
  if (candidate) {
    const char* path = std::getenv("VT_B70_TEACHER_REFERENCE");
    REQUIRE(path != nullptr);
    std::ifstream stream(fs::path(path) / "results.json");
    REQUIRE(stream.good());
    stream >> reference;
    REQUIRE(reference.size() >= static_cast<size_t>(count));
  }
  json results = json::array();
  for (int i = 0; i < count; ++i) {
    const auto& [name, prompt] = kTeacherCases[static_cast<size_t>(i)];
    CAPTURE(name);
    const std::vector<vllm::entrypoints::openai::ChatMessage> messages = {
        {"system", std::string(kQualitySystem)}, {"user", std::string(prompt)}};
    const auto prompt_ids = tokenizer.EncodeWithSpecialTokens(
        render_prompt(messages, true, {}, template_kwargs));
    REQUIRE_FALSE(prompt_ids.empty());
    REQUIRE(prompt_ids.size() <= 512);  // One prefill chunk in this fixed corpus.
    const json request = {{"prompt_token_ids", prompt_ids}, {"messages", messages},
                          {"chat_template_kwargs", template_kwargs}, {"temperature", 0},
                          {"ignore_eos", true}, {"max_tokens", steps}};
    std::vector<int32_t> forced;
    if (candidate) {
      const auto& prior = reference.at(i);
      REQUIRE(prior.at("name") == name);
      REQUIRE(prior.at("request") == request);
      forced = prior.at("teacher_forced").at("forced_token_ids").get<std::vector<int32_t>>();
      REQUIRE(forced.size() == static_cast<size_t>(steps));
    }
    std::vector<fs::path> before;
    for (const auto& entry : fs::directory_iterator(directory))
      if (entry.path().extension() == ".f32") before.push_back(entry.path());
    auto sampling = vllm_sampling_params_default();
    sampling.temperature = 0; sampling.ignore_eos = 1; sampling.max_tokens = steps;
    ForcedTokens selector{&forced};
    if (candidate) {
      sampling.logits_processor = ForcedTokens::Callback;
      sampling.logits_processor_user_data = &selector;
    }
    std::vector<int32_t> output(static_cast<size_t>(steps));
    int32_t output_count = 0;
    vllm_completion completion{};
    REQUIRE_MESSAGE(vllm_complete_tokens(engine, prompt_ids.data(),
                    static_cast<int32_t>(prompt_ids.size()), &sampling, output.data(), steps,
                    &output_count, &completion) == VLLM_OK, std::string(vllm_last_error()));
    REQUIRE_FALSE(selector.invalid);
    REQUIRE(output_count == steps);
    REQUIRE(completion.completion_tokens == steps);
    REQUIRE(completion.prompt_tokens == static_cast<int32_t>(prompt_ids.size()));
    output.resize(static_cast<size_t>(output_count));
    if (candidate) REQUIRE(output == forced);
    else forced = output;
    std::vector<int64_t> positions;
    for (int step = 0; step < steps; ++step)
      positions.push_back(static_cast<int64_t>(prompt_ids.size()) + step - 1);
    std::vector<fs::path> added;
    for (const auto& entry : fs::directory_iterator(directory))
      if (entry.path().extension() == ".f32" &&
          std::find(before.begin(), before.end(), entry.path()) == before.end()) added.push_back(entry.path());
    REQUIRE(added.size() == 1);
    results.push_back({{"name", name}, {"request", request},
        {"response", {{"usage", {{"prompt_tokens", prompt_ids.size()},
                                  {"completion_tokens", output_count}}}}},
        {"logits", added.front().filename().string()},
        {"answer", completion.text ? completion.text : ""},
        {"teacher_forced", {{"planned_steps", steps}, {"forced_token_ids", forced},
            {"prefix_hashes", PrefixHashes(prompt_ids, forced)},
            {"prefix_hash_algorithm", "blake3-token-ids-le32-v1"},
            {"position_ids", positions}, {"chunk_schedule", {prompt_ids.size()}},
            {"state_reset", true}, {"checkpoint_revision", revision}}}});
    vllm_completion_free(&completion);
    std::ofstream(directory / "results.json") << results.dump(2);
    std::cout << "TEACHER case=" << name << " prompt_tokens=" << prompt_ids.size()
              << " forced_steps=" << steps << std::endl;
  }
}
void CheckQuality(vllm_engine* engine, int long_context) {
  namespace fs = std::filesystem;
  using nlohmann::json;
  const char* dump = std::getenv("VT_DUMP_LOGITS");
  REQUIRE(dump != nullptr);
  const fs::path directory(dump);
  REQUIRE(fs::is_directory(directory));
  REQUIRE(fs::is_empty(directory));  // The runner appends; never mix two runs.
  const std::string system = kQualitySystem;
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
void ProfileSingleRequest(vllm_engine* engine, int requested_prompt, int outputs) {
  using nlohmann::json;
  const bool graph = Setting("VT_XPU_GRAPH", 0) == 1;
  const char* path = std::getenv("VT_B70_PROFILE_OUT");
  REQUIRE(path != nullptr);
  REQUIRE(outputs >= 2); REQUIRE(outputs <= 100);
  const std::string prompt = requested_prompt ? RepeatedPrompt(requested_prompt) :
      "The capital of France is";
  auto sampling = vllm_sampling_params_default();
  sampling.temperature = 0; sampling.ignore_eos = 1;
  sampling.max_tokens = graph ? 10 : 2;
  vllm_completion warm{};
  REQUIRE_MESSAGE(vllm_complete(engine, prompt.c_str(), &sampling, &warm) == VLLM_OK,
                  std::string(vllm_last_error()));
  const int prompt_tokens = warm.prompt_tokens;
  vllm_completion_free(&warm);
  if (requested_prompt) REQUIRE(prompt_tokens == requested_prompt);
  // The warm request has completed; its events do not belong to this window.
  (void)vt::xpu::DrainProfileEvents();
  (void)vt::xpu::DrainHostProfileRecords();
  const auto clock_before = vt::xpu::CaptureProfileClockAnchor();
  sampling.max_tokens = outputs;
  TokenTimes times;
  times.profile_counts = true;
  const auto start = std::chrono::steady_clock::now();
  REQUIRE_MESSAGE(vllm_complete_stream(engine, prompt.c_str(), &sampling,
                    TokenTimes::Callback, &times) == VLLM_OK, std::string(vllm_last_error()));
  const auto finish = std::chrono::steady_clock::now();
  const auto clock_after = vt::xpu::CaptureProfileClockAnchor();
  REQUIRE(times.finished);
  REQUIRE(times.tokens.size() == static_cast<size_t>(outputs));
  const auto records = vt::xpu::DrainProfileEvents();
  const auto host_records = vt::xpu::DrainHostProfileRecords();
  REQUIRE_FALSE(records.empty());
  REQUIRE(times.submission_counts.size() == times.tokens.size());
  REQUIRE(times.submission_counts.back() <= records.size());
  std::ofstream out(path);
  REQUIRE(out.good());
  const auto ns = [](std::chrono::steady_clock::time_point value) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(value.time_since_epoch()).count();
  };
  json callbacks = json::array();
  for (const auto& token : times.tokens) callbacks.push_back(ns(token));
  const auto memory = vt::xpu::GetMemoryInfo();
  const auto anchor_json = [](const vt::xpu::ProfileClockAnchor& anchor) {
    return json{{"host_before_ns", anchor.host_before_ns}, {"host_after_ns", anchor.host_after_ns},
                {"device_start_ns", anchor.device_start_ns}, {"device_end_ns", anchor.device_end_ns}};
  };
  out << json({{"event", "request"}, {"profiling_mode", graph ? "sycl_graph_span_events" : "sycl_eager_events"},
               {"device", json::parse(vt::xpu::DeviceDescription())},
               {"prompt_tokens", prompt_tokens}, {"output_tokens", outputs},
               {"start_steady_ns", ns(start)}, {"callback_steady_ns", callbacks},
               {"clock_anchor_before", anchor_json(clock_before)},
               {"clock_anchor_after", anchor_json(clock_after)},
               {"callback_submission_counts", times.submission_counts},
               {"finish_steady_ns", ns(finish)}, {"profiled_submissions", records.size()},
               {"profiled_host_spans", host_records.size()},
               {"peak_tracked_gpu_bytes", memory.peak_allocated_bytes},
               {"live_tracked_gpu_bytes", memory.allocated_bytes}}).dump() << '\n';
  for (const auto& record : records)
    out << json({{"event", "gpu_submission"}, {"stage", record.stage},
                 {"matrix", record.matrix},
                 {"queue_id", record.queue_id}, {"submit_ns", record.submit_ns},
                 {"start_ns", record.start_ns}, {"end_ns", record.end_ns}}).dump() << '\n';
  for (const auto& record : host_records)
    out << json({{"event", "host_span"}, {"stage", record.stage},
                 {"queue_id", record.queue_id}, {"start_steady_ns", record.start_steady_ns},
                 {"end_steady_ns", record.end_steady_ns}}).dump() << '\n';
  REQUIRE(out.good());
  std::cout << "PROFILE prompt_tokens=" << prompt_tokens << " output_tokens=" << outputs
            << " submissions=" << records.size() << " path=" << path << std::endl;
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
    // Both persistent input slots must complete their cold/warm/capture phases
    // before timing. Two output tokens would charge graph capture to decode.
    const int warm_outputs = Setting("VT_XPU_GRAPH", 0) == 1 ? 10 : 2;
    sampling.temperature = 0; sampling.ignore_eos = 1; sampling.max_tokens = warm_outputs;
    vllm_completion warm{};
    REQUIRE_MESSAGE(vllm_complete(engine, prompt.c_str(), &sampling, &warm) == VLLM_OK,
                    std::string(vllm_last_error()));
    const int prompt_tokens = warm.prompt_tokens;
    CHECK(warm.completion_tokens == warm_outputs);
    vllm_completion_free(&warm);
    REQUIRE(prompt_tokens > 0); REQUIRE(prompt_tokens <= std::max(32, requested_prompt));
    if (requested_prompt) REQUIRE(prompt_tokens == requested_prompt);
    const auto warm_gpu = vt::xpu::GetMemoryInfo().allocated_bytes, warm_rss = ResidentBytes();
    const auto warm_captures = vt::GetBackend(vt::DeviceType::kXPU).GraphsCaptured();
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
      CHECK(vt::GetBackend(vt::DeviceType::kXPU).GraphsCaptured() == warm_captures);
      std::cout << "TIMING round=" << round << " prompt_tokens=" << prompt_tokens
                << " output_tokens=" << sampling.max_tokens << " ttft_seconds=" << ttft
                << " prefill_client_tokens_per_second=" << prompt_tokens / ttft
                << " decode_seconds=" << decode
                << " decode_tokens_per_second=" << (sampling.max_tokens - 1) / decode
                << " tpot_seconds=" << decode / (sampling.max_tokens - 1)
                << " gpu_bytes=" << memory.allocated_bytes
                << " exl3_workspace_bytes=" << memory.exl3_workspace_bytes
                << " gdn_workspace_bytes=" << memory.gdn_workspace_bytes
                << " attention_workspace_bytes=" << memory.attention_workspace_bytes
                << " graph_count=" << memory.graph_count << " graph_nodes=" << memory.graph_nodes
                << " graph_device_bytes=" << memory.graph_device_bytes << std::endl;
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
  const bool profile = Setting("VT_B70_PROFILE", 0) != 0;
  const bool quality = Setting("VT_B70_QUALITY", 0) != 0;
  const bool batch = Setting("VT_B70_BATCH", 0) != 0;
  const bool prefix = Setting("VT_B70_PREFIX", 0) != 0;
  const char* teacher_mode = std::getenv("VT_B70_TEACHER_FORCED");
  const bool teacher = teacher_mode != nullptr;
  const bool teacher_conflicts = teacher && (quality || timing || profile || batch || prefix);
  REQUIRE_FALSE(teacher_conflicts);
  const bool profile_conflicts = profile && (quality || timing || batch || prefix ||
      Setting("VT_XPU_PROFILE", 0) != 1 ||
      (Setting("VT_XPU_GRAPH", 0) != 0 && Setting("VT_XPU_GRAPH_PROFILE", 0) != 1));
  REQUIRE_FALSE(profile_conflicts);
  if (teacher) {
    const bool valid_mode = std::string(teacher_mode) == "reference" ||
                            std::string(teacher_mode) == "candidate";
    REQUIRE(valid_mode);
  }
  REQUIRE_FALSE((prefix && timing));
  REQUIRE_FALSE((batch && (quality || timing)));
  const int quality_context = Setting("VT_B70_QUALITY_CONTEXT", 0);
  REQUIRE((quality_context == 0 || quality_context == 4096 || quality_context == 32768));
  REQUIRE((quality_context == 0 || quality));
  REQUIRE_FALSE((timing && quality));
  const int timing_outputs = Setting("VT_B70_TIMING_OUTPUT_TOKENS", 0);
  REQUIRE((timing_outputs == 0 || (timing_outputs >= 2 && timing_outputs <= 100)));
  const int profile_outputs = profile ? Setting("VT_B70_PROFILE_OUTPUT_TOKENS", 17) : 0;
  if (profile) { REQUIRE(profile_outputs >= 2); REQUIRE(profile_outputs <= 100); }
  const int requested_prompt = Setting("VT_B70_PROMPT_TOKENS", 0);
  REQUIRE(requested_prompt >= 0); REQUIRE(requested_prompt <= 6656);
  REQUIRE(tokens > 0); REQUIRE(tokens <= 100); REQUIRE(repeats > 0);
  struct GraphCoverage {
    bool enabled;
    int64_t captures, replays;
    ~GraphCoverage() {
      auto& backend = vt::GetBackend(vt::DeviceType::kXPU);
      const auto captured = backend.GraphsCaptured() - captures, replayed = backend.GraphReplays() - replays;
      if (enabled) { CHECK(captured > 0); CHECK(replayed > 0); }
      std::cout << "GRAPH captures=" << captured << " replays=" << replayed << std::endl;
    }
  } graph_coverage{Setting("VT_XPU_GRAPH", 0) == 1, vt::GetBackend(vt::DeviceType::kXPU).GraphsCaptured(),
                    vt::GetBackend(vt::DeviceType::kXPU).GraphReplays()};
  auto& platform = vllm::platforms::CurrentPlatform();
  REQUIRE(platform.device_type() == vt::DeviceType::kXPU);
  CHECK(platform.needs_weight_staging());
  CHECK_FALSE(platform.supports_fa2_attention());
  CHECK(platform.supports_graph_capture());
  CHECK(platform.support_static_graph_mode() == (Setting("VT_XPU_GRAPH", 0) == 1));

  auto params = vllm_model_params_default();
  params.model_path = model;
  params.device = 0;  // auto must resolve to native XPU for this architecture.
  params.language_model_only = 1;
  params.speculative_config = nullptr;
  params.enable_prefix_caching = prefix ? 1 : 2;
  params.max_num_seqs = batch ? 4 : 1;
  params.max_num_batched_tokens = teacher ? 512 : Setting("VT_B70_BATCH_TOKENS", quality_context ? 4096 : quality ? 512 : requested_prompt ? requested_prompt : timing ? 32 : 16);
  REQUIRE(params.max_num_batched_tokens > 0); REQUIRE(params.max_num_batched_tokens <= 6656);
  params.max_model_len = teacher ? 1024 : quality ? quality_context + 640 :
      std::max(128, requested_prompt + std::max({tokens, timing_outputs, profile_outputs, 32}));
  if (batch) { params.max_model_len = 512; params.max_num_batched_tokens = 128; }
  if (prefix) { params.max_num_batched_tokens = 128; params.max_model_len = std::max(params.max_model_len, 512); }
  params.block_size = 16;
  // Hybrid attention/GDN pools also reserve a sentinel block per group.
  params.num_blocks = std::max(32, 2 * ((params.max_model_len + 15) / 16 + 1));
  if (batch) params.num_blocks *= 4;
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
  if (batch) {
    CheckBatch(engine.get());
    CHECK(vt::GetReferenceTierHits() == initialization_hits);
    return;
  }
  if (profile) {
    ProfileSingleRequest(engine.get(), requested_prompt, profile_outputs);
    CHECK(vt::GetReferenceTierHits() == initialization_hits);
    return;
  }
  if (teacher) {
    CheckTeacherForced(engine.get(), model, std::string(teacher_mode) == "candidate");
    CHECK(vt::GetReferenceTierHits() == initialization_hits);
    return;
  }
  if (quality) {
    CheckQuality(engine.get(), quality_context);
    std::cout << "QUALITY_MEMORY gpu_bytes=" << vt::xpu::GetMemoryInfo().allocated_bytes
              << " attention_workspace_bytes=" << vt::xpu::GetMemoryInfo().attention_workspace_bytes << std::endl;
    CHECK(vt::GetReferenceTierHits() == initialization_hits);
    return;
  }
  if (prefix) {
    CheckPrefix(engine.get()); CHECK(vt::GetReferenceTierHits() == initialization_hits);
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
