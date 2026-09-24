#include <doctest/doctest.h>
#include "vllm.h"
#include "vllm/platforms/interface.h"
#include "vt/ops.h"
#include "vt/xpu.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
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
void MeasureWarm(vllm_engine* engine) {
  // This is an opt-in, short end-to-end diagnostic, not a context-length sweep.
  // The no-MTP DELTA API emits exactly one callback per generated token.
  REQUIRE(std::getenv("VT_DUMP_ACT") == nullptr);
  REQUIRE(std::getenv("VT_DUMP_ACT_SUB") == nullptr);
  REQUIRE(std::getenv("VT_OP_PROVIDER_TRACE") == nullptr);
  std::string longer;
  for (int i = 0; i < 32; ++i) longer += " Hello";
  for (const std::string& prompt : {std::string("The capital of France is"), longer}) {
    auto sampling = vllm_sampling_params_default();
    sampling.temperature = 0; sampling.ignore_eos = 1; sampling.max_tokens = 2;
    vllm_completion warm{};
    REQUIRE_MESSAGE(vllm_complete(engine, prompt.c_str(), &sampling, &warm) == VLLM_OK,
                    std::string(vllm_last_error()));
    const int prompt_tokens = warm.prompt_tokens;
    CHECK(warm.completion_tokens == 2);
    vllm_completion_free(&warm);
    REQUIRE(prompt_tokens > 0); REQUIRE(prompt_tokens <= 32);
    sampling.max_tokens = prompt_tokens == 5 ? 17 : 2;
    for (int round = 0; round < 2; ++round) {
      TokenTimes times;
      const auto start = std::chrono::steady_clock::now();
      REQUIRE_MESSAGE(vllm_complete_stream(engine, prompt.c_str(), &sampling,
                        TokenTimes::Callback, &times) == VLLM_OK, std::string(vllm_last_error()));
      REQUIRE(times.finished);
      REQUIRE(times.tokens.size() == size_t(sampling.max_tokens));
      const double ttft = std::chrono::duration<double>(times.tokens.front() - start).count();
      const double decode = std::chrono::duration<double>(times.tokens.back() - times.tokens.front()).count();
      std::cout << "TIMING round=" << round << " prompt_tokens=" << prompt_tokens
                << " output_tokens=" << sampling.max_tokens << " ttft_seconds=" << ttft
                << " prefill_client_tokens_per_second=" << prompt_tokens / ttft
                << " decode_seconds=" << decode
                << " decode_tokens_per_second=" << (sampling.max_tokens - 1) / decode
                << " tpot_seconds=" << decode / (sampling.max_tokens - 1) << std::endl;
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
  // The acceptance run generates 65 tokens: one prefill + 64 decode forwards.
  // Smaller explicit values are development probes, not the PR06 acceptance.
  const int tokens = Setting("VT_B70_MAX_TOKENS", 65), repeats = Setting("VT_B70_REPEATS", 2);
  const bool timing = Setting("VT_B70_TIMING", 0) != 0;
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
  params.max_num_batched_tokens = timing ? 32 : 16;
  params.max_model_len = 128;
  params.block_size = 16;
  params.num_blocks = 32;  // Shared hybrid KV groups need more than ceil(128/16).
  params.kv_cache_dtype = "bfloat16";
  std::cout << "LOAD " << model << " device=xpu" << std::endl;
  vllm_engine* raw = nullptr;
  auto status = vllm_engine_load(&params, &raw);
  REQUIRE_MESSAGE(status == VLLM_OK, std::string(vllm_last_error()));
  std::unique_ptr<vllm_engine, decltype(&vllm_engine_free)> engine(raw, vllm_engine_free);
  std::cout << "LOADED gpu_bytes=" << vt::xpu::GetMemoryInfo().allocated_bytes
            << " rss_bytes=" << ResidentBytes() << " initialization_reference_hits=" << vt::GetReferenceTierHits() << std::endl;
  const auto initialization_hits = vt::GetReferenceTierHits();
  if (timing) {
    MeasureWarm(engine.get());
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
    status = vllm_complete(engine.get(), "The capital of France is", &sampling, &result);
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
