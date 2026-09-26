#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <oneapi/dnnl/dnnl.hpp>

#include "gptq4_model_bench_metadata.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/backend.h"
#include "vt/xpu.h"
#include "vt/xpu/xpu_gptq4.h"

namespace {

using Clock = std::chrono::steady_clock;

const char* Env(const char* name, const char* fallback = "") {
  const char* value = std::getenv(name);
  return value != nullptr ? value : fallback;
}

std::string TokenHash(const std::vector<int32_t>& tokens) {
  uint64_t hash = 14695981039346656037ull;
  for (int32_t token : tokens) {
    const uint32_t bits = static_cast<uint32_t>(token);
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= static_cast<uint8_t>(bits >> (byte * 8));
      hash *= 1099511628211ull;
    }
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << hash;
  return out.str();
}

struct Resources {
  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  std::vector<void*> allocations;
  void* Zero(size_t bytes) {
    void* data = vt::Alloc(queue.device, bytes);
    vt::GetBackend(queue.device).Memset(queue, data, 0, bytes);
    allocations.push_back(data);
    return data;
  }
  ~Resources() {
    vt::GetBackend(queue.device).Synchronize(queue);
    for (void* data : allocations) vt::Free(queue.device, data);
    vt::DestroyQueue(queue);
  }
};

double Seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

void Emit(const nlohmann::json& record) {
  std::cout << record.dump() << '\n';
  std::cout.flush();
}

nlohmann::json SummarizeHostSpans(
    const std::vector<vt::xpu::HostProfileRecord>& records) {
  nlohmann::json stages = nlohmann::json::object();
  for (const auto& event : records) {
    auto& stage = stages[event.stage];
    if (stage.is_null()) stage = {{"count", 0}, {"host_ms", 0.0}};
    stage["count"] = stage["count"].get<int>() + 1;
    stage["host_ms"] = stage["host_ms"].get<double>() +
        static_cast<double>(event.end_steady_ns - event.start_steady_ns) / 1.0e6;
  }
  return stages;
}

int Run(const std::string& checkpoint, int prompt_tokens, int output_tokens,
        int rounds) {
  if (prompt_tokens < 64 || prompt_tokens > 4096 || output_tokens < 2 ||
      output_tokens > 64 || rounds < 1 || rounds > 20)
    throw std::invalid_argument("expected 64<=prompt<=4096, 2<=output<=64, 1<=rounds<=20");
  constexpr int block_size = 128;
  Resources resources;
  auto& queue = resources.queue;
  auto& backend = vt::GetBackend(queue.device);
  const auto config = vllm::LoadHfConfig(checkpoint + "/config.json");
  std::vector<vllm::SafetensorsFile> shards;
  for (int index = 1; index <= 5; ++index)
    shards.push_back(vllm::SafetensorsFile::Open(
        checkpoint + "/model-0000" + std::to_string(index) +
        "-of-00005.safetensors"));
  auto model = vllm::ModelRegistry::Load(
      config, vllm::ModelSource::FromSafetensors(shards, &queue));
  vllm::ModelRegistry::Prepare(*model, config, queue);

  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  const int blocks = (prompt_tokens + output_tokens + block_size - 2) /
                         block_size + 2;
  for (const std::string& type : config.layer_types) {
    if (type == "linear_attention") {
      const int64_t hv = config.linear_num_value_heads;
      const int64_t dv = config.linear_value_head_dim;
      const int64_t dk = config.linear_key_head_dim;
      const int64_t conv_dim = 2 * config.linear_num_key_heads * dk + hv * dv;
      const int64_t conv_len = config.linear_conv_kernel_dim - 1;
      vllm::GdnStateCache state;
      state.ssm_state = vt::Tensor::Contiguous(
          resources.Zero(static_cast<size_t>(2 * hv * dv * dk) * 4),
          vt::DType::kF32, queue.device, {2, hv, dv, dk});
      state.conv_state = vt::Tensor::Contiguous(
          resources.Zero(static_cast<size_t>(2 * conv_dim * conv_len) * 2),
          vt::DType::kF16, queue.device, {2, conv_dim, conv_len});
      gdn_state.push_back(state);
    } else {
      vllm::PagedKvCache cache;
      cache.num_blocks = blocks;
      cache.data = resources.Zero(static_cast<size_t>(
          blocks * 2 * block_size * config.num_key_value_heads *
          config.head_dim) * 2);
      cache.dtype = vt::DType::kF16;
      cache.block_size = block_size;
      cache.num_kv_heads = config.num_key_value_heads;
      cache.head_size = config.head_dim;
      attn_kv.push_back(cache);
    }
  }
  if (attn_kv.size() + gdn_state.size() != 64)
    throw std::runtime_error("expected 64 language layers");

  std::vector<int32_t> prompt_ids(prompt_tokens), prompt_positions(prompt_tokens);
  std::vector<int32_t> decode_ids(output_tokens - 1);
  for (int token = 0; token < prompt_tokens; ++token) {
    prompt_ids[token] = 100 + token % 11;
    prompt_positions[token] = token;
  }
  for (int step = 0; step < output_tokens - 1; ++step)
    decode_ids[step] = 300 + step;
  const auto prompt_meta = gptq4_model_bench::AttentionMetadata(
      prompt_tokens, 0, block_size);
  const auto prompt_gdn = gptq4_model_bench::GdnMetadata(prompt_tokens, false);
  const std::vector<int32_t> prompt_logits_index{prompt_tokens - 1};
  vllm::ModelForwardInput prompt_input{
      prompt_ids, prompt_positions, prompt_meta, prompt_gdn, attn_kv,
      gdn_state, config, queue, prompt_logits_index};
  prompt_input.num_reqs = 1;

  const auto reset = [&] {
    for (const auto& cache : attn_kv)
      backend.Memset(queue, cache.data, 0,
                     static_cast<size_t>(cache.num_blocks * 2 * block_size *
                                         cache.num_kv_heads * cache.head_size) * 2);
    for (const auto& state : gdn_state) {
      backend.Memset(queue, state.ssm_state.data, 0, state.ssm_state.Bytes());
      backend.Memset(queue, state.conv_state.data, 0, state.conv_state.Bytes());
    }
    backend.Synchronize(queue);
  };
  const auto prefill = [&] {
    const auto result = vllm::ModelRegistry::Forward(*model, prompt_input);
    if (!result.on_device() || result.rows != 1 ||
        result.vocab != config.vocab_size)
      throw std::runtime_error("invalid prefill output");
    backend.Synchronize(queue);
  };
  const auto decode = [&] {
    for (int step = 0; step < output_tokens - 1; ++step) {
      const std::vector<int32_t> ids{decode_ids[step]};
      const std::vector<int32_t> positions{prompt_tokens + step};
      const auto meta = gptq4_model_bench::AttentionMetadata(
          1, prompt_tokens + step, block_size);
      const auto gdn = gptq4_model_bench::GdnMetadata(1, true);
      const std::vector<int32_t> logits_index{0};
      vllm::ModelForwardInput input{
          ids, positions, meta, gdn, attn_kv, gdn_state, config, queue,
          logits_index};
      input.num_reqs = 1;
      input.gdn_state_slots = 2;
      input.pure_decode = true;
      input.uniform_query_len = 1;
      const auto result = vllm::ModelRegistry::Forward(*model, input);
      if (!result.on_device() || result.rows != 1 ||
          result.vocab != config.vocab_size)
        throw std::runtime_error("invalid decode output");
      backend.Synchronize(queue);
    }
  };

  reset();
  prefill();
  decode();
  const bool graph_profile = std::string(Env("VT_XPU_GRAPH_PROFILE")) == "1";
  const bool host_profile = std::string(Env("VT_XPU_HOST_PROFILE")) == "1";
  if (graph_profile || host_profile) {
    (void)vt::xpu::DrainProfileEvents(queue.device.index);
    (void)vt::xpu::DrainHostProfileRecords(queue.device.index);
  }
  const auto warm_stats = vt::xpu::GetGptq4RuntimeStats(queue.device.index);
  const auto warm_memory = vt::xpu::GetMemoryInfo(queue.device.index);
  Emit({{"event", "gptq4_benchmark_config"},
        {"source_base_commit", VLLM_CPP_BENCH_BASE_COMMIT},
        {"build_id", std::string(__DATE__) + " " + __TIME__},
        {"device", nlohmann::json::parse(vt::xpu::DeviceDescription(0))},
        {"onednn_version", {dnnl::version()->major, dnnl::version()->minor,
                              dnnl::version()->patch}},
        {"onednn_source_hash", dnnl::version()->hash != nullptr
                                   ? dnnl::version()->hash : ""},
        {"checkpoint_revision_declared", Env("VLLM_CPP_GPTQ4_CHECKPOINT_REVISION")},
        {"checkpoint", checkpoint},
        {"model_dtype", "f16"}, {"kv_dtype", "f16"},
        {"weights", "gptq4-g128 with dense BA/head"},
        {"prompt_tokens", prompt_tokens}, {"output_tokens", output_tokens},
        {"prompt_ids_fnv1a64", TokenHash(prompt_ids)},
        {"decode_ids_fnv1a64", TokenHash(decode_ids)},
        {"rounds", rounds}, {"warm_blocks", 1},
        {"gdn_chunk_size", 64}, {"kv_block_size", block_size},
        {"gdn_prefill_mode", Env("VT_XPU_GDN_PREFILL", "auto")},
        {"attention_mode", Env("VT_XPU_ATTENTION", "auto")},
        {"graph_opt_in", std::string(Env("VT_GPTQ4_GRAPH")) == "1"},
        {"graph_profile", graph_profile},
        {"host_profile", host_profile},
        {"measurement_scope", "synchronized full-model forward; load, JIT and warm block excluded"},
        {"warm_primitive_count", warm_stats.primitive_count},
        {"warm_scratch_allocations", warm_stats.scratchpad_allocation_count},
        {"warm_allocated_bytes", warm_memory.allocated_bytes}});

  for (int round = 0; round < rounds; ++round) {
    reset();
    const auto captures_before = backend.GraphsCaptured();
    const auto replays_before = backend.GraphReplays();
    const auto start = Clock::now();
    prefill();
    const auto prefill_end = Clock::now();
    nlohmann::json prefill_host_spans = nullptr;
    if (host_profile && !graph_profile)
      prefill_host_spans = SummarizeHostSpans(
          vt::xpu::DrainHostProfileRecords(queue.device.index));
    const auto decode_start = Clock::now();
    decode();
    const auto decode_end = Clock::now();
    const auto stats = vt::xpu::GetGptq4RuntimeStats(queue.device.index);
    const auto memory = vt::xpu::GetMemoryInfo(queue.device.index);
    const uint64_t replays = backend.GraphReplays() - replays_before;
    const double prefill_seconds = Seconds(start, prefill_end);
    const double decode_seconds = Seconds(decode_start, decode_end);
    nlohmann::json graph_timeline = nullptr;
    if (graph_profile) {
      graph_timeline = nlohmann::json::object();
      for (const auto& event : vt::xpu::DrainProfileEvents(queue.device.index)) {
        if (event.stage != "graph_compute" &&
            event.stage != "graph_validation") continue;
        auto& stage = graph_timeline[event.stage];
        if (stage.is_null()) stage = {{"count", 0}, {"device_ms", 0.0}};
        stage["count"] = stage["count"].get<int>() + 1;
        stage["device_ms"] = stage["device_ms"].get<double>() +
            static_cast<double>(event.end_ns - event.start_ns) / 1.0e6;
      }
      for (const auto& event : vt::xpu::DrainHostProfileRecords(queue.device.index)) {
        auto& stage = graph_timeline[event.stage];
        if (stage.is_null()) stage = {{"count", 0}, {"host_ms", 0.0}};
        stage["count"] = stage["count"].get<int>() + 1;
        stage["host_ms"] = stage["host_ms"].get<double>() +
            static_cast<double>(event.end_steady_ns - event.start_steady_ns) / 1.0e6;
      }
    }
    nlohmann::json decode_host_spans = nullptr;
    if (host_profile && !graph_profile)
      decode_host_spans = SummarizeHostSpans(
          vt::xpu::DrainHostProfileRecords(queue.device.index));
    Emit({{"event", "gptq4_benchmark_round"}, {"round", round},
          {"prefill_seconds", prefill_seconds},
          {"prefill_tokens_per_second", prompt_tokens / prefill_seconds},
          {"decode_seconds", decode_seconds},
          {"decode_tokens_per_second", (output_tokens - 1) / decode_seconds},
          {"decode_forwards", output_tokens - 1},
          {"graph_captures", backend.GraphsCaptured() - captures_before},
          {"graph_replays", replays},
          {"decode_replay_fraction", static_cast<double>(replays) /
                                         (output_tokens - 1)},
          {"primitive_count", stats.primitive_count},
          {"scratch_allocations", stats.scratchpad_allocation_count},
          {"scratch_capacity_bytes", stats.scratchpad_capacity_bytes},
          {"allocated_bytes", memory.allocated_bytes},
          {"peak_allocated_bytes", memory.peak_allocated_bytes},
          {"graph_device_bytes", memory.graph_device_bytes}});
    if (graph_profile)
      Emit({{"event", "gptq4_graph_timeline"}, {"round", round},
            {"stages", graph_timeline},
            {"scope", "graph replay events and host waits; not an operator breakdown"}});
    if (host_profile && !graph_profile)
      Emit({{"event", "gptq4_host_spans"}, {"round", round},
            {"prefill", prefill_host_spans}, {"decode", decode_host_spans},
            {"scope", "nested host spans may overlap; do not sum as disjoint time"}});
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 5) {
    std::cerr << "usage: bench_gptq4_model CHECKPOINT [PROMPT OUTPUT ROUNDS]\n";
    return 2;
  }
  try {
    return Run(argv[1], argc == 5 ? std::stoi(argv[2]) : 512,
               argc == 5 ? std::stoi(argv[3]) : 8,
               argc == 5 ? std::stoi(argv[4]) : 3);
  } catch (const std::exception& error) {
    std::cerr << "bench_gptq4_model: " << error.what() << '\n';
    return 1;
  }
}
