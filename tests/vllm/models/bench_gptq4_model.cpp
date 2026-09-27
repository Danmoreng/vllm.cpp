#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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

struct DecodeStepTiming {
  int context_before;
  int input_token;
  double seconds;
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

nlohmann::json StageTrace(const std::vector<vt::xpu::ProfileRecord>& records) {
  nlohmann::json events = nlohmann::json::array();
  for (const auto& record : records)
    events.push_back({{"stage", record.stage}, {"matrix", record.matrix},
        {"stream_span", record.stream_span}, {"submit_ns", record.submit_ns},
        {"start_ns", record.start_ns}, {"end_ns", record.end_ns}});
  return events;
}

int Run(const std::string& checkpoint, int prompt_tokens, int output_tokens,
        int rounds) {
  // The native Xe2 GDN route splits 8K/16K prefill into bounded 4K macros.
  // Larger prompts still need an explicit scheduler-chunked benchmark path.
  if (prompt_tokens < 1 || prompt_tokens > 16384 || output_tokens < 1 ||
      rounds < 1 || rounds > 20)
    throw std::invalid_argument("expected 1<=prompt<=16384, generated>=1, 1<=rounds<=20");
  const std::string_view kv_dtype = Env("VT_B70_BENCH_KV_DTYPE", "f16");
  if (kv_dtype != "f16" && kv_dtype != "fp8_e4m3")
    throw std::invalid_argument("VT_B70_BENCH_KV_DTYPE must be f16 or fp8_e4m3");
  const bool fp8_kv = kv_dtype == "fp8_e4m3";
  const std::string prefill_chunk_arg = Env("VT_B70_BENCH_PREFILL_CHUNK");
  const int prefill_chunk = prefill_chunk_arg.empty()
      ? prompt_tokens : std::stoi(prefill_chunk_arg);
  if (prefill_chunk < 1 || prefill_chunk > prompt_tokens)
    throw std::invalid_argument("VT_B70_BENCH_PREFILL_CHUNK must be in 1..prompt_tokens");
  const int block_size = std::stoi(Env("VT_B70_BENCH_BLOCK_SIZE", fp8_kv ? "1600" : "128"));
  if (block_size != 64 && block_size != 128 && block_size != 1600)
    throw std::invalid_argument("VT_B70_BENCH_BLOCK_SIZE must be 64, 128 or 1600");
  Resources resources;
  auto& queue = resources.queue;
  auto& backend = vt::GetBackend(queue.device);
  const auto config = vllm::LoadHfConfig(checkpoint + "/config.json");
  const std::string max_context_arg = Env("VT_B70_BENCH_MAX_CONTEXT");
  const int64_t max_context = max_context_arg.empty()
      ? config.max_position_embeddings : std::stoll(max_context_arg);
  const int64_t total_context = static_cast<int64_t>(prompt_tokens) +
      output_tokens - 1;
  const int64_t required_model_len = total_context + 1;
  if (max_context < 1 || max_context > config.max_position_embeddings ||
      max_context > std::numeric_limits<int32_t>::max() ||
      required_model_len > max_context)
    throw std::invalid_argument("prompt + generated exceeds benchmark or model context limit");
  if (config.vocab_size <= 300)
    throw std::invalid_argument("benchmark synthetic token stream needs vocabulary > 300");
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
  const int blocks = static_cast<int>((total_context + block_size - 1) /
                                      block_size + 2);
  const size_t attention_layers = static_cast<size_t>(std::count_if(
      config.layer_types.begin(), config.layer_types.end(),
      [](const std::string& type) { return type != "linear_attention"; }));
  const size_t gdn_layers = config.layer_types.size() - attention_layers;
  const uint64_t kv_bytes = static_cast<uint64_t>(attention_layers) * blocks *
      2 * block_size * config.num_key_value_heads * config.head_dim *
      (fp8_kv ? 1 : 2);
  const uint64_t conv_dim = 2 * config.linear_num_key_heads *
      config.linear_key_head_dim + config.linear_num_value_heads *
      config.linear_value_head_dim;
  const uint64_t gdn_bytes = static_cast<uint64_t>(gdn_layers) * 2 *
      (config.linear_num_value_heads * config.linear_value_head_dim *
           config.linear_key_head_dim * 4 +
       conv_dim * (config.linear_conv_kernel_dim - 1) * 2);
  const auto before_cache = vt::xpu::GetMemoryInfo(queue.device.index);
  const uint64_t budget_free = before_cache.budget_bytes > before_cache.allocated_bytes
      ? before_cache.budget_bytes - before_cache.allocated_bytes : 0;
  const uint64_t available = before_cache.free_known
      ? std::min<uint64_t>(budget_free, before_cache.free_bytes) : budget_free;
  constexpr uint64_t kWorkspaceReserve = 1ull << 30;
  if (kv_bytes + gdn_bytes > available ||
      kWorkspaceReserve > available - (kv_bytes + gdn_bytes))
    throw std::runtime_error("benchmark KV/state cache does not fit measured GPU budget with 1 GiB workspace reserve");
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
      cache.data = resources.Zero(static_cast<size_t>(blocks) * 2 * block_size *
          config.num_key_value_heads * config.head_dim * (fp8_kv ? 1 : 2));
      cache.dtype = fp8_kv ? vt::DType::kI8 : vt::DType::kF16;
      if (fp8_kv) {
        cache.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
        cache.k_scale = 1.0f;
        cache.v_scale = 1.0f;
      }
      cache.block_size = block_size;
      cache.num_kv_heads = config.num_key_value_heads;
      cache.head_size = config.head_dim;
      attn_kv.push_back(cache);
    }
  }
  if (attn_kv.size() + gdn_state.size() != 64)
    throw std::runtime_error("expected 64 language layers");

  std::vector<int32_t> prompt_ids(prompt_tokens);
  std::vector<int32_t> decode_ids(output_tokens - 1);
  for (int token = 0; token < prompt_tokens; ++token)
    prompt_ids[token] = 100 + token % 11;
  const std::string prompt_ids_file = Env("VT_B70_BENCH_PROMPT_IDS_FILE");
  if (!prompt_ids_file.empty()) {
    std::ifstream file(prompt_ids_file);
    if (!file) throw std::runtime_error("cannot open prompt IDs: " + prompt_ids_file);
    const auto ids = nlohmann::json::parse(file).get<std::vector<int32_t>>();
    if (ids.size() != static_cast<size_t>(prompt_tokens) ||
        std::any_of(ids.begin(), ids.end(), [&](int32_t id) {
          return id < 0 || id >= config.vocab_size;
        }))
      throw std::runtime_error("prompt IDs must match prompt length and vocabulary");
    prompt_ids = ids;
  }
  for (int step = 0; step < output_tokens - 1; ++step)
    decode_ids[step] = 300 + step % (config.vocab_size - 300);
  if (const char* first = std::getenv("VT_B70_BENCH_DECODE_FIRST_TOKEN");
      first && !decode_ids.empty())
    decode_ids[0] = std::stoi(first);
  const std::string decode_ids_file = Env("VT_B70_BENCH_DECODE_IDS_FILE");
  if (!decode_ids_file.empty()) {
    std::ifstream file(decode_ids_file);
    if (!file) throw std::runtime_error("cannot open decode IDs: " + decode_ids_file);
    const auto ids = nlohmann::json::parse(file).get<std::vector<int32_t>>();
    if (ids.size() != decode_ids.size() ||
        std::any_of(ids.begin(), ids.end(), [&](int32_t id) {
          return id < 0 || id >= config.vocab_size;
        }))
      throw std::runtime_error("decode IDs must match output length - 1 and vocabulary");
    decode_ids = ids;
  }
  const std::string quality_dir = Env("VT_B70_BENCH_QUALITY_DIR");
  const std::string quality_steps_arg = Env("VT_B70_BENCH_QUALITY_STEPS");
  if (quality_dir.empty() && !quality_steps_arg.empty())
    throw std::invalid_argument("VT_B70_BENCH_QUALITY_STEPS requires VT_B70_BENCH_QUALITY_DIR");
  std::vector<int> quality_steps;
  if (!quality_dir.empty() && !quality_steps_arg.empty()) {
    if (quality_steps_arg.back() == ',')
      throw std::invalid_argument("quality decode steps must not end with a comma");
    std::istringstream stream(quality_steps_arg);
    std::string item;
    while (std::getline(stream, item, ',')) {
      size_t parsed = 0;
      const int step = std::stoi(item, &parsed);
      if (parsed != item.size())
        throw std::invalid_argument("quality decode steps must be integers");
      if (step < 1 || step >= output_tokens)
        throw std::invalid_argument("quality decode step must be in 1..generated-1");
      quality_steps.push_back(step);
    }
    std::sort(quality_steps.begin(), quality_steps.end());
    if (quality_steps.empty() ||
        std::adjacent_find(quality_steps.begin(), quality_steps.end()) !=
            quality_steps.end())
      throw std::invalid_argument("quality decode steps must be nonempty and unique");
  } else if (!quality_dir.empty() && output_tokens > 1) {
    quality_steps.push_back(1);
  }
  const auto capture_logits = [&](const vllm::ForwardLogits& logits,
                                  const std::string& phase) {
    if (quality_dir.empty()) return;
    if (!logits.on_device() || logits.rows != 1 ||
        logits.vocab != config.vocab_size ||
        logits.device_tensor.dtype != vt::DType::kF32)
      throw std::runtime_error("quality capture requires one F32 device logits row");
    std::vector<float> values(static_cast<size_t>(logits.vocab));
    backend.Copy(queue, values.data(), logits.device_tensor.data,
                 values.size() * sizeof(float));
    backend.Synchronize(queue);
    const std::string path = quality_dir + "/p" +
        std::to_string(prompt_tokens) + "_" + phase + ".f32";
    std::ofstream file(path, std::ios::binary);
    if (!file.write(reinterpret_cast<const char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(float))))
      throw std::runtime_error("cannot write quality logits: " + path);
  };
  const auto reset = [&] {
    for (const auto& cache : attn_kv)
      backend.Memset(queue, cache.data, 0,
                     static_cast<size_t>(cache.num_blocks * 2 * block_size *
                                         cache.num_kv_heads * cache.head_size) *
                         vt::SizeOf(cache.dtype));
    for (const auto& state : gdn_state) {
      backend.Memset(queue, state.ssm_state.data, 0, state.ssm_state.Bytes());
      backend.Memset(queue, state.conv_state.data, 0, state.conv_state.Bytes());
    }
    backend.Synchronize(queue);
  };
  struct PrefillStepTiming { int before, after; double seconds; };
  const auto prefill = [&](bool capture = false,
                           std::vector<PrefillStepTiming>* steps = nullptr) {
    for (int context = 0; context < prompt_tokens; context += prefill_chunk) {
      const int query_len = std::min(prefill_chunk, prompt_tokens - context);
      const auto step_start = Clock::now();
      std::vector<int32_t> positions(query_len);
      for (int token = 0; token < query_len; ++token)
        positions[token] = context + token;
      const std::vector<int32_t> ids(prompt_ids.begin() + context,
                                     prompt_ids.begin() + context + query_len);
      const auto meta = gptq4_model_bench::AttentionMetadata(
          query_len, context, block_size);
      const auto gdn = gptq4_model_bench::GdnMetadata(
          query_len, false, context > 0);
      const std::vector<int32_t> logits_index{query_len - 1};
      vllm::ModelForwardInput input{
          ids, positions, meta, gdn, attn_kv,
          gdn_state, config, queue, logits_index};
      input.num_reqs = 1;
      const auto result = vllm::ModelRegistry::Forward(*model, input);
      if (!result.on_device() || result.rows != 1 ||
          result.vocab != config.vocab_size)
        throw std::runtime_error("invalid prefill output");
      backend.Synchronize(queue);
      const auto step_end = Clock::now();
      if (steps != nullptr)
        steps->push_back({context, context + query_len,
                          Seconds(step_start, step_end)});
      if (capture && context + query_len == prompt_tokens)
        capture_logits(result, "prefill");
    }
  };
  const auto decode = [&](bool capture = false,
                          std::vector<DecodeStepTiming>* steps = nullptr) {
    for (int step = 0; step < output_tokens - 1; ++step) {
      const auto step_start = Clock::now();
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
      const auto step_end = Clock::now();
      if (steps != nullptr)
        steps->push_back({prompt_tokens + step, decode_ids[step],
                          Seconds(step_start, step_end)});
      if (capture && std::binary_search(quality_steps.begin(),
                                       quality_steps.end(), step + 1))
        capture_logits(result, quality_steps_arg.empty()
            ? "decode" : "decode_" + std::to_string(step + 1));
    }
  };

  reset();
  prefill();
  decode();
  const bool graph_profile = std::string(Env("VT_XPU_GRAPH_PROFILE")) == "1";
  const bool host_profile = std::string(Env("VT_XPU_HOST_PROFILE")) == "1";
  const bool stage_profile = std::string(Env("VT_B70_BENCH_STAGE_PROFILE")) == "1";
  if (stage_profile && std::string(Env("VT_XPU_PROFILE")) != "1")
    throw std::invalid_argument("VT_B70_BENCH_STAGE_PROFILE requires VT_XPU_PROFILE=1");
  if (graph_profile || host_profile || stage_profile) {
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
        {"model_dtype", "f16"}, {"kv_dtype", kv_dtype},
        {"weights", "gptq4-g128 with dense BA/head"},
        {"prompt_tokens", prompt_tokens}, {"output_tokens", output_tokens},
        {"prefill_chunk_tokens", prefill_chunk},
        {"prefill_model_calls", (prompt_tokens + prefill_chunk - 1) / prefill_chunk},
        {"generated_tokens", output_tokens},
        {"decode_forward_steps", output_tokens - 1},
        {"max_context", max_context},
        {"required_model_len", required_model_len},
        {"kv_cache_bytes", kv_bytes}, {"gdn_state_bytes", gdn_bytes},
        {"available_gpu_bytes_before_cache", available},
        {"prompt_ids_fnv1a64", TokenHash(prompt_ids)},
        {"decode_ids_fnv1a64", TokenHash(decode_ids)},
        {"decode_ids_file", decode_ids_file},
        {"decode_first_token", decode_ids.empty()
            ? nlohmann::json(nullptr) : nlohmann::json(decode_ids[0])},
        {"quality_capture", !quality_dir.empty()},
        {"quality_decode_steps", quality_steps},
        {"rounds", rounds}, {"warm_blocks", 1},
        {"gdn_chunk_size", 64}, {"kv_block_size", block_size},
        {"gdn_prefill_mode", Env("VT_XPU_GDN_PREFILL", "auto")},
        {"attention_mode", Env("VT_XPU_ATTENTION", "auto")},
        {"graph_opt_in", std::string(Env("VT_GPTQ4_GRAPH")) == "1"},
        {"graph_profile", graph_profile},
        {"host_profile", host_profile},
        {"measurement_scope", quality_dir.empty()
            ? "synchronized forward including host input/metadata preparation and upload; load, JIT, reset excluded"
            : "quality logits capture included; timing invalid"},
        {"warm_primitive_count", warm_stats.primitive_count},
        {"warm_scratch_allocations", warm_stats.scratchpad_allocation_count},
        {"warm_allocated_bytes", warm_memory.allocated_bytes}});

  for (int round = 0; round < rounds; ++round) {
    reset();
    std::vector<PrefillStepTiming> prefill_steps;
    std::vector<DecodeStepTiming> decode_steps;
    decode_steps.reserve(static_cast<size_t>(output_tokens - 1));
    const auto captures_before = backend.GraphsCaptured();
    const auto replays_before = backend.GraphReplays();
    const auto start = Clock::now();
    prefill(!quality_dir.empty(), &prefill_steps);
    const auto prefill_end = Clock::now();
    if (stage_profile)
      Emit({{"event", "gptq4_stage_trace"}, {"phase", "prefill"}, {"round", round},
            {"events", StageTrace(vt::xpu::DrainProfileEvents(queue.device.index))}});
    nlohmann::json prefill_host_spans = nullptr;
    if (host_profile && !graph_profile)
      prefill_host_spans = SummarizeHostSpans(
          vt::xpu::DrainHostProfileRecords(queue.device.index));
    const auto decode_start = Clock::now();
    decode(!quality_dir.empty(), &decode_steps);
    const auto decode_end = Clock::now();
    if (stage_profile)
      Emit({{"event", "gptq4_stage_trace"}, {"phase", "decode"}, {"round", round},
            {"events", StageTrace(vt::xpu::DrainProfileEvents(queue.device.index))}});
    const auto stats = vt::xpu::GetGptq4RuntimeStats(queue.device.index);
    const auto memory = vt::xpu::GetMemoryInfo(queue.device.index);
    const uint64_t replays = backend.GraphReplays() - replays_before;
    const double prefill_seconds = Seconds(start, prefill_end);
    const double decode_seconds = Seconds(decode_start, decode_end);
    const int decode_forwards = output_tokens - 1;
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
          {"decode_seconds", decode_forwards ? nlohmann::json(decode_seconds)
                                               : nlohmann::json(nullptr)},
          {"decode_tokens_per_second", decode_forwards
              ? nlohmann::json(decode_forwards / decode_seconds)
              : nlohmann::json(nullptr)},
          {"decode_forwards", decode_forwards},
          {"decode_forward_steps", decode_forwards},
          {"generated_tokens", output_tokens},
          {"graph_captures", backend.GraphsCaptured() - captures_before},
          {"graph_replays", replays},
          {"decode_replay_fraction", decode_forwards
              ? nlohmann::json(static_cast<double>(replays) / decode_forwards)
              : nlohmann::json(nullptr)},
          {"primitive_count", stats.primitive_count},
          {"scratch_allocations", stats.scratchpad_allocation_count},
          {"scratch_capacity_bytes", stats.scratchpad_capacity_bytes},
          {"allocated_bytes", memory.allocated_bytes},
          {"peak_allocated_bytes", memory.peak_allocated_bytes},
          {"graph_device_bytes", memory.graph_device_bytes}});
    for (const auto& step : prefill_steps)
      Emit({{"event", "gptq4_benchmark_step"}, {"round", round},
            {"phase", "prefill"}, {"query_tokens", step.after - step.before},
            {"context_before", step.before}, {"context_after", step.after},
            {"seconds", step.seconds}});
    for (size_t step = 0; step < decode_steps.size(); ++step) {
      const auto& timing = decode_steps[step];
      Emit({{"event", "gptq4_benchmark_step"}, {"round", round},
            {"phase", "decode"}, {"decode_forward_index", step},
            {"query_tokens", 1}, {"context_before", timing.context_before},
            {"context_after", timing.context_before + 1},
            {"input_token", timing.input_token},
            {"seconds", timing.seconds}});
    }
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
    std::cerr << "usage: bench_gptq4_model CHECKPOINT [PROMPT GENERATED ROUNDS]\n";
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
