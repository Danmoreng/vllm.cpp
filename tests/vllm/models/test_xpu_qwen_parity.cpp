// Compare the full B70 run's first four layers (three GDN + one full attention)
// to a CPU sequential reference over the same checkpoint bytes and tokenizer.
// Only the reference config's layer count changes; no weights are reconstructed.
#include <doctest/doctest.h>
#include <nlohmann/json.hpp>
#include "vllm.h"
#include "vt/dtype.h"
#include "vt/xpu_test_helpers.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
struct Blob {
  std::string dtype, file;
  int64_t rows = 0, cols = 0;
  size_t bytes = 0;
};
using Key = std::pair<int64_t, std::string>;
std::map<Key, Blob> Manifest(const fs::path& dir) {
  std::ifstream input(dir / "manifest.tsv");
  REQUIRE(input.good());
  std::map<Key, Blob> blobs;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream row(line);
    int64_t step, layer; std::string stage; Blob blob;
    REQUIRE(bool(row >> step >> layer >> stage >> blob.dtype >> blob.rows >> blob.cols >> blob.bytes >> blob.file));
    if (step == 0 && layer < 4) REQUIRE(blobs.emplace(Key{layer, stage}, blob).second);
  }
  return blobs;
}
std::vector<float> Read(const fs::path& dir, const Blob& blob) {
  std::ifstream file(dir / blob.file, std::ios::binary | std::ios::ate);
  REQUIRE(file.good()); REQUIRE(size_t(file.tellg()) == blob.bytes);
  file.seekg(0);
  std::vector<unsigned char> raw(blob.bytes);
  REQUIRE(bool(file.read(reinterpret_cast<char*>(raw.data()), raw.size())));
  std::vector<float> values(blob.rows * blob.cols);
  REQUIRE((blob.dtype == "f32" || blob.dtype == "bf16" || blob.dtype == "f16"));
  REQUIRE(blob.bytes == values.size() * (blob.dtype == "f32" ? 4 : 2));
  for (size_t i = 0; i < values.size(); ++i) {
    if (blob.dtype == "f32") std::memcpy(&values[i], raw.data() + 4 * i, 4);
    else {
      uint16_t bits; std::memcpy(&bits, raw.data() + 2 * i, 2);
      values[i] = blob.dtype == "bf16" ? vt::BF16ToF32(bits) : vt::F16ToF32(bits);
    }
  }
  return values;
}
}

TEST_CASE("XPU Qwen checkpoint: first input RMSNorm matches CPU at BF16 rounding boundaries") {
  const char* model = std::getenv("VT_B70_MODEL_DIR");
  const char* acts = std::getenv("VT_B70_XPU_ACTS");
  if (!model || !acts) std::exit(77);
  const fs::path root(model), actual(acts);
  const auto blobs = Manifest(actual);
  const auto hidden = Read(actual, blobs.at({-1, "hidden"}));
  const auto residual = Read(actual, blobs.at({-1, "res"}));
  const auto index = vllm::LoadSafetensorsIndex((root / "model.safetensors.index.json").string());
  const std::string name = "model.language_model.layers.0.input_layernorm.weight";
  auto shard = vllm::SafetensorsFile::Open((root / index.at(name)).string());
  const auto& weight = shard.Get(name);
  REQUIRE(weight.dtype == "BF16"); REQUIRE(weight.shape == std::vector<int64_t>{5120});
  std::ifstream config_file(root / "config.json"); nlohmann::json config; config_file >> config;
  const float eps = config.at("text_config").at("rms_norm_eps");
  xpu_test::Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  std::vector<unsigned char> expected;
  for (auto* q : {&cpu.q, &gpu.q}) {
    xpu_test::Buffer x(*q, vt::DType::kBF16, {5, 5120}), r(*q, vt::DType::kBF16, {5, 5120});
    xpu_test::Buffer w(*q, vt::DType::kBF16, {5120}), out(*q, vt::DType::kBF16, {5, 5120});
    x.put(hidden); r.put(residual); w.upload(weight.data);
    vt::RmsNorm(*q, out.tensor, x.tensor, w.tensor, {eps, true}, &r.tensor);
    if (q == &cpu.q) expected = out.download();
    else xpu_test::SameBytes(out.download(), expected);
  }
}

TEST_CASE("XPU Qwen checkpoint: full-run intermediate activations match a real-weight CPU prefix") {
  const char* model_env = std::getenv("VT_B70_MODEL_DIR");
  const char* actual_env = std::getenv("VT_B70_XPU_ACTS");
  if (!model_env || !actual_env) {
    std::cerr << "SKIP: set VT_B70_MODEL_DIR and VT_B70_XPU_ACTS (the full checkpoint test's VT_DUMP_ACT directory).\n";
    std::exit(77);
  }
  const fs::path model = fs::absolute(model_env), actual_dir = fs::absolute(actual_env);
  const fs::path work = actual_dir.parent_path() / ("pr06-cpu-prefix-" + std::to_string(getpid()));
  const fs::path reference_dir = work / "activations", prefix = work / "model";
  REQUIRE(fs::create_directories(reference_dir)); REQUIRE(fs::create_directories(prefix));
  for (const auto& item : fs::directory_iterator(model))
    if (item.is_regular_file() && item.path().filename() != "config.json")
      fs::create_symlink(item.path(), prefix / item.path().filename());
  std::ifstream input(model / "config.json");
  nlohmann::json config; input >> config;
  auto& text = config.at("text_config");
  REQUIRE(text.at("hidden_size") == 5120);
  REQUIRE(text.at("num_hidden_layers") == 64);
  text["num_hidden_layers"] = 4;
  text["layer_types"] = {"linear_attention", "linear_attention", "linear_attention", "full_attention"};
  std::ofstream(prefix / "config.json") << config.dump(2);
  setenv("VT_DUMP_ACT", reference_dir.c_str(), 1);
  setenv("VT_GDN_CHUNKED", "0", 1);
  setenv("VT_GDN_INDEXED_STATE_IO", "1", 1);
  setenv("VT_FUSE_ATTN_PREAMBLE", "0", 1);
  auto params = vllm_model_params_default();
  const std::string prefix_string = prefix.string(); params.model_path = prefix_string.c_str();
  params.device = 1; params.language_model_only = 1; params.speculative_config = nullptr;
  params.enable_prefix_caching = 2; params.max_num_seqs = 1; params.max_num_batched_tokens = 16;
  params.max_model_len = 16; params.block_size = 16; params.num_blocks = 32; params.kv_cache_dtype = "bfloat16";
  std::cout << "CPU PREFIX reference=" << work << std::endl;
  vllm_engine* raw = nullptr;
  REQUIRE_MESSAGE(vllm_engine_load(&params, &raw) == VLLM_OK, std::string(vllm_last_error()));
  std::unique_ptr<vllm_engine, decltype(&vllm_engine_free)> engine(raw, vllm_engine_free);
  auto sampling = vllm_sampling_params_default(); sampling.temperature = 0; sampling.max_tokens = 1; sampling.ignore_eos = 1;
  vllm_completion result{};
  REQUIRE_MESSAGE(vllm_complete(engine.get(), "The capital of France is", &sampling, &result) == VLLM_OK,
                    std::string(vllm_last_error()));
  CHECK(result.prompt_tokens == 5);
  vllm_completion_free(&result);
  engine.reset();
  const auto actual = Manifest(actual_dir), reference = Manifest(reference_dir);
  size_t compared = 0, stream = 0;
  for (const auto& [key, ref] : reference) {
    REQUIRE(actual.contains(key));
    const auto& got = actual.at(key);
    REQUIRE(got.dtype == ref.dtype); REQUIRE(got.rows == ref.rows); REQUIRE(got.cols == ref.cols);
    const auto a = Read(actual_dir, got), b = Read(reference_dir, ref);
    double error = 0, norm = 0, max_error = 0, max_value = 0;
    bool finite = true;
    for (size_t i = 0; i < a.size(); ++i) {
      finite &= std::isfinite(a[i]) && std::isfinite(b[i]);
      const double diff = double(a[i]) - b[i]; error += diff * diff; norm += double(b[i]) * b[i];
      max_error = std::max(max_error, std::abs(diff)); max_value = std::max(max_value, std::abs(double(b[i])));
    }
    REQUIRE(finite);
    const double relative = std::sqrt(error / std::max(norm, 1e-30));
    std::cout << "PARITY layer=" << key.first << " stage=" << key.second << " relative_rms=" << relative
              << " max_abs=" << max_error << std::endl;
    CAPTURE(key.first);
    CAPTURE(key.second);
    // BF16 has seven explicit fraction bits. Use separate contracts for the
    // exact stem, first nonlinear mixer, residual stream and accumulated
    // projection outputs, rather than one tolerance for unlike stages.
    // These bounds allow float transcendental/reduction rounding, not a change
    // of cache dtype or an approximate EXL3 weight decoder.
    constexpr double bf16_unit = 1.0 / 128;
    const bool exact = key.first == -1 || (key.first == 0 &&
        (key.second == "gdn_mixed" || key.second == "gdn_conv" ||
         key.second == "gdn_postconv_q" || key.second == "gdn_postconv_v"));
    const double limit = key.first == 0 && key.second.starts_with("gdn_")
        ? bf16_unit / 128 : (key.second == "res" || key.first == 0)
        ? bf16_unit / 4 : bf16_unit;
    CHECK(relative <= limit);
    // A chained projection may accumulate more than one rounding step at an
    // individual element; retain a separate two-unit peak-error budget.
    CHECK(max_error <= 1e-5 + 2 * bf16_unit * max_value);
    if (exact) CHECK(a == b);
    if (key.second == "hidden" || key.second == "res") ++stream;
    ++compared;
  }
  CHECK(compared >= 20);
  CHECK(stream == 10);  // Embedding + each of the four layer boundaries, both streams.
}
