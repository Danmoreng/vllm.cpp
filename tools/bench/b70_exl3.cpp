// Opt-in, resident-weight microbenchmark for every real EXL3 projection family.
// Timings include both Had128 transforms and scratch lifetime, but not uploads.
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/unaligned.h"
#include "vt/xpu.h"
#include "vt/xpu/xpu_exl3_strategy.h"
#include "vt/xpu/xpu_common.h"
#include "b70_exl3_xmx.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <string_view>
#include <tuple>
#include <vector>

namespace {
struct Queue {
  vt::Queue q;
  explicit Queue(vt::DeviceType type = vt::DeviceType::kXPU) : q(vt::CreateQueue({type, 0})) {}
  ~Queue() { vt::DestroyQueue(q); }
};
struct Buffer {
  vt::Queue& q;
  vt::Tensor t;
  Buffer(vt::Queue& queue, vt::DType type, std::initializer_list<int64_t> shape) : q(queue) {
    t = vt::Tensor::Contiguous(nullptr, type, q.device, shape);
    t.data = vt::Alloc(q.device, t.Bytes());
  }
  ~Buffer() { vt::Free(q.device, t.data); }
  Buffer(const Buffer&) = delete;
  void Upload(const void* data) {
    auto& backend = vt::GetBackend(q.device);
    backend.Copy(q, t.data, data, t.Bytes()); backend.Synchronize(q);
  }
};
struct Family { std::string prefix; int count = 0; };

nlohmann::json CheckPanels(vt::Queue& cpu, const std::vector<float>& actual,
    const std::vector<uint16_t>& input, const vllm::StTensor& packed,
    const vllm::StTensor& u, const vllm::StTensor& v, int bits, int64_t m, int64_t k, int64_t n,
    bool approximate) {
  double error = 0, norm = 0, peak_error = 0, peak_value = 0;
  size_t different = 0, elements = 0;
  for (int64_t column : {int64_t{0}, (n / 256) * 128, n - 128}) {
    std::vector<unsigned char> panel(k / 16 * 8 * 32 * bits);
    for (int64_t tile = 0; tile < k / 16; ++tile)
      std::memcpy(panel.data() + tile * 8 * 32 * bits,
          packed.data + (tile * (n / 16) + column / 16) * 32 * bits, 8 * 32 * bits);
    Buffer a(cpu, vt::DType::kF16, {m, k}), ah(cpu, vt::DType::kF16, {m, k});
    Buffer b(cpu, vt::DType::kI8, {k / 16, 8, 32 * bits});
    Buffer suh(cpu, vt::DType::kF16, {k}), svh(cpu, vt::DType::kF16, {128});
    Buffer out(cpu, vt::DType::kF32, {m, 128});
    a.Upload(input.data()); b.Upload(panel.data()); suh.Upload(u.data); svh.Upload(v.data + column * 2);
    vt::Exl3Gemm(cpu, out.t, a.t, b.t, suh.t, svh.t, ah.t, {bits, 2});
    std::vector<float> reference(m * 128);
    vt::GetBackend(cpu.device).Copy(cpu, reference.data(), out.t.data, out.t.Bytes());
    vt::GetBackend(cpu.device).Synchronize(cpu);
    for (int64_t row = 0; row < m; ++row) for (int col = 0; col < 128; ++col) {
      const float got = actual[row * n + column + col], ref = reference[row * 128 + col];
      VT_CHECK(std::isfinite(ref), "Non-finite CPU reference");
      const double diff = double(got) - ref;
      different += std::bit_cast<uint32_t>(got) != std::bit_cast<uint32_t>(ref);
      error += diff * diff; norm += double(ref) * ref;
      peak_error = std::max(peak_error, std::abs(diff)); peak_value = std::max(peak_value, std::abs(double(ref)));
      ++elements;
    }
  }
  const double relative = std::sqrt(error / std::max(norm, 1e-30));
  if (approximate) {
    // XMX uses hardware FP32 reduction order. Retain the operator budget even
    // after qualification by separate model-level answer/probability gates.
    VT_CHECK(relative <= 3e-5 && peak_error <= 2e-6 + 3e-4 * peak_value, "XMX reference error exceeds probe budget");
  } else VT_CHECK(different == 0, "Bit-exact strategy differs from CPU reference");
  return {{"elements", elements}, {"different_bits", different}, {"relative_rms", relative}, {"max_abs", peak_error}};
}
}

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 4) {
      std::cerr << "Usage: b70_exl3_bench MODEL_DIR [M=1] [REPEATS=5]\n";
      return 2;
    }
    const std::filesystem::path root(argv[1]);
    const int m = argc > 2 ? std::stoi(argv[2]) : 1;
    const int repeats = argc > 3 ? std::stoi(argv[3]) : 5;
    VT_CHECK(m > 0 && m <= 6656 && repeats >= 3 && repeats <= 20, "Invalid M or repeat count");
    const auto index = vllm::LoadSafetensorsIndex((root / "model.safetensors.index.json").string());
    std::map<std::string, vllm::SafetensorsFile> files;
    auto get = [&](const std::string& name) -> const vllm::StTensor& {
      const auto& shard = index.at(name);
      if (!files.contains(shard)) files.emplace(shard, vllm::SafetensorsFile::Open((root / shard).string()));
      return files.at(shard).Get(name);
    };
    std::map<std::tuple<int, int64_t, int64_t>, Family> families;
    for (const auto& [name, shard] : index) {
      (void)shard;
      if (!name.ends_with(".trellis") || name.starts_with("mtp.")) continue;
      const auto& packed = get(name);
      VT_CHECK(packed.dtype == "I16" && packed.shape.size() == 3, "Unexpected trellis layout");
      auto& family = families[{packed.shape[2] / 16, packed.shape[0] * 16, packed.shape[1] * 16}];
      if (family.count++ == 0) family.prefix = name.substr(0, name.size() - 8);
    }
    Queue queue, cpu(vt::DeviceType::kCPU); auto& q = queue.q;
    const auto initial_refs = vt::GetReferenceTierHits();
    const char* strategy = std::getenv("VT_XPU_EXL3_STRATEGY");
    const std::string_view strategy_name = strategy ? strategy : "auto";
    const bool matrix_probe = strategy_name == "matrix";
    const auto device = nlohmann::json::parse(vt::xpu::DeviceDescription());
    const vt::xpu::exl3::StrategyDomain domain{device.value("device_id", -1), device.at("driver"),
        device.at("runtime"), device.at("compiler"), vt::xpu::exl3::kKernelVersion};
    std::cout << nlohmann::json{{"event", "device"}, {"device", device},
        {"m", m}, {"repeats", repeats}, {"output_dtype", "f32"}, {"strategy", strategy ? strategy : "auto"},
        {"kernel_version", vt::xpu::exl3::kKernelVersion},
        {"warmup_min_ms", 200}}.dump() << std::endl;
    double weighted_ms = 0;
    for (const auto& [key, family] : families) {
      const auto [bits, k, n] = key;
      const auto& packed = get(family.prefix + ".trellis");
      const auto& u = get(family.prefix + ".suh"); const auto& v = get(family.prefix + ".svh");
      VT_CHECK(vt::LoadUnaligned<uint32_t>(get(family.prefix + ".mul1").data) == 0x83DCD12Du,
               "Benchmark requires the pinned mul1 checkpoint");
      VT_CHECK(u.dtype == "F16" && v.dtype == "F16", "Unexpected EXL3 scale dtype");
      Buffer trellis(q, vt::DType::kI8, {k / 16, n / 16, 32 * bits});
      Buffer suh(q, vt::DType::kF16, {k}), svh(q, vt::DType::kF16, {n});
      Buffer in(q, vt::DType::kF16, {m, k}), had(q, vt::DType::kF16, {m, k});
      Buffer out(q, vt::DType::kF32, {m, n});
      VT_CHECK(trellis.t.Bytes() == packed.nbytes && suh.t.Bytes() == u.nbytes && svh.t.Bytes() == v.nbytes,
               "Checkpoint byte count mismatch");
      trellis.Upload(packed.data); suh.Upload(u.data); svh.Upload(v.data);
      std::vector<uint16_t> input(m * k);
      uint32_t seed = 0x7248135u;
      for (auto& value : input) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        value = vt::F32ToF16((static_cast<int>(seed % 65536) - 32768) / 32768.0f * 0.2f);
      }
      in.Upload(input.data());
      auto run = [&] {
        if (matrix_probe) vt::xpu::Exl3MatrixProbe(q, out.t, in.t, trellis.t, suh.t, svh.t, had.t, bits);
        else vt::Exl3Gemm(q, out.t, in.t, trellis.t, suh.t, svh.t, had.t, {bits, 2});
        vt::GetBackend(q.device).Synchronize(q);
      };
      // First call may JIT. Warm for a duration AFTER it: two tiny operations
      // did not yield stable first-family timings on the B70.
      run();
      const auto warm_start = std::chrono::steady_clock::now();
      int warmups = 1;
      do { run(); ++warmups; }
      while (std::chrono::steady_clock::now() - warm_start < std::chrono::milliseconds(200));
      std::vector<double> times;
      for (int rep = 0; rep < repeats; ++rep) {
        const auto start = std::chrono::steady_clock::now(); run();
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      }
      std::sort(times.begin(), times.end());
      const double median = (times[(times.size() - 1) / 2] + times[times.size() / 2]) / 2;
      weighted_ms += family.count * median;
      Buffer finite(q, vt::DType::kI32, {1});
      auto& backend = vt::GetBackend(q.device);
      backend.Memset(q, finite.t.data, 0, sizeof(int));
      const auto* values = static_cast<const float*>(out.t.data);
      auto* invalid = static_cast<int*>(finite.t.data);
      vt::xpu::NativeQueue(q).parallel_for(sycl::range<1>(m * n), [=](sycl::id<1> i) {
        if (!sycl::isfinite(values[i[0]]))
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
              sycl::access::address_space::global_space>(*invalid).store(1);
      });
      int nonfinite = 0;
      backend.Copy(q, &nonfinite, finite.t.data, sizeof(int));
      backend.Synchronize(q);
      VT_CHECK(nonfinite == 0, "Non-finite benchmark output");
      std::vector<int64_t> rows;
      if (m <= 20) for (int64_t row = 0; row < m; ++row) rows.push_back(row);
      else {
        rows = {0, 1, 15, 16, 31, 32, m / 2, m - 1};
        std::sort(rows.begin(), rows.end());
        rows.erase(std::remove_if(rows.begin(), rows.end(), [&](int64_t row) { return row >= m; }), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
      }
      std::vector<float> result(rows.size() * n);
      std::vector<uint16_t> sampled_input(rows.size() * k);
      for (size_t i = 0; i < rows.size(); ++i) {
        backend.Copy(q, result.data() + i * n, values + rows[i] * n, n * sizeof(float));
        std::copy_n(input.data() + rows[i] * k, k, sampled_input.data() + i * k);
      }
      vt::GetBackend(q.device).Synchronize(q);
      std::string selected(strategy_name);
      if (strategy_name == "auto" || (m < 128 && (strategy_name == "prefill" || strategy_name == "panel"))) {
        const auto choice = vt::xpu::exl3::MeasuredStrategy(domain, {bits, k, n, m, vt::DType::kF32});
        selected = choice == vt::xpu::exl3::Strategy::kPrefill ? "prefill" :
            choice == vt::xpu::exl3::Strategy::kFused ? "fused" : "packed";
      }
      const auto accuracy = CheckPanels(cpu.q, result, sampled_input, packed, u, v, bits, rows.size(), k, n,
          matrix_probe || selected == "prefill");
      const int64_t scratch_rows = matrix_probe && m != 1 ? ((m + 7) / 8) * 8 : m;
      std::cout << nlohmann::json{{"event", "family"}, {"prefix", family.prefix}, {"count", family.count},
          {"bits", bits}, {"k", k}, {"n", n}, {"m", m}, {"median_ms", median},
          {"min_ms", times.front()}, {"max_ms", times.back()}, {"packed_bytes", packed.nbytes},
          {"samples_ms", times}, {"warmups", warmups}, {"selected", selected},
          {"validated_rows", rows}, {"persistent_workspace_bytes", vt::xpu::GetMemoryInfo().exl3_workspace_bytes},
          {"global_gemm_scratch_bytes", selected == "fused" || selected == "prefill" || selected == "panel" ? 0 : scratch_rows * n * 4},
          {"slm_bytes", selected == "fused" ? (m == 1 ? 512 : 2048) :
              (selected == "prefill" ? 12288 :
              (matrix_probe ? (m == 1 ? (32 + 32 * 64) * 2 : (8 * 16 + 16 * 16) * 2) : 0))},
          {"accuracy", accuracy}}.dump() << std::endl;
    }
    VT_CHECK(vt::GetReferenceTierHits() == initial_refs, "CPU reference fallback in benchmark");
    std::cout << nlohmann::json{{"event", "summary"}, {"families", families.size()},
        {"weighted_projection_ms", weighted_ms}, {"note", "sum of isolated family medians, not an engine timing"}}.dump() << std::endl;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
