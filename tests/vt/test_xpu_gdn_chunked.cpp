#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include "vt/xpu/xpu_kernels.h"
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <map>
#include <nlohmann/json.hpp>

namespace {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;
constexpr int D = 128;

std::vector<float> Values(int count, int salt, float scale = 0.01f) {
  std::vector<float> data(count);
  uint32_t seed = 0x83175u + salt;
  for (auto& value : data) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    value = (int(seed % 65536) - 32768) / 32768.0f * scale;
  }
  return data;
}
std::vector<float> Normalized(int count, int salt) {
  auto data = Values(count, salt, 1.0f);
  for (int row = 0; row < count; row += D) {
    float norm = 1e-6f;
    for (int d = 0; d < D; ++d) norm += data[row + d] * data[row + d];
    for (int d = 0; d < D; ++d) data[row + d] /= std::sqrt(norm);
  }
  return data;
}
void Accuracy(const std::vector<float>& actual, const std::vector<float>& expected, bool bf16 = false) {
  REQUIRE(actual.size() == expected.size());
  double error = 0, norm = 0, peak_error = 0, peak = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) FAIL("nonfinite GDN output/state");
    const double delta = double(actual[i]) - expected[i];
    error += delta * delta; norm += double(expected[i]) * expected[i];
    peak_error = std::max(peak_error, std::abs(delta)); peak = std::max(peak, std::abs(double(expected[i])));
  }
  const double rms = std::sqrt(error / std::max(norm, 1e-30));
  std::cout << "GDN_ERROR bf16=" << bf16 << " rms=" << rms << " max_abs=" << peak_error << std::endl;
  CHECK(rms <= (bf16 ? 1e-3 : 3e-5));
  CHECK(peak_error <= 2e-6 + (bf16 ? 0.008 : 3e-5) * peak);
}
struct Result { std::vector<float> output, state; };
Result Run(vt::Queue& queue, const std::vector<int32_t>& offsets, int heads, DType output_type,
           bool initial, int split = 0, bool public_call = false, bool zero_decay = false,
           DType input_type = DType::kBF16) {
  const int tokens = offsets.back(), sequences = offsets.size() - 1, kh = heads / 3;
  Buffer q(queue, input_type, {tokens, kh, D}), k(queue, input_type, {tokens, kh, D});
  Buffer v(queue, input_type, {tokens, heads, D}), out(queue, output_type, {tokens, heads, D});
  Buffer g(queue, DType::kF32, {tokens, heads}), beta(queue, DType::kF32, {tokens, heads});
  Buffer state(queue, DType::kF32, {sequences, heads, D, D}), qsl(queue, DType::kI32, {sequences + 1});
  q.put(Normalized(tokens * kh * D, 1)); k.put(Normalized(tokens * kh * D, 2));
  v.put(Values(tokens * heads * D, 3, 0.5f));
  auto gate = Values(tokens * heads, 4, 0.2f), b = Values(tokens * heads, 5, 0.45f);
  for (int i = 0; i < tokens * heads; ++i) {
    gate[i] = zero_decay ? 0 : i % 19 ? -std::abs(gate[i]) : 0;
    b[i] = zero_decay ? float(i % 2) : b[i] + 0.5f;
  }
  g.put(gate); beta.put(b);
  state.put(initial ? Values(sequences * heads * D * D, 6, 0.1f) : std::vector<float>(sequences * heads * D * D));
  qsl.upload(offsets.data());
  if (public_call) {
    vt::GdnPrefill(queue, out.tensor, q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor, state.tensor, qsl.tensor, {0.0883883476f});
  } else if (queue.device.type == vt::DeviceType::kCPU) {
    // Explicit independent oracle: CPU's BF16 chunked algorithm has additional
    // rounding sites and is not the scalar F32 recurrence required here.
    const char* value = std::getenv("VT_GDN_CHUNKED");
    const bool had = value != nullptr; const std::string old = had ? value : "";
    setenv("VT_GDN_CHUNKED", "0", 1);
    vt::GdnPrefill(queue, out.tensor, q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor, state.tensor, qsl.tensor, {0.0883883476f});
    if (had) setenv("VT_GDN_CHUNKED", old.c_str(), 1); else unsetenv("VT_GDN_CHUNKED");
  } else {
    auto rows = [](const vt::Tensor& tensor, int begin, int count) {
      auto view = tensor; view.shape[0] = count;
      view.data = static_cast<char*>(view.data) + begin * view.stride[0] * vt::SizeOf(view.dtype);
      return view;
    };
    if (split) {
      REQUIRE(sequences == 1); REQUIRE(split < tokens);
      const int32_t prefix[] = {0, split}; qsl.upload(prefix);
      auto target = rows(out.tensor, 0, split);
      REQUIRE(vt::xpu::GdnChunkedPrefillKernel(queue, target, rows(q.tensor, 0, split), rows(k.tensor, 0, split),
          rows(v.tensor, 0, split), rows(g.tensor, 0, split), rows(beta.tensor, 0, split), state.tensor, qsl.tensor, {0.0883883476f}));
      for (int token = split; token < tokens; ++token) {
        target = rows(out.tensor, token, 1);
        vt::GdnDecode(queue, target, rows(q.tensor, token, 1), rows(k.tensor, token, 1), rows(v.tensor, token, 1),
                      rows(g.tensor, token, 1), rows(beta.tensor, token, 1), state.tensor, {0.0883883476f});
      }
    } else REQUIRE(vt::xpu::GdnChunkedPrefillKernel(queue, out.tensor, q.tensor, k.tensor, v.tensor,
                        g.tensor, beta.tensor, state.tensor, qsl.tensor, {0.0883883476f}));
  }
  return {out.floats(), state.floats()};
}
}

TEST_CASE("XPU GDN chunk64: F16 XMX output and F32 state match sequential CPU") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int length : {64, 65}) {
    const auto ref = Run(cpu.q, {0, length}, 48, DType::kF32, true,
                         0, false, false, DType::kF16);
    const auto got = Run(gpu.q, {0, length}, 48, DType::kF16, true,
                         0, false, false, DType::kF16);
    Accuracy(got.output, ref.output, true);
    Accuracy(got.state, ref.state);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU GDN short F16 prefill selects chunked kernel"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  const auto ref = Run(cpu.q, {0, 64}, 48, DType::kF32, true,
                       0, false, false, DType::kF16);
  (void)vt::xpu::DrainProfileEvents();
  const auto got = Run(gpu.q, {0, 64}, 48, DType::kF16, true,
                       0, true, false, DType::kF16);
  Accuracy(got.output, ref.output, true);
  Accuracy(got.state, ref.state);
  const auto records = vt::xpu::DrainProfileEvents();
  size_t dots = 0, recurrence = 0;
  for (const auto& record : records) {
    dots += record.stage == "gdn_chunk_dots_qk";
    recurrence += record.stage == "gdn_prefill_recurrence";
  }
  CHECK(dots == 1);
  CHECK(recurrence == 0);
}

TEST_CASE("XPU GDN chunk64: full output and F32 state against sequential CPU") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (bool initial : {false, true}) for (int length : {1, 63, 64, 65, 127, 128, 129}) {
    CAPTURE(length);
    CAPTURE(initial);
    const auto ref = Run(cpu.q, {0, length}, 6, DType::kF32, initial);
    const auto got = Run(gpu.q, {0, length}, 6, DType::kF32, initial);
    Accuracy(got.output, ref.output); Accuracy(got.state, ref.state);
  }
  for (DType dtype : {DType::kF32, DType::kBF16}) {
    const auto ref = Run(cpu.q, {0, 129}, 48, dtype, true);
    const auto got = Run(gpu.q, {0, 129}, 48, dtype, true);
    Accuracy(got.output, ref.output, dtype == DType::kBF16); Accuracy(got.state, ref.state);
  }
  CHECK(vt::xpu::GetMemoryInfo().gdn_workspace_bytes == 16 * 1024 * 1024);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU GDN profile: chunk stages and decode recurrence"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  const auto expected = Run(cpu.q, {0, 65}, 6, DType::kF32, true);
  (void)vt::xpu::DrainProfileEvents();
  const auto actual = Run(gpu.q, {0, 65}, 6, DType::kF32, true, 64);
  Accuracy(actual.output, expected.output);
  Accuracy(actual.state, expected.state);
  const auto records = vt::xpu::DrainProfileEvents();
  std::map<std::string, size_t> counts;
  for (const auto& record : records) {
    ++counts[record.stage];
    CHECK(record.queue_id == gpu.q.id);
    CHECK(record.start_ns > 0);
    CHECK(record.end_ns >= record.start_ns);
  }
  for (const char* stage : {"gdn_chunk_gates", "gdn_chunk_inputs", "gdn_chunk_dots_qk",
                            "gdn_chunk_system", "gdn_chunk_inverse", "gdn_chunk_wu_tile4",
                            "gdn_chunk_delta_cross_tile4", "gdn_chunk_output_tile4", "gdn_chunk_state_tile4",
                            "gdn_decode_recurrence"}) CHECK(counts[stage] == 1);
  CHECK(records.size() == 10);
}

TEST_CASE("XPU GDN chunk64: empty and unequal sequences, long drift, decode continuation") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (const auto& offsets : {std::vector<int32_t>{0, 0, 1, 64, 129}, std::vector<int32_t>{0, 4097}}) {
    const auto ref = Run(cpu.q, offsets, 6, DType::kF32, true);
    const auto got = Run(gpu.q, offsets, 6, DType::kF32, true);
    Accuracy(got.output, ref.output); Accuracy(got.state, ref.state);
    if (offsets[1] == 0) CHECK(std::equal(got.state.begin(), got.state.begin() + 6 * D * D, ref.state.begin()));
  }
  const auto ref = Run(cpu.q, {0, 145}, 6, DType::kF32, true);
  const auto got = Run(gpu.q, {0, 145}, 6, DType::kF32, true, 79);
  Accuracy(got.output, ref.output); Accuracy(got.state, ref.state);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 16 * 1024 * 1024);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU GDN chunk64: persistent state with zero decay and saturated gates") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  const auto ref = Run(cpu.q, {0, 1025}, 6, DType::kF32, true, 0, false, true);
  const auto got = Run(gpu.q, {0, 1025}, 6, DType::kF32, true, 0, false, true);
  Accuracy(got.output, ref.output); Accuracy(got.state, ref.state);
}

TEST_CASE("XPU GDN chunk64: workspace reuse after queue replacement") {
  Result first;
  for (int repeat = 0; repeat < 3; ++repeat) {
    Queue gpu(vt::DeviceType::kXPU);
    const auto got = Run(gpu.q, {0, 65}, 6, DType::kF32, true);
    if (!repeat) first = got;
    else { CHECK(got.output == first.output); CHECK(got.state == first.state); }
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 16 * 1024 * 1024);
  }
}

TEST_CASE("XPU GDN chunk64: insufficient workspace budget uses native reference"
          * doctest::skip(!std::getenv("VT_B70_LOW_MEMORY_TEST"))) {
  const char* mode = std::getenv("VT_XPU_GDN_PREFILL");
  REQUIRE(mode != nullptr); REQUIRE(std::string(mode) == "chunked");
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  REQUIRE(vt::xpu::GetMemoryInfo().budget_bytes == 8 * 1024 * 1024);
  const auto ref = Run(cpu.q, {0, 65}, 6, DType::kF32, true);
  const auto got = Run(gpu.q, {0, 65}, 6, DType::kF32, true, 0, true);
  Accuracy(got.output, ref.output); Accuracy(got.state, ref.state);
  CHECK(vt::xpu::GetMemoryInfo().gdn_workspace_bytes == 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU GDN chunk64: timing" * doctest::skip(!std::getenv("VT_B70_GDN_BENCH"))) {
  setenv("VT_GDN_CHUNKED", "0", 1);  // public call remains the native sequential comparator
  Queue gpu(vt::DeviceType::kXPU); auto& queue = gpu.q;
  auto& backend = vt::GetBackend(queue.device);
  constexpr int heads = 48, kh = 16;
  for (int tokens : {128, 512, 2048, 4096}) {
    Buffer q(queue, DType::kBF16, {tokens, kh, D}), k(queue, DType::kBF16, {tokens, kh, D});
    Buffer v(queue, DType::kBF16, {tokens, heads, D}), out(queue, DType::kBF16, {tokens, heads, D});
    Buffer g(queue, DType::kF32, {tokens, heads}), beta(queue, DType::kF32, {tokens, heads});
    Buffer state(queue, DType::kF32, {1, heads, D, D}), qsl(queue, DType::kI32, {2});
    q.put(Normalized(tokens * kh * D, 1)); k.put(Normalized(tokens * kh * D, 2));
    v.put(Values(tokens * heads * D, 3, 0.5f));
    g.put(std::vector<float>(tokens * heads, -0.13f)); beta.put(std::vector<float>(tokens * heads, 0.61f));
    const int32_t offsets[] = {0, tokens}; qsl.upload(offsets);
    Result reference;
    for (bool chunked : {false, true}) {
      auto run = [&] {
        if (chunked) REQUIRE(vt::xpu::GdnChunkedPrefillKernel(queue, out.tensor, q.tensor, k.tensor,
              v.tensor, g.tensor, beta.tensor, state.tensor, qsl.tensor, {0.0883883476f}));
        else vt::GdnPrefill(queue, out.tensor, q.tensor, k.tensor, v.tensor, g.tensor,
                            beta.tensor, state.tensor, qsl.tensor, {0.0883883476f});
        backend.Synchronize(queue);
      };
      auto reset = [&] { backend.Memset(queue, state.tensor.data, 0, state.bytes); backend.Synchronize(queue); };
      reset(); run();
      auto start = std::chrono::steady_clock::now();
      do { reset(); run(); } while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(200));
      std::vector<double> times;
      for (int i = 0; i < 5; ++i) {
        reset(); start = std::chrono::steady_clock::now(); run();
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      }
      std::sort(times.begin(), times.end());
      std::cout << nlohmann::json{{"tokens", tokens}, {"chunked", chunked}, {"median_ms", times[2]},
          {"samples_ms", times}, {"workspace_bytes", vt::xpu::GetMemoryInfo().gdn_workspace_bytes}}.dump() << std::endl;
      Result got{out.floats(), state.floats()};
      if (!chunked) reference = got;
      else { Accuracy(got.output, reference.output, true); Accuracy(got.state, reference.state); }
    }
  }
}
