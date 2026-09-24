#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include "vt/fp8_kv.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <nlohmann/json.hpp>

namespace {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;
std::vector<float> Random(size_t size, uint32_t seed, float scale = 1) {
  std::vector<float> values(size);
  for (auto& v : values) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    v = float(int(seed & 65535) - 32768) / 32768.0f * scale;
  }
  return values;
}
void Accuracy(const std::vector<float>& actual, const std::vector<float>& expected, bool quantized = false, bool matrix = false) {
  REQUIRE(actual.size() == expected.size());
  double error = 0, norm = 0, worst = 0, peak = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i])) FAIL("nonfinite attention output");
    const double delta = actual[i] - expected[i];
    error += delta * delta; norm += double(expected[i]) * expected[i];
    worst = std::max(worst, std::abs(delta)); peak = std::max(peak, std::abs(double(expected[i])));
  }
  const double rms = std::sqrt(error / std::max(1e-30, norm));
  std::cout << "ATTN_ERROR quantized=" << quantized << " rms=" << rms << " max_abs=" << worst << std::endl;
  CHECK(rms <= (quantized ? 0.05 : matrix ? 0.001 : 1e-4));
  CHECK(worst <= 3e-6 + (quantized ? 0.1 : matrix ? 0.002 : 1e-4) * peak);
}
struct Fixture {
  vt::Queue& queue;
  int requests, tokens, context, page = 16, blocks, columns;
  Buffer query, out, cache, table, lens, offsets;
  vt::Tensor kc, vc;
  vt::PagedAttentionArgs args;
  Fixture(vt::Queue& q, int nreq, int chunk, int length, bool fp8, bool unequal = false, int block = 16)
      : queue(q), requests(nreq), tokens(nreq * chunk), context(length), page(block),
        blocks(nreq * ((length + block - 1) / block)), columns((length + block - 1) / block),
        query(q, DType::kF32, {tokens, 24, 256}), out(q, DType::kF32, {tokens, 24, 256}),
        cache(q, fp8 ? DType::kI8 : DType::kBF16, {blocks, 2 * page, 4, 256}),
        table(q, DType::kI32, {requests, 2 * columns + 1}),
        lens(q, DType::kI32, {requests}), offsets(q, DType::kI32, {requests + 1}) {
    std::vector<int32_t> lens_data(requests), qsl{0}, bt_data(requests * (2 * columns + 1), -1);
    for (int r = 0; r < requests; ++r) {
      lens_data[r] = length - (unequal ? r * 17 : 0);
      qsl.push_back(qsl.back() + (unequal && r ? chunk - r : chunk));
      for (int b = 0; b < columns; ++b) bt_data[r * (2 * columns + 1) + 2 * b] = blocks - 1 - r * columns - b;
    }
    tokens = qsl.back();
    auto queries = Random(query.tensor.Numel(), 83175);
    query.put(queries); query.tensor.shape[0] = out.tensor.shape[0] = tokens;
    auto data = Random(cache.tensor.Numel(), 991); // same BF16-rounded source for both cache formats
    for (auto& v : data) v = vt::BF16ToF32(vt::F32ToBF16(v));
    args.scale = 1.0f / 16;
    if (fp8) {
      args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
      args.k_scale = 0.125f; args.v_scale = 0.0625f;
      std::vector<uint8_t> bytes(data.size());
      for (size_t i = 0; i < data.size(); ++i)
        bytes[i] = vt::StoreKvFp8E4M3(data[i], (i / (page * 4 * 256)) % 2 ? args.v_scale : args.k_scale);
      cache.upload(bytes.data());
    } else cache.put(data);
    kc = vt::Tensor::Contiguous(cache.tensor.data, cache.tensor.dtype, q.device, {blocks, page, 4, 256});
    kc.stride[0] *= 2; vc = kc;
    vc.data = static_cast<char*>(kc.data) + page * 4 * 256 * vt::SizeOf(kc.dtype);
    table.upload(bt_data.data()); table.tensor.shape[1] = columns; table.tensor.stride[1] = 2;
    lens.upload(lens_data.data()); offsets.upload(qsl.data()); args.max_seq_len = length;
  }
  void run(const char* mode) {
    setenv("VT_XPU_ATTENTION", mode, 1);
    vt::PagedAttention(queue, out.tensor, query.tensor, kc, vc, table.tensor, lens.tensor, offsets.tensor, args);
    vt::GetBackend(queue.device).Synchronize(queue);
  }
  std::vector<float> result() { auto data = out.floats(); data.resize(tokens * 24 * 256); return data; }
};
}
TEST_CASE("XPU split-KV: 32k, batch4, M2-5, uneven requests and windowed softcap") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (bool fp8 : {false, true}) for (int chunk : {1, 2, 3, 4, 5}) {
    const int nreq = chunk == 5 ? 4 : 1, length = chunk == 1 || chunk == 5 ? 32768 : 4097;
    CAPTURE(fp8);
    CAPTURE(chunk);
    Fixture ref(cpu.q, nreq, chunk, length, fp8, nreq == 4);
    Fixture got(gpu.q, nreq, chunk, length, fp8, nreq == 4);
    ref.run("reference"); got.run("split"); Accuracy(got.result(), ref.result());
    if (chunk == 5) {
      ref.args.causal = got.args.causal = false;
      ref.args.window_size = got.args.window_size = vt::AttentionWindow{129, 3};
      ref.args.logits_soft_cap = got.args.logits_soft_cap = 0.7f;
      ref.run("reference"); got.run("split"); Accuracy(got.result(), ref.result());
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 16 * 1024 * 1024);
}
TEST_CASE("XPU split-KV: E4M3 quantization against BF16 at 32k") {
  Queue gpu(vt::DeviceType::kXPU);
  Fixture bf16(gpu.q, 1, 5, 32768, false), fp8(gpu.q, 1, 5, 32768, true);
  bf16.run("split"); fp8.run("split");
  Accuracy(fp8.result(), bf16.result(), true);
  CHECK(fp8.cache.bytes * 2 == bf16.cache.bytes);
}
TEST_CASE("XPU attention: timing" * doctest::skip(!std::getenv("VT_B70_ATTN_BENCH"))) {
  Queue gpu(vt::DeviceType::kXPU);
  std::cout << "DEVICE " << vt::xpu::DeviceDescription() << std::endl;
  for (int length : {128, 512, 4096, 32768}) for (int chunk : {1, 5}) for (bool fp8 : {false, true}) {
    Fixture f(gpu.q, 1, chunk, length, fp8);
    for (const char* mode : {"reference", "split"}) {
      f.run(mode);
      auto start = std::chrono::steady_clock::now();
      int warmups = 1;  // includes the initial dispatch above
      do { f.run(mode); ++warmups; } while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(100));
      std::vector<double> ms;
      for (int i = 0; i < 5; ++i) {
        start = std::chrono::steady_clock::now(); f.run(mode);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      }
      std::sort(ms.begin(), ms.end());
      std::cout << nlohmann::json{{"context", length}, {"queries", chunk}, {"mode", mode}, {"fp8", fp8},
          {"median_ms", ms[2]}, {"samples_ms", ms}, {"warmups", warmups}, {"cache_bytes", f.cache.bytes},
          {"workspace_bytes", vt::xpu::GetMemoryInfo().attention_workspace_bytes}}.dump() << std::endl;
    }
  }
}

TEST_CASE("XPU XMX attention prefill: tails, batch4, appended 32k context and masks") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (bool fp8 : {false, true}) for (int chunk : {31, 33, 63, 128, 511}) {
    CAPTURE(fp8);
    CAPTURE(chunk);
    const int nreq = chunk == 63 ? 4 : 1, context = chunk == 128 ? 32768 : chunk + 3 + (nreq - 1) * 17;
    Fixture ref(cpu.q, nreq, chunk, context, fp8, nreq == 4);
    Fixture got(gpu.q, nreq, chunk, context, fp8, nreq == 4);
    ref.run("reference"); got.run("prefill"); Accuracy(got.result(), ref.result(), false, true);
    if (chunk == 63) {
      ref.args.causal = got.args.causal = false;
      ref.args.window_size = got.args.window_size = vt::AttentionWindow{13, 7};
      ref.args.logits_soft_cap = got.args.logits_soft_cap = 0.7f;
      ref.run("reference"); got.run("prefill"); Accuracy(got.result(), ref.result(), false, true);
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU XMX attention prefill: FP16 overflow eligibility fallback") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  Fixture ref(cpu.q, 1, 33, 40, false), got(gpu.q, 1, 33, 40, false);
  auto values = Random(33 * 24 * 256, 828, 100000);
  ref.query.put(values); got.query.put(values);
  ref.run("reference"); got.run("prefill"); Accuracy(got.result(), ref.result());
}
TEST_CASE("XPU XMX attention prefill: empty requests and aliased query output") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  Fixture ref(cpu.q, 4, 17, 256, false), got(gpu.q, 4, 17, 256, false);
  const int32_t qsl[] = {0, 0, 17, 17, 66};
  for (auto* f : {&ref, &got}) {
    f->tokens = 66; f->query.tensor.shape[0] = f->out.tensor.shape[0] = 66;
    f->offsets.upload(qsl);
  }
  ref.run("reference");
  setenv("VT_XPU_ATTENTION", "prefill", 1);
  vt::PagedAttention(gpu.q, got.query.tensor, got.query.tensor, got.kc, got.vc,
                      got.table.tensor, got.lens.tensor, got.offsets.tensor, got.args);
  auto actual = got.query.floats(); actual.resize(66 * 24 * 256);
  Accuracy(actual, ref.result(), false, true);
}
TEST_CASE("XPU split-KV: insufficient workspace budget uses native reference"
          * doctest::skip(!std::getenv("VT_B70_LOW_MEMORY_TEST"))) {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  Fixture ref(cpu.q, 1, 5, 128, false), got(gpu.q, 1, 5, 128, false);
  ref.run("reference"); got.run("split"); Accuracy(got.result(), ref.result());
  CHECK(vt::xpu::GetMemoryInfo().attention_workspace_bytes == 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU XMX attention prefill: non-power-of-two pages") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (bool fp8 : {false, true}) {
    Fixture ref(cpu.q, 2, 33, 97, fp8, false, 3), got(gpu.q, 2, 33, 97, fp8, false, 3);
    ref.run("reference"); got.run("prefill"); Accuracy(got.result(), ref.result(), false, true);
  }
}
TEST_CASE("XPU split-KV: workspace survives queue replacement") {
  std::vector<float> first;
  for (int repeat = 0; repeat < 3; ++repeat) {
    Queue gpu(vt::DeviceType::kXPU);
    {
      Fixture got(gpu.q, 1, 5, 513, false); got.run("split");
      if (!repeat) first = got.result(); else CHECK(got.result() == first);
    }
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 16 * 1024 * 1024);
  }
}
TEST_CASE("XPU attention prefill: timing" * doctest::skip(!std::getenv("VT_B70_ATTN_BENCH"))) {
  Queue gpu(vt::DeviceType::kXPU);
  std::cout << "DEVICE " << vt::xpu::DeviceDescription() << std::endl;
  for (int length : {128, 512, 4096}) for (bool fp8 : {false, true}) {
    Fixture f(gpu.q, 1, length, length, fp8);
    std::vector<float> reference;
    for (const char* mode : {"reference", "prefill"}) {
      f.run(mode);
      auto start = std::chrono::steady_clock::now();
      int warmups = 1;  // includes the initial dispatch above
      do { f.run(mode); ++warmups; } while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(100));
      std::vector<double> ms;
      for (int i = 0; i < 5; ++i) {
        start = std::chrono::steady_clock::now(); f.run(mode);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      }
      std::sort(ms.begin(), ms.end());
      std::cout << nlohmann::json{{"context", length}, {"queries", length}, {"mode", mode}, {"fp8", fp8},
          {"median_ms", ms[2]}, {"samples_ms", ms}, {"warmups", warmups}, {"cache_bytes", f.cache.bytes}}.dump() << std::endl;
      if (std::string(mode) == "reference") reference = f.result();
      else Accuracy(f.result(), reference, false, true);
    }
  }
}
