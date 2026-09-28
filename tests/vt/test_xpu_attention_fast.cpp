#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include "vt/fp8_kv.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string_view>
#include <utility>
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
  Fixture(vt::Queue& q, int nreq, int chunk, int length, bool fp8, bool unequal = false,
          int block = 16, DType query_type = DType::kF32,
          DType output_type = DType::kF32, DType cache_type = DType::kBF16,
          bool unit_scales = false)
      : queue(q), requests(nreq), tokens(nreq * chunk), context(length), page(block),
        blocks(nreq * ((length + block - 1) / block)), columns((length + block - 1) / block),
        query(q, query_type, {tokens, 24, 256}), out(q, output_type, {tokens, 24, 256}),
        cache(q, fp8 ? DType::kI8 : cache_type, {blocks, 2 * page, 4, 256}),
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
    for (auto& v : data)
      v = cache_type == DType::kF16 ? vt::F16ToF32(vt::F32ToF16(v))
                                    : vt::BF16ToF32(vt::F32ToBF16(v));
    args.scale = 1.0f / 16;
    if (fp8) {
      args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
      args.k_scale = unit_scales ? 1.0f : 0.125f;
      args.v_scale = unit_scales ? 1.0f : 0.0625f;
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
    if (fp8 && chunk == 5 && std::getenv("VT_XPU_PROFILE"))
      (void)vt::xpu::DrainProfileEvents();
    ref.run("reference"); got.run("split"); Accuracy(got.result(), ref.result());
    if (fp8 && chunk == 5 && std::getenv("VT_XPU_PROFILE")) {
      int split_events = 0, reduce_events = 0;
      for (const auto& event : vt::xpu::DrainProfileEvents()) {
        split_events += event.stage == "attention_split_partial";
        reduce_events += event.stage == "attention_split_reduce_cooperative";
      }
      CHECK(split_events == 1);
      CHECK(reduce_events == 1);
    }
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
TEST_CASE("XPU FP8 split-K: padded block tables retain active-page plan"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue gpu(vt::DeviceType::kXPU);
  for (int page : {1600, 1664}) {
    Fixture f(gpu.q, 1, 1, 4096, true, false, page,
              DType::kF16, DType::kF16, DType::kF16, true);
    for (int cols : {3, 6, 164}) {
      CAPTURE(page);
      CAPTURE(cols);
      Buffer table(gpu.q, DType::kI32, {1, cols});
      std::vector<int32_t> ids(cols, 0);
      ids[0] = 2; ids[1] = 1; ids[2] = 0;
      table.upload(ids.data());
      auto run = [&](const char* mode) {
        setenv("VT_XPU_ATTENTION", mode, 1);
        vt::PagedAttention(gpu.q, f.out.tensor, f.query.tensor, f.kc, f.vc,
                           table.tensor, f.lens.tensor, f.offsets.tensor, f.args);
        vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
      };
      run("reference");
      const auto expected = f.result();
      (void)vt::xpu::DrainProfileEvents();
      run("auto");
      Accuracy(f.result(), expected, false, true);
      int split = 0;
      for (const auto& event : vt::xpu::DrainProfileEvents())
        split += event.stage == "attention_split_partial";
      CHECK(split == 1);
      if (cols == 164) {
        setenv("VT_XPU_ATTN_SPLIT_ACTIVE_PAGE_CAP", "0", 1);
        run("auto");
        Accuracy(f.result(), expected, false, true);
        unsetenv("VT_XPU_ATTN_SPLIT_ACTIVE_PAGE_CAP");
      }
    }
  }
}
TEST_CASE("XPU FP8 attention: 20-query split boundary at 4K context"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue gpu(vt::DeviceType::kXPU);
  for (int queries : {20, 21, 31, 32, 33}) {
    CAPTURE(queries);
    Fixture f(gpu.q, 1, queries, 4096 + queries, true, false, 1600,
              DType::kF16, DType::kF16, DType::kF16, true);
    f.run("reference");
    const auto expected = f.result();
    (void)vt::xpu::DrainProfileEvents();
    f.run("auto");
    Accuracy(f.result(), expected, false, true);
    int split = 0, q32 = 0, fallback = 0;
    for (const auto& event : vt::xpu::DrainProfileEvents()) {
      split += event.stage == "attention_split_partial";
      q32 += event.stage == "attention_prefill_q32";
      fallback += event.stage == "attention_reference";
    }
    CHECK(split + q32 + fallback == 1);
    CHECK((queries <= 31 ? split == 1 : q32 == 1));
    if (queries == 21) {
      setenv("VT_XPU_ATTN_SPLIT_EXTENDED", "0", 1);
      (void)vt::xpu::DrainProfileEvents();
      f.run("auto");
      int opted_out = 0;
      for (const auto& event : vt::xpu::DrainProfileEvents())
        opted_out += event.stage == "attention_reference";
      CHECK(opted_out == 1);
      unsetenv("VT_XPU_ATTN_SPLIT_EXTENDED");
    }
    if (std::getenv("VT_B70_ATTN_BENCH")) {
      std::vector<double> samples;
      for (int repeat = 0; repeat < 5; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        f.run("auto");
        samples.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
      }
      std::sort(samples.begin(), samples.end());
      std::cout << nlohmann::json{{"event", "fp8_attention_boundary"},
          {"query_tokens", queries}, {"context_tokens", 4096 + queries},
          {"route", split ? "split" : q32 ? "q32" : "reference"},
          {"median_ms", samples[2]}, {"samples_ms", samples}}.dump()
                << std::endl;
    }
  }
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
TEST_CASE("XPU short F16 attention prefill selects XMX path"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  Fixture ref(cpu.q, 1, 64, 67, false, false, 16,
              DType::kF16, DType::kF32, DType::kF16);
  Fixture got(gpu.q, 1, 64, 67, false, false, 16,
              DType::kF16, DType::kF16, DType::kF16);
  ref.run("reference");
  (void)vt::xpu::DrainProfileEvents();
  got.run("auto");
  Accuracy(got.result(), ref.result(), false, true);
  const auto records = vt::xpu::DrainProfileEvents();
  size_t prefill = 0, fallback = 0;
  for (const auto& record : records) {
    prefill += record.stage == "attention_prefill_q64";
    fallback += record.stage == "attention_reference";
  }
  CHECK(prefill == 1);
  CHECK(fallback == 0);
}
TEST_CASE("XPU Q32/Q64 F16 attention prefill: page boundaries, continuation and masks") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int chunk : {31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512}) {
    CAPTURE(chunk);
    const int page = chunk == 129 || chunk == 257 ? 3 : 16;
    const int context = chunk + 17;
    Fixture ref(gpu.q, 1, chunk, context, false, false, page,
                DType::kF16, DType::kF16, DType::kF16);
    Fixture got(gpu.q, 1, chunk, context, false, false, page,
                DType::kF16, DType::kF16, DType::kF16);
    ref.run("reference");
    if (chunk == 127) {
      setenv("VT_XPU_ATTN_PREFILL_TILE", "q16", 1);
      got.run("prefill");
      Accuracy(got.result(), ref.result(), false, true);
    }
    setenv("VT_XPU_ATTN_PREFILL_TILE", "q32", 1);
    got.run("prefill");
    Accuracy(got.result(), ref.result(), false, true);
    setenv("VT_XPU_ATTN_PREFILL_TILE", "q64", 1);
    got.run("prefill");
    Accuracy(got.result(), ref.result(), false, true);
    if (chunk == 127 || chunk == 257) {
      ref.args.causal = got.args.causal = false;
      ref.args.window_size = got.args.window_size = vt::AttentionWindow{37, 5};
      ref.args.logits_soft_cap = got.args.logits_soft_cap = 0.7f;
      ref.run("reference"); got.run("prefill");
      Accuracy(got.result(), ref.result(), false, true);
    }
    if (chunk == 127) {
      Fixture independent(cpu.q, 1, chunk, context, false, false, page,
                          DType::kF16, DType::kF32, DType::kF16);
      independent.run("reference");
      ref.args.causal = true;
      ref.args.window_size.reset();
      ref.args.logits_soft_cap = 0;
      ref.run("reference");
      Accuracy(ref.result(), independent.result(), false, true);
    }
  }
  unsetenv("VT_XPU_ATTN_PREFILL_TILE");
}
TEST_CASE("XPU Q32/Q64 F16 attention prefill: ragged batch and aliased output") {
  Queue gpu(vt::DeviceType::kXPU);
  Fixture ref(gpu.q, 4, 63, 129, false, true, 3,
              DType::kF16, DType::kF16, DType::kF16);
  Fixture got(gpu.q, 4, 63, 129, false, true, 3,
              DType::kF16, DType::kF16, DType::kF16);
  ref.run("reference");
  setenv("VT_XPU_ATTN_PREFILL_TILE", "q32", 1);
  got.run("prefill");
  Accuracy(got.result(), ref.result(), false, true);
  setenv("VT_XPU_ATTN_PREFILL_TILE", "q64", 1);
  got.run("prefill");
  Accuracy(got.result(), ref.result(), false, true);
  ref.args.causal = got.args.causal = false;
  ref.args.window_size = got.args.window_size = vt::AttentionWindow{33, 7};
  ref.args.logits_soft_cap = got.args.logits_soft_cap = 0.7f;
  ref.run("reference"); got.run("prefill");
  Accuracy(got.result(), ref.result(), false, true);

  Fixture aliased(gpu.q, 1, 33, 49, false, false, 16,
                  DType::kF16, DType::kF16, DType::kF16);
  Fixture alias_ref(gpu.q, 1, 33, 49, false, false, 16,
                    DType::kF16, DType::kF16, DType::kF16);
  alias_ref.run("reference");
  setenv("VT_XPU_ATTN_PREFILL_TILE", "q32", 1);
  setenv("VT_XPU_ATTENTION", "prefill", 1);
  vt::PagedAttention(gpu.q, aliased.query.tensor, aliased.query.tensor,
                     aliased.kc, aliased.vc, aliased.table.tensor,
                     aliased.lens.tensor, aliased.offsets.tensor, aliased.args);
  Accuracy(aliased.query.floats(), alias_ref.result(), false, true);
  unsetenv("VT_XPU_ATTN_PREFILL_TILE");
}
TEST_CASE("XPU Q32/Q64 F16 attention prefill: BF16 and E4M3 cache") {
  Queue gpu(vt::DeviceType::kXPU);
  for (bool fp8 : {false, true}) {
    CAPTURE(fp8);
    Fixture ref(gpu.q, 1, 129, 146, fp8, false, 16,
                DType::kF16, DType::kF16, DType::kBF16);
    Fixture got(gpu.q, 1, 129, 146, fp8, false, 16,
                DType::kF16, DType::kF16, DType::kBF16);
    ref.run("reference");
    setenv("VT_XPU_ATTN_PREFILL_TILE", "q32", 1);
    got.run("prefill");
    Accuracy(got.result(), ref.result(), false, true);
    setenv("VT_XPU_ATTN_PREFILL_TILE", "q64", 1);
    got.run("prefill");
    Accuracy(got.result(), ref.result(), false, true);
  }
  unsetenv("VT_XPU_ATTN_PREFILL_TILE");
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

TEST_CASE("XPU FP8 KV auto selects fast prefill and decode"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue gpu(vt::DeviceType::kXPU);
  Fixture prefill(gpu.q, 1, 64, 67, true, false, 64,
                  DType::kF16, DType::kF16, DType::kF16);
  prefill.run("prefill");
  const auto prefill_expected = prefill.result();
  (void)vt::xpu::DrainProfileEvents();
  prefill.run("auto");
  Accuracy(prefill.result(), prefill_expected, false, true);
  bool saw_prefill = false;
  for (const auto& event : vt::xpu::DrainProfileEvents())
    saw_prefill |= event.stage == "attention_prefill_q64";
  CHECK(saw_prefill);

  Fixture decode(gpu.q, 1, 1, 513, true, false, 64,
                 DType::kF16, DType::kF16, DType::kF16);
  decode.run("split");
  const auto decode_expected = decode.result();
  (void)vt::xpu::DrainProfileEvents();
  decode.run("auto");
  Accuracy(decode.result(), decode_expected);
  bool saw_split = false;
  for (const auto& event : vt::xpu::DrainProfileEvents())
    saw_split |= event.stage == "attention_split_partial";
  CHECK(saw_split);
}

TEST_CASE("XPU FP8 split span candidate"
          * doctest::skip(!std::getenv("VT_B70_ATTENTION_SPLIT_CANDIDATE"))) {
  Queue gpu(vt::DeviceType::kXPU);
  for (int length : {64, 129, 513, 4097}) {
    Fixture f(gpu.q, 1, 1, length, true, false, 64,
              DType::kF16, DType::kF16, DType::kF16);
    std::vector<float> reference;
    for (const char* span : {"256", "128", "64", "32"}) {
      setenv("VT_XPU_ATTN_SPLIT_SPAN", span, 1);
      f.run("split");
      std::vector<double> ms;
      for (int i = 0; i < 15; ++i) {
        const auto start = std::chrono::steady_clock::now();
        f.run("split");
        ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
      }
      std::sort(ms.begin(), ms.end());
      std::cout << nlohmann::json{{"length", length}, {"span", span},
                                  {"median_ms", ms[7]},
                                  {"samples_ms", ms}}.dump() << std::endl;
      if (reference.empty()) reference = f.result();
      else Accuracy(f.result(), reference, false, true);
    }
  }
}


// Focused measurement scaffold for the d35295682f FP8 prefill review.
// This is deliberately separate from the existing F32/page16 timing matrix.
// No model, generic path, tolerance or production dispatch is changed.
TEST_CASE("XPU FP8 prefill: focused F16 P64 baseline"
          * doctest::skip(!std::getenv("VT_B70_FP8_PREFILL_FOCUS"))) {
  const std::string length_text = std::getenv("VT_B70_FP8_PREFILL_FOCUS");
  REQUIRE((length_text == "512" || length_text == "4096"));
  const int length = length_text == "512" ? 512 : 4096;
  // Run in an isolated test process. Set these before creating the queue.
  const char* tile = std::getenv("VT_XPU_ATTN_PREFILL_TILE");
  const char* probability = std::getenv("VT_XPU_ATTN_PROBABILITY");
  REQUIRE(tile != nullptr);
  REQUIRE(probability != nullptr);
  REQUIRE(std::string(tile) == "q64");
  REQUIRE(std::string(probability) == "single");
  const char* profile_setting = std::getenv("VT_XPU_PROFILE");
  const bool profiling = profile_setting && std::string(profile_setting) == "1";
  Queue gpu(vt::DeviceType::kXPU);
  Fixture f(gpu.q, 1, length, length, true, false, 64,
            DType::kF16, DType::kF16, DType::kF16);

  // Dense table columns, reversed physical pages. Buffer::upload copies the
  // allocation size, so keep a full-sized host buffer despite the smaller view.
  std::vector<int32_t> table_data(f.table.bytes / sizeof(int32_t), -1);
  for (int i = 0; i < f.columns; ++i) table_data[i] = f.blocks - 1 - i;
  f.table.upload(table_data.data());
  f.table.tensor.stride[0] = f.columns;
  f.table.tensor.stride[1] = 1;
  // Match the current model benchmark's unit scales. Re-encode once, outside
  // timing, rather than changing the interpretation of already encoded bytes.
  f.args.k_scale = f.args.v_scale = 1.0f;
  auto values = Random(f.cache.tensor.Numel(), 991);
  std::vector<uint8_t> encoded(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    encoded[i] = vt::StoreKvFp8E4M3(vt::F16ToF32(vt::F32ToF16(values[i])), 1.0f);
  f.cache.upload(encoded.data());
  std::cout << "DEVICE " << vt::xpu::DeviceDescription() << std::endl;

  // One generic GPU reference, not repeated generic-reference warmups.
  f.run("reference");
  const auto reference = f.result();
  (void)vt::xpu::DrainProfileEvents();
  f.run("prefill"); // initial dispatch/JIT, outside samples
  Accuracy(f.result(), reference, false, true); // same-cache strict matrix gate
  auto events = vt::xpu::DrainProfileEvents();
  if (profiling) {
    int count = 0;
    for (const auto& event : events) count += event.stage == "attention_prefill_q64";
    REQUIRE(count == 1); // never silently benchmark a generic fallback
  }
  const auto warm_start = std::chrono::steady_clock::now();
  int warmups = 1;
  // Bounded screen, not a claim of clocks/thermal stabilization.
  while (warmups < 32 && std::chrono::steady_clock::now() - warm_start <
                             std::chrono::milliseconds(100)) {
    f.run("prefill");
    ++warmups;
    (void)vt::xpu::DrainProfileEvents();
  }
  std::vector<double> wall_ms, kernel_ms;
  for (int sample = 0; sample < 5; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    f.run("prefill");
    wall_ms.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count());
    events = vt::xpu::DrainProfileEvents(); // outside the wall-clock sample
    if (profiling) {
      int count = 0;
      double elapsed = 0;
      for (const auto& event : events) if (event.stage == "attention_prefill_q64") {
        REQUIRE(event.end_ns >= event.start_ns);
        elapsed += double(event.end_ns - event.start_ns) / 1.0e6;
        ++count;
      }
      REQUIRE(count == 1);
      kernel_ms.push_back(elapsed);
    }
  }
  auto sorted = wall_ms;
  std::sort(sorted.begin(), sorted.end());
  std::cout << nlohmann::json{
      {"event", "fp8_prefill_focus"}, {"context", length}, {"queries", length},
      {"query_dtype", "float16"}, {"output_dtype", "float16"},
      {"kv_dtype", "fp8_e4m3"}, {"page_size", 64},
      {"table_column_stride", f.table.tensor.stride[1]},
      {"k_page_stride_elements", f.kc.stride[0]},
      {"k_token_stride_elements", f.kc.stride[1]},
      {"k_head_stride_elements", f.kc.stride[2]},
      {"v_page_stride_elements", f.vc.stride[0]},
      {"k_scale", f.args.k_scale}, {"v_scale", f.args.v_scale},
      {"tile", tile}, {"probability", probability}, {"mode", "prefill"},
      {"profiling", profiling}, {"dispatch_verified", profiling},
      {"scope", "one operator; wall includes synchronization/status, kernel is event time"},
      {"warmups", warmups}, {"wall_samples_ms", wall_ms},
      {"wall_median_ms", sorted[2]}, {"kernel_samples_ms", kernel_ms},
      {"cache_bytes", f.cache.bytes}}.dump() << std::endl;
  Accuracy(f.result(), reference, false, true);
  CHECK(vt::GetReferenceTierHits() == 0); // generic reference above is a GPU kernel
}

TEST_CASE("XPU Xe2 FP8 prefill: default dispatch with 1600/1664-token pages"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE") ||
                          std::getenv("VT_XPU_XE2_PREFILL"))) {
  Queue gpu(vt::DeviceType::kXPU);
  for (int page : {1600, 1664}) {
    CAPTURE(page);
    Fixture reference(gpu.q, 1, 4096, 4096, true, false, page,
                      DType::kF16, DType::kF16, DType::kF16, true);
    Fixture selected(gpu.q, 1, 4096, 4096, true, false, page,
                     DType::kF16, DType::kF16, DType::kF16, true);
    Buffer contiguous_table(gpu.q, DType::kI32, {1, 3});
    const int32_t page_ids[] = {2, 1, 0};
    contiguous_table.upload(page_ids);
    setenv("VT_XPU_XE2_PREFILL", "0", 1);
    reference.run("prefill");
    unsetenv("VT_XPU_XE2_PREFILL");
    (void)vt::xpu::DrainProfileEvents();
    setenv("VT_XPU_ATTENTION", "auto", 1);
    vt::PagedAttention(gpu.q, selected.out.tensor, selected.query.tensor,
                       selected.kc, selected.vc, contiguous_table.tensor,
                       selected.lens.tensor, selected.offsets.tensor, selected.args);
    vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
    Accuracy(selected.result(), reference.result(), false, true);
    const auto events = vt::xpu::DrainProfileEvents();
    size_t xe2 = 0, fallback = 0;
    for (const auto& event : events) {
      xe2 += event.stage == "attention_prefill_xe2";
      fallback += event.stage == "attention_prefill_q64";
    }
    CHECK(xe2 == 1);
    CHECK(fallback == 0);
  }
}

TEST_CASE("XPU Xe2 FP8 prefill: 2K-8K ragged initial prompt with paged KV"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE") ||
                          std::getenv("VT_XPU_XE2_PREFILL"))) {
  Queue gpu(vt::DeviceType::kXPU);
  for (int tokens : {2048, 2049, 2050, 3071, 3072, 3073,
                     4095, 4096, 4097, 6143, 6144, 6145, 8191, 8192})
      for (int page : {64, 1600}) {
    CAPTURE(tokens);
    CAPTURE(page);
    Fixture f(gpu.q, 1, tokens, tokens, true, false, page,
              DType::kF16, DType::kF16, DType::kF16, true);
    const int pages = (tokens + page - 1) / page;
    Buffer contiguous_table(gpu.q, DType::kI32, {1, pages});
    std::vector<int32_t> page_ids(pages);
    for (int b = 0; b < pages; ++b) page_ids[b] = pages - 1 - b;
    contiguous_table.upload(page_ids.data());
    auto run = [&](const char* mode) {
      setenv("VT_XPU_ATTENTION", mode, 1);
      vt::PagedAttention(gpu.q, f.out.tensor, f.query.tensor, f.kc, f.vc,
                         contiguous_table.tensor, f.lens.tensor, f.offsets.tensor,
                         f.args);
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
    };
    setenv("VT_XPU_XE2_PREFILL", "0", 1);
    run("prefill");
    const auto expected = f.result();
    unsetenv("VT_XPU_XE2_PREFILL");
    (void)vt::xpu::DrainProfileEvents();
    run("auto");
    Accuracy(f.result(), expected, false, true);
    int xe2 = 0, fallback = 0;
    for (const auto& event : vt::xpu::DrainProfileEvents()) {
      xe2 += event.stage == "attention_prefill_xe2";
      fallback += event.stage == "attention_prefill_q64";
    }
    CHECK(xe2 == 1);
    CHECK(fallback == 0);
  }
}

TEST_CASE("XPU Xe2 FP8 prefill: ragged continuation through 8K KV"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE") ||
                          std::getenv("VT_XPU_XE2_PREFILL"))) {
  Queue gpu(vt::DeviceType::kXPU);
  for (auto [queries, prefix] : {std::pair{2048, 63},
                                 std::pair{2048, 2048},
                                 std::pair{2048, 4096},
                                 std::pair{2049, 4096},
                                 std::pair{3071, 4096},
                                 std::pair{3072, 1600},
                                 std::pair{3072, 4096},
                                 std::pair{3073, 4096},
                                 std::pair{4095, 4096},
                                 std::pair{4096, 4096}})
      for (int page : {64, 1600}) {
    const int context = prefix + queries;
    CAPTURE(queries);
    CAPTURE(prefix);
    CAPTURE(page);
    Fixture f(gpu.q, 1, queries, context, true, false, page,
              DType::kF16, DType::kF16, DType::kF16, true);
    const int pages = (context + page - 1) / page;
    Buffer contiguous_table(gpu.q, DType::kI32, {1, pages});
    std::vector<int32_t> page_ids(pages);
    for (int b = 0; b < pages; ++b) page_ids[b] = pages - 1 - b;
    contiguous_table.upload(page_ids.data());
    auto run = [&](const char* mode) {
      setenv("VT_XPU_ATTENTION", mode, 1);
      vt::PagedAttention(gpu.q, f.out.tensor, f.query.tensor, f.kc, f.vc,
                         contiguous_table.tensor, f.lens.tensor, f.offsets.tensor,
                         f.args);
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
    };
    setenv("VT_XPU_XE2_PREFILL", "0", 1);
    run("prefill");
    const auto expected = f.result();
    unsetenv("VT_XPU_XE2_PREFILL");
    (void)vt::xpu::DrainProfileEvents();
    run("auto");
    Accuracy(f.result(), expected, false, true);
    int xe2 = 0, fallback = 0;
    for (const auto& event : vt::xpu::DrainProfileEvents()) {
      xe2 += event.stage == "attention_prefill_xe2";
      fallback += event.stage == "attention_prefill_q64";
    }
    CHECK(xe2 == 1);
    CHECK(fallback == 0);
    if (page == 1600) {
      setenv("VT_XPU_XE2_CONTINUATION", "0", 1);
      (void)vt::xpu::DrainProfileEvents();
      run("auto");
      int opted_out = 0;
      for (const auto& event : vt::xpu::DrainProfileEvents())
        opted_out += event.stage == "attention_prefill_q64";
      CHECK(opted_out == 1);
      unsetenv("VT_XPU_XE2_CONTINUATION");
    }
    if (std::getenv("VT_B70_ATTN_BENCH") &&
        ((queries == 3072 && prefix == 4096) ||
         (queries == 2048 && prefix == 4096) ||
         (queries == 4096 && prefix == 4096))) {
      std::vector<double> old_ms, xe2_ms;
      for (int repeat = 0; repeat < 5; ++repeat) {
        setenv("VT_XPU_XE2_PREFILL", "0", 1);
        auto start = std::chrono::steady_clock::now();
        run("prefill");
        old_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
        unsetenv("VT_XPU_XE2_PREFILL");
        start = std::chrono::steady_clock::now();
        run("auto");
        xe2_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
      }
      std::sort(old_ms.begin(), old_ms.end());
      std::sort(xe2_ms.begin(), xe2_ms.end());
      std::cout << nlohmann::json{{"event", "xe2_continuation_benchmark"},
          {"query_tokens", queries}, {"context_tokens", context}, {"page", page},
          {"q64_median_ms", old_ms[2]}, {"xe2_median_ms", xe2_ms[2]},
          {"q64_samples_ms", old_ms}, {"xe2_samples_ms", xe2_ms}}.dump()
                << std::endl;
    }
  }
}

TEST_CASE("XPU FP8 prefill: captured 4096-token Python operator replay"
          * doctest::skip(!std::getenv("VT_B70_FP8_PREFILL_REPLAY_DIR"))) {
  const std::string directory = std::getenv("VT_B70_FP8_PREFILL_REPLAY_DIR");
  auto read = [&](const std::string& name, size_t size) {
    std::ifstream file(directory + "/" + name, std::ios::binary);
    REQUIRE(file.good());
    std::vector<unsigned char> bytes(size);
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    REQUIRE(file.gcount() == static_cast<std::streamsize>(size));
    REQUIRE(file.peek() == std::char_traits<char>::eof());
    return bytes;
  };
  const auto metadata_bytes = [&] {
    std::ifstream file(directory + "/metadata.json", std::ios::binary);
    REQUIRE(file.good());
    return nlohmann::json::parse(file);
  }();
  const int tokens = metadata_bytes.at("q_original_shape").at(0).get<int>();
  const int heads = metadata_bytes.at("q_original_shape").at(1).get<int>();
  const int dim = metadata_bytes.at("q_original_shape").at(2).get<int>();
  const int source_pages = metadata_bytes.at("files").at("k").at("shape").at(0).get<int>();
  const int source_page = metadata_bytes.at("files").at("k").at("shape").at(1).get<int>();
  const char* xe2_setting = std::getenv("VT_XPU_XE2_PREFILL");
  const bool xe2 = !xe2_setting || std::string_view(xe2_setting) == "1";
  const int page = xe2 ? 64 : source_page;
  const int pages = (tokens + page - 1) / page;
  const int physical_pages = xe2 ? pages + 7 : pages;
  const int kv_heads = metadata_bytes.at("files").at("k").at("shape").at(2).get<int>();
  REQUIRE(tokens == 4096);
  REQUIRE(heads == 24);
  REQUIRE(dim == 256);
  REQUIRE(kv_heads == 4);
  REQUIRE(source_pages * source_page >= tokens);
  REQUIRE(metadata_bytes.at("seqused_k").get<int>() == tokens);
  REQUIRE(metadata_bytes.at("causal").get<bool>());
  REQUIRE(metadata_bytes.at("softcap").get<int>() == 0);
  REQUIRE(metadata_bytes.at("window_size") == nlohmann::json::array({-1, -1}));
  REQUIRE(metadata_bytes.at("k_original_stride").at(1).get<int>() == 2 * kv_heads * dim);
  REQUIRE(metadata_bytes.at("k_original_stride").at(2).get<int>() == 2 * dim);
  REQUIRE(metadata_bytes.at("v_original_stride") == metadata_bytes.at("k_original_stride"));
  for (const char* name : {"k_descale", "v_descale"})
    for (const auto& row : metadata_bytes.at(name))
      for (const auto& scale : row) REQUIRE(scale.get<float>() == 1.0f);

  const size_t q_bytes = size_t(tokens) * heads * dim * sizeof(uint16_t);
  const size_t source_kv_bytes = size_t(source_pages) * source_page * kv_heads * dim;
  const size_t kv_bytes = size_t(physical_pages) * page * kv_heads * dim;
  const auto q_host = read("q.bin", q_bytes);
  const auto k_host = read("k.bin", source_kv_bytes);
  const auto v_host = read("v.bin", source_kv_bytes);
  const auto expected_host = read("output.bin", q_bytes);
  // VT's public paged-attention contract requires head-contiguous K/V views.
  // Repack the same FP8 values into VT's [K page, V page] allocation layout.
  // This checks arithmetic on identical values; it does not make the memory
  // transaction pattern identical to Python's LBNHC layout.
  std::vector<unsigned char> kv_host(kv_bytes * 2);
  const size_t page_bytes = size_t(page) * kv_heads * dim;
  const size_t token_bytes = size_t(kv_heads) * dim;
  for (int token = 0; token < tokens; ++token) {
    const int physical = xe2 ? physical_pages - 1 - token / page : token / page;
    const size_t source_offset = size_t(token / source_page) * source_page * token_bytes +
                                 size_t(token % source_page) * token_bytes;
    const size_t dest_offset = size_t(physical) * 2 * page_bytes +
                               size_t(token % page) * token_bytes;
    std::memcpy(kv_host.data() + dest_offset, k_host.data() + source_offset, token_bytes);
    std::memcpy(kv_host.data() + dest_offset + page_bytes,
                v_host.data() + source_offset, token_bytes);
  }

  Queue gpu(vt::DeviceType::kXPU);
  Buffer query(gpu.q, DType::kF16, {tokens, heads, dim});
  Buffer output(gpu.q, DType::kF16, {tokens, heads, dim});
  Buffer cache(gpu.q, DType::kI8, {physical_pages, 2 * page, kv_heads, dim});
  Buffer table(gpu.q, DType::kI32, {1, pages});
  Buffer lengths(gpu.q, DType::kI32, {1});
  Buffer offsets(gpu.q, DType::kI32, {2});
  query.upload(q_host.data());
  cache.upload(kv_host.data());
  std::vector<int32_t> page_ids(pages);
  std::iota(page_ids.begin(), page_ids.end(), 0);
  if (xe2) for (int& id : page_ids) id = physical_pages - 1 - id;
  table.upload(page_ids.data());
  const int32_t length_data[] = {tokens}, offset_data[] = {0, tokens};
  lengths.upload(length_data);
  offsets.upload(offset_data);
  auto kc = vt::Tensor::Contiguous(cache.tensor.data, DType::kI8, gpu.q.device,
                                    {physical_pages, page, kv_heads, dim});
  kc.stride[0] *= 2;
  auto vc = kc;
  vc.data = static_cast<char*>(kc.data) + page * kv_heads * dim;
  vt::PagedAttentionArgs args;
  args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
  args.k_scale = args.v_scale = 1.0f;
  args.scale = metadata_bytes.at("softmax_scale").get<float>();
  args.max_seq_len = tokens;
  args.causal = true;
  auto run = [&](const char* mode) {
    setenv("VT_XPU_ATTENTION", mode, 1);
    vt::PagedAttention(gpu.q, output.tensor, query.tensor, kc, vc,
                       table.tensor, lengths.tensor, offsets.tensor, args);
    vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
  };
  std::vector<float> expected(q_bytes / 2);
  for (size_t i = 0; i < expected.size(); ++i) {
    uint16_t bits;
    std::memcpy(&bits, expected_host.data() + 2 * i, 2);
    expected[i] = vt::F16ToF32(bits);
  }
  run("prefill");
  if (const char* path = std::getenv("VT_B70_FP8_PREFILL_OUTPUT_DUMP")) {
    const auto bytes = output.download();
    std::ofstream dump(path, std::ios::binary);
    REQUIRE(dump.good());
    dump.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    REQUIRE(dump.good());
  }
  Accuracy(output.floats(), expected, false, true);
  const auto events = vt::xpu::DrainProfileEvents();
  if (std::getenv("VT_XPU_PROFILE")) {
    int selected = 0;
    for (const auto& event : events)
      selected += event.stage == (xe2 ? "attention_prefill_xe2" : "attention_prefill_q64");
    REQUIRE(selected == 1);
  }
  std::vector<double> wall_ms;
  for (int sample = 0; sample < 5; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    run("prefill");
    wall_ms.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count());
  }
  auto sorted = wall_ms;
  std::sort(sorted.begin(), sorted.end());
  std::cout << nlohmann::json{{"event", "fp8_prefill_python_replay"},
      {"tokens", tokens}, {"page", page}, {"pages", pages},
      {"physical_pages", physical_pages},
      {"kv_row_stride", kc.stride[1]}, {"wall_samples_ms", wall_ms},
      {"wall_median_ms", sorted[2]}}.dump() << std::endl;
  Accuracy(output.floats(), expected, false, true);
  if (xe2 && std::getenv("VT_XPU_PROFILE")) {
    (void)vt::xpu::DrainProfileEvents();
    args.k_scale = 0.1f;
    args.v_scale = 0.3f;
    run("prefill");
    int fallback = 0;
    for (const auto& event : vt::xpu::DrainProfileEvents())
      fallback += event.stage == "attention_prefill_q64";
    CHECK(fallback == 1);  // Nonunit scales retain the existing GPU implementation.
  }
}

TEST_CASE("XPU FP8 attention: captured Python M04 Q5 operator replay"
          * doctest::skip(!std::getenv("VT_B70_M04_REPLAY_DIR"))) {
  const std::string directory = std::getenv("VT_B70_M04_REPLAY_DIR");
  auto read = [&](const std::string& name, size_t bytes) {
    std::ifstream file(directory + "/" + name, std::ios::binary);
    REQUIRE(file.good());
    std::vector<unsigned char> data(bytes);
    file.read(reinterpret_cast<char*>(data.data()), data.size());
    REQUIRE(file.gcount() == static_cast<std::streamsize>(bytes));
    REQUIRE(file.peek() == std::char_traits<char>::eof());
    return data;
  };
  std::ifstream manifest(directory + "/metadata.json");
  REQUIRE(manifest.good());
  const auto meta = nlohmann::json::parse(manifest);
  const int tokens = meta.at("q_shape").at(0).get<int>();
  const int heads = meta.at("q_shape").at(1).get<int>();
  const int dim = meta.at("q_shape").at(2).get<int>();
  const int pages = meta.at("kv_shape").at(0).get<int>();
  const int page = meta.at("kv_shape").at(1).get<int>();
  const int kv_heads = meta.at("kv_shape").at(2).get<int>();
  const int used = meta.at("used_k").get<int>();
  REQUIRE(tokens == 5);
  REQUIRE(heads == 24);
  REQUIRE(dim == 256);
  REQUIRE(page == 1664);
  REQUIRE(kv_heads == 4);
  REQUIRE(meta.at("kv_shape").at(3).get<int>() == dim);
  REQUIRE(pages == (used + page - 1) / page);
  REQUIRE(used >= 4096 + tokens);
  REQUIRE(meta.at("cu_seqlens_q") == nlohmann::json::array({0, tokens}));
  REQUIRE(meta.at("softmax_scale").get<float>() == 0.0625f);
  REQUIRE(meta.at("causal").get<bool>());

  const size_t q_bytes = size_t(tokens) * heads * dim * sizeof(uint16_t);
  const size_t page_bytes = size_t(page) * kv_heads * dim;
  const auto q_host = read("q.bin", q_bytes);
  const auto k_host = read("k.bin", size_t(pages) * page_bytes);
  const auto v_host = read("v.bin", size_t(pages) * page_bytes);
  const auto expected_host = read("output.bin", q_bytes);
  std::vector<unsigned char> cache_host(size_t(pages) * 2 * page_bytes);
  for (int p = 0; p < pages; ++p) {
    std::memcpy(cache_host.data() + size_t(2 * p) * page_bytes,
                k_host.data() + size_t(p) * page_bytes, page_bytes);
    std::memcpy(cache_host.data() + size_t(2 * p + 1) * page_bytes,
                v_host.data() + size_t(p) * page_bytes, page_bytes);
  }

  Queue gpu(vt::DeviceType::kXPU);
  Buffer query(gpu.q, DType::kF16, {tokens, heads, dim});
  Buffer output(gpu.q, DType::kF16, {tokens, heads, dim});
  Buffer cache(gpu.q, DType::kI8, {pages, 2 * page, kv_heads, dim});
  Buffer table(gpu.q, DType::kI32, {1, pages});
  Buffer lengths(gpu.q, DType::kI32, {1});
  Buffer offsets(gpu.q, DType::kI32, {2});
  query.upload(q_host.data());
  cache.upload(cache_host.data());
  std::vector<int32_t> ids(pages);
  std::iota(ids.begin(), ids.end(), 0);
  table.upload(ids.data());
  const int32_t seq_len[] = {used}, q_offsets[] = {0, tokens};
  lengths.upload(seq_len);
  offsets.upload(q_offsets);
  auto kc = vt::Tensor::Contiguous(cache.tensor.data, DType::kI8, gpu.q.device,
                                    {pages, page, kv_heads, dim});
  kc.stride[0] *= 2;
  auto vc = kc;
  vc.data = static_cast<char*>(kc.data) + page_bytes;
  vt::PagedAttentionArgs args;
  args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
  args.k_scale = meta.at("k_scale").get<float>();
  args.v_scale = meta.at("v_scale").get<float>();
  args.scale = meta.at("softmax_scale").get<float>();
  args.max_seq_len = used;
  args.causal = true;
  if (std::getenv("VT_XPU_PROFILE")) (void)vt::xpu::DrainProfileEvents();
  setenv("VT_XPU_ATTENTION", "split", 1);
  vt::PagedAttention(gpu.q, output.tensor, query.tensor, kc, vc,
                     table.tensor, lengths.tensor, offsets.tensor, args);
  vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
  if (std::getenv("VT_XPU_PROFILE")) {
    int split = 0;
    for (const auto& event : vt::xpu::DrainProfileEvents())
      split += event.stage == "attention_split_partial";
    CHECK(split == 1);
  }
  std::vector<float> expected(q_bytes / sizeof(uint16_t));
  for (size_t i = 0; i < expected.size(); ++i) {
    uint16_t bits;
    std::memcpy(&bits, expected_host.data() + i * sizeof(bits), sizeof(bits));
    expected[i] = vt::F16ToF32(bits);
  }
  const auto actual = output.floats();
  for (int row = 0; row < tokens; ++row) {
    double delta2 = 0, norm2 = 0, maximum = 0;
    for (int i = 0; i < heads * dim; ++i) {
      const size_t index = size_t(row) * heads * dim + i;
      const double delta = double(actual[index]) - expected[index];
      delta2 += delta * delta;
      norm2 += double(expected[index]) * expected[index];
      maximum = std::max(maximum, std::abs(delta));
    }
    std::cout << "M04_REPLAY row=" << row
              << " relative_rms=" << std::sqrt(delta2 / norm2)
              << " max_abs=" << maximum << std::endl;
  }
  Accuracy(actual, expected, false, true);
  CHECK(vt::GetReferenceTierHits() == 0);
}
