#include "xpu_test_helpers.h"
#include "vt/exl3_grouped.h"
#include "vt/exl3_w8a8_panel_plan.h"
#include "vt/breakable_graph.h"
#include "vt/xpu.h"
#include "vt/unaligned.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

namespace {
using xpu_test::Buffer;
using xpu_test::Queue;
using vt::DType;
void Accuracy(const std::vector<unsigned char>& raw, const vllm::StTensor& ref,
              const char* label) {
  REQUIRE(ref.dtype == "F16");
  REQUIRE(raw.size() == ref.nbytes);
  double error = 0, norm = 0, maximum = 0;
  bool finite = true;
  size_t changed = 0;
  for (size_t i = 0; i < raw.size(); i += 2) {
    const auto a = vt::LoadUnaligned<uint16_t>(raw.data() + i);
    const auto b = vt::LoadUnaligned<uint16_t>(ref.data + i);
    const double got = vt::F16ToF32(a), expected = vt::F16ToF32(b);
    finite &= std::isfinite(got) && std::isfinite(expected);
    error += (got - expected) * (got - expected); norm += expected * expected;
    maximum = std::max(maximum, std::abs(got - expected)); changed += a != b;
  }
  const double relative = std::sqrt(error / std::max(norm, 1e-30));
  CAPTURE(label);
  CAPTURE(relative);
  CAPTURE(maximum);
  std::cout << "W8A8_OPERATOR " << label << " relative=" << relative
            << " max_error=" << maximum << " half_differences=" << changed << '\n';
  CHECK(finite);
  CHECK(relative < 2e-3);
}
}

TEST_CASE("EXL3 W8A8 plan: explicit boundary bounded panel and aligned regions") {
  CHECK(vt::PlanExl3SmallM(128, 5120, 16384, 4).padded_rows == 128);
  CHECK_THROWS(vt::PlanExl3SmallM(129, 5120, 16384, 4));
  CHECK_THROWS(vt::PlanExl3W8A8(128, 5120, 16384, 2, 4));
  for (int m : {129, 256, 4096}) {
    const auto p = vt::PlanExl3W8A8(m, 5120, 16384, 2, 4);
    CHECK(p.padded_rows == (m + 255) / 256 * 256);
    CHECK(p.weight_panel_bytes == 5120 * 128);
    CHECK(p.row_scale_offset == size_t(2) * p.padded_rows * 5120);
    CHECK(p.intermediate_offset >= p.row_scale_offset + size_t(2) * p.padded_rows * 4);
    CHECK(p.weight_scale_offset >= p.intermediate_offset + size_t(p.padded_rows) * 16384 * 2);
    CHECK(p.workspace_bytes == p.weight_scale_offset + 64);
    for (const size_t offset : {p.activation_offset, p.row_scale_offset, p.intermediate_offset,
                                p.weight_scale_offset, p.workspace_bytes}) CHECK(offset % 64 == 0);
  }
  CHECK_THROWS(vt::PlanExl3W8A8(4097, 5120, 16384, 2, 4));
  CHECK_THROWS(vt::PlanExl3W8A8(129, 5130, 16384, 2, 4));
  CHECK_THROWS(vt::PlanExl3W8A8(129, 5120, 16384, 0, 4));
  CHECK_THROWS(vt::PlanExl3W8A8(129, 5120, 16384, 2, 5));
  CHECK_THROWS(vt::PlanExl3W8A8(129, 133248, 128, 1, 4));
  CHECK_THROWS(vt::PlanExl3W8A8(4096, 2147483520LL, 2147483520LL, 32767, 6));
}

TEST_CASE("XPU EXL3 W8A8 P2: real rows wider panels preserve rounded intermediates") {
  const char* env = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!env) std::exit(77);
  const auto f = vllm::SafetensorsFile::Open(env);
  const auto& tr = f.Get("merged_trellis");
  const auto& su = f.Get("stacked_suh");
  const auto& sv = f.Get("merged_svh");
  const auto& map = f.Get("source_map");
  int m = 256;
  if (const char* rows = std::getenv("VT_B70_EXL3_PANEL_ROWS")) {
    REQUIRE((std::string_view(rows) == "256" || std::string_view(rows) == "896" ||
             std::string_view(rows) == "1600"));
    m = std::atoi(rows);
  }
  const int k = int(su.shape[1]), n = int(sv.shape[0]);
  const int bits = int(tr.shape[2] / 16), groups = int(su.shape[0]);
  REQUIRE(groups >= 2);
  REQUIRE(std::memcmp(su.data, su.data + k * 2, k * 2) != 0);
  REQUIRE(map.nbytes == size_t(n / 128) * sizeof(int32_t));
  const auto suffix = "_m" + std::to_string(m);
  const auto& x = f.Get("input" + suffix);
  REQUIRE(x.dtype == "F16");
  REQUIRE((x.shape == std::vector<int64_t>{m, k}));
  std::vector<int32_t> source_map(n / 128);
  std::memcpy(source_map.data(), map.data, map.nbytes);
  Queue gpu(vt::DeviceType::kXPU);
  auto& backend = vt::GetBackend(gpu.q.device);
  Buffer trellis(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
  Buffer mapping(gpu.q, DType::kI32, {n / 128}), input(gpu.q, DType::kF16, {m, k});
  trellis.upload(tr.data); suh.upload(su.data); svh.upload(sv.data);
  mapping.upload(map.data); input.upload(x.data);
  std::vector<unsigned char> baseline_output, baseline_workspace;
  nlohmann::json reports = nlohmann::json::array();
  for (int width : {128, 1024, 2048}) {
    CAPTURE(width);
    const auto p = vt::PlanExl3W8A8(m, k, n, groups, bits, width);
    Buffer out(gpu.q, DType::kF16, {m, n});
    Buffer workspace(gpu.q, DType::kI8, {int64_t(p.workspace_bytes)});
    Buffer panel_storage(gpu.q, DType::kI8, {int64_t(p.weight_panel_bytes + 64)});
    auto panel = vt::Tensor::Contiguous(panel_storage.tensor.data, DType::kI8, gpu.q.device,
                                      {k, p.weight_panel_columns});
    const vt::Exl3GroupedLinearArgs args{bits, 2, "P2_REAL_ROWS", width};
    std::vector<unsigned char> poison(panel_storage.bytes, 0xcd);
    panel_storage.upload(poison.data());
    auto execute = [&] {
      const auto start = std::chrono::steady_clock::now();
      vt::Exl3GroupedW8A8(gpu.q, out.tensor, input.tensor, trellis.tensor, suh.tensor,
          svh.tensor, mapping.tensor, workspace.tensor, panel, args);
      backend.Synchronize(gpu.q);
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
    };
    const double cold = execute();
    const std::vector<double> warm{execute(), execute(), execute()};
    const auto result = out.download(), scratch = workspace.download(), weights = panel_storage.download();
    const auto parts = vt::PlanExl3W8A8Panels(source_map, groups, width);
    // Check the last128 columns of the actual reconstructed tail in its compact
    // layout, and confirm no store crosses the planned capacity into its guard.
    const int tail = parts.back().columns;
    const auto& last = f.Get("last_weight_panel" + suffix);
    REQUIRE(last.nbytes == size_t(k) * 128);
    bool reconstructed_tail_exact = true;
    for (int row = 0; row < k; ++row)
      reconstructed_tail_exact &= std::memcmp(weights.data() + size_t(row) * tail + tail - 128,
                                               last.data + size_t(row) * 128, 128) == 0;
    CHECK(reconstructed_tail_exact);
    CHECK(std::all_of(weights.begin() + p.weight_panel_bytes, weights.end(),
                      [](unsigned char v) { return v == 0xcd; }));
    bool preparation_exact = true, y_exact = true, output_exact = true;
    if (width == 128) {
      baseline_output = result; baseline_workspace = scratch;
      const auto& qref = f.Get("xq" + suffix);
      const auto& sref = f.Get("sx" + suffix);
      const auto& yref = f.Get("y" + suffix);
      REQUIRE(qref.nbytes == size_t(groups) * m * k);
      REQUIRE(sref.nbytes == size_t(groups) * m * sizeof(float));
      REQUIRE(yref.nbytes == size_t(m) * n * 2);
      bool original_quantized_exact = true, original_scales_exact = true, padding_exact = true;
      for (int group = 0; group < groups; ++group) for (int row = 0; row < p.padded_rows; ++row) {
        const auto* quantized = scratch.data() + p.activation_offset + (size_t(group) * p.padded_rows + row) * k;
        const auto* scale = scratch.data() + p.row_scale_offset + (size_t(group) * p.padded_rows + row) * sizeof(float);
        if (row < m) {
          original_quantized_exact &= std::memcmp(quantized, qref.data + (size_t(group) * m + row) * k, k) == 0;
          original_scales_exact &= std::memcmp(scale, sref.data + (size_t(group) * m + row) * sizeof(float), sizeof(float)) == 0;
        } else {
          padding_exact &= std::all_of(quantized, quantized + k, [](unsigned char v) { return v == 0; });
          const float one = 1.f;
          padding_exact &= std::memcmp(scale, &one, sizeof(one)) == 0;
        }
      }
      CHECK(original_quantized_exact);
      CHECK(original_scales_exact);
      CHECK(padding_exact);
      CHECK(std::memcmp(scratch.data() + p.intermediate_offset, yref.data, yref.nbytes) == 0);
      const auto* pad_y = scratch.data() + p.intermediate_offset + yref.nbytes;
      CHECK(std::all_of(pad_y, scratch.data() + p.weight_scale_offset,
                        [](unsigned char v) { return v == 0; }));
      const auto& original_output = f.Get("output" + suffix);
      REQUIRE(original_output.nbytes == result.size());
      CHECK(std::memcmp(result.data(), original_output.data, result.size()) == 0);
      Accuracy(result, f.Get("output" + suffix), "real_rows_128_control");
    } else {
      preparation_exact = std::equal(scratch.begin(), scratch.begin() + p.intermediate_offset,
                                    baseline_workspace.begin());
      y_exact = std::equal(scratch.begin() + p.intermediate_offset,
                          scratch.begin() + p.weight_scale_offset,
                          baseline_workspace.begin() + p.intermediate_offset);
      output_exact = result == baseline_output;
      CHECK(preparation_exact);
      CHECK(y_exact);
      CHECK(output_exact);
    }
    reports.push_back({{"m", m}, {"padded_m", p.padded_rows}, {"width", width}, {"panel_count", parts.size()},
        {"panel_bytes", p.weight_panel_bytes}, {"workspace_bytes", p.workspace_bytes},
        {"cold_operator_ms", cold}, {"warm_operator_ms", warm},
        {"preparation_exact", preparation_exact}, {"Y_exact", y_exact},
        {"output_exact", output_exact}, {"reconstructed_tail_exact", reconstructed_tail_exact}});
    std::cout << "P2_PANEL_OPERATOR " << reports.back().dump() << '\n';
  }
  if (const char* output = std::getenv("VT_B70_EXL3_PANEL_REPORT")) {
    REQUIRE_FALSE(std::filesystem::exists(output));
    std::ofstream file(output); file << reports.dump(2) << '\n';
    REQUIRE(file.good());
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 W8A8 P2: shared scratch completes growth reuse and cross-queue consumers") {
  const char* env = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!env) std::exit(77);
  const auto f = vllm::SafetensorsFile::Open(env);
  const auto& tr = f.Get("merged_trellis");
  const auto& su = f.Get("stacked_suh");
  const int k = int(su.shape[1]), n = int(f.Get("merged_svh").shape[0]);
  const int bits = int(tr.shape[2] / 16), groups = int(su.shape[0]);
  Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU);
  Buffer trellis(first.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(first.q, DType::kF16, {groups, k}), svh(first.q, DType::kF16, {n});
  Buffer map(first.q, DType::kI32, {n / 128});
  trellis.upload(tr.data); suh.upload(su.data); svh.upload(f.Get("merged_svh").data);
  map.upload(f.Get("source_map").data);
  std::vector<unsigned char> baseline;
  size_t capacity = vt::xpu::GetMemoryInfo().w8a8_workspace_bytes;
  std::cout << "P2_SHARED_INITIAL retained_bytes=" << capacity << '\n';
  for (const int width : {128, 1024, 2048, 1024, 128}) {
    CAPTURE(width);
    Buffer input(first.q, DType::kF16, {256, k}), out(first.q, DType::kF16, {256, n});
    input.upload(f.Get("input_m256").data);
    const auto p = vt::PlanExl3W8A8(256, k, n, groups, bits, width);
    const vt::Exl3GroupedLinearArgs args{bits, 2, "P2_SHARED", width};
    const auto start = std::chrono::steady_clock::now();
    vt::Exl3GroupedW8A8(first.q, out.tensor, input.tensor, trellis.tensor, suh.tensor,
                        svh.tensor, map.tensor, args);
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    // The owned call itself has completed all consumers. A download or a
    // second queue may now reuse the one pool, with no caller retirement fence.
    const auto raw = out.download();
    if (baseline.empty()) { baseline = raw; Accuracy(raw, f.Get("output_m256"), "shared128"); }
    else xpu_test::SameBytes(raw, baseline);
    capacity = std::max(capacity, p.workspace_bytes + p.weight_panel_bytes);
    const auto info = vt::xpu::GetMemoryInfo();
    CHECK(info.w8a8_workspace_bytes == capacity);
    CHECK(info.allocated_bytes >= capacity);
    std::cout << "P2_SHARED_OPERATOR width=" << width << " cold_or_reuse_ms=" << elapsed
              << " retained_bytes=" << capacity << " device_peak_bytes="
              << info.peak_allocated_bytes << '\n';
  }
  {
    Buffer input(second.q, DType::kF16, {129, k}), out(second.q, DType::kF16, {129, n});
    input.upload(f.Get("input_m129").data);
    const vt::Exl3GroupedLinearArgs args{bits, 2, "P2_SECOND_QUEUE", 2048};
    vt::Exl3GroupedW8A8(second.q, out.tensor, input.tensor, trellis.tensor, suh.tensor,
                        svh.tensor, map.tensor, args);
    const auto before = out.download();
    Accuracy(before, f.Get("output_m129"), "shared_second_queue");
    CHECK(vt::xpu::GetMemoryInfo().w8a8_workspace_bytes == capacity);
    auto invalid = map.download(); const int32_t bad = groups;
    std::memcpy(invalid.data(), &bad, sizeof(bad)); map.upload(invalid.data());
    CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(second.q, out.tensor, input.tensor, trellis.tensor,
        suh.tensor, svh.tensor, map.tensor, args), doctest::Contains("group out of range"),
        std::runtime_error);
    xpu_test::SameBytes(out.download(), before);
    map.upload(f.Get("source_map").data);
    // A rejected lease must leave the pool safe for a later valid consumer.
    vt::Exl3GroupedW8A8(second.q, out.tensor, input.tensor, trellis.tensor, suh.tensor,
                        svh.tensor, map.tensor, args);
    xpu_test::SameBytes(out.download(), before);
    auto& backend = vt::GetBackend(second.q.device);
    if (backend.SupportsGraphCapture()) {
      Buffer witness(second.q, DType::kI8, {64});
      const auto memory_before = vt::xpu::GetMemoryInfo();
      backend.BeginCapture(second.q);
      backend.Memset(second.q, witness.tensor.data, 0, witness.bytes);
      CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(second.q, out.tensor, input.tensor, trellis.tensor,
          suh.tensor, svh.tensor, map.tensor, args), doctest::Contains("eager-only"), std::runtime_error);
      void* graph = backend.EndCaptureGraph(second.q);
      backend.DestroyGraph(graph);
      xpu_test::SameBytes(out.download(), before);
      const auto memory_after = vt::xpu::GetMemoryInfo();
      CHECK(memory_after.w8a8_workspace_bytes == capacity);
      CHECK(memory_after.graph_device_bytes == memory_before.graph_device_bytes);
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 W8A8: real grouped operator boundary large-M and reuse") {
  const char* env = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!env) { std::cerr << "SKIP: VT_B70_EXL3_W8A8_FIXTURE required\n"; std::exit(77); }
  auto f = vllm::SafetensorsFile::Open(env);
  const auto& tr = f.Get("merged_trellis");
  const auto& su = f.Get("stacked_suh"); const auto& sv = f.Get("merged_svh");
  const auto& map = f.Get("source_map");
  const int k = int(tr.shape[0] * 16), n = int(tr.shape[1] * 16);
  const int bits = int(tr.shape[2] / 16), groups = int(su.shape[0]);
  REQUIRE(tr.dtype == "I16"); REQUIRE(su.dtype == "F16"); REQUIRE(sv.dtype == "F16");
  REQUIRE(map.dtype == "I32");
  Queue gpu(vt::DeviceType::kXPU);
  Buffer trellis(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
  Buffer mapping(gpu.q, DType::kI32, {n / 128});
  trellis.upload(tr.data); suh.upload(su.data); svh.upload(sv.data); mapping.upload(map.data);
  const std::string matrix = std::filesystem::path(env).stem().string();
  const vt::Exl3GroupedLinearArgs args{bits, 2, matrix.c_str()};
  const bool four_k = std::find(f.Names().begin(), f.Names().end(), "input_m4096") != f.Names().end();
  const std::vector<int> sequence = four_k ? std::vector<int>{4096} : std::vector<int>{129, 128, 256, 128, 129};
  const auto maximum = vt::PlanExl3W8A8(four_k ? 4096 : 256, k, n, groups, bits);
  Buffer scratch(gpu.q, DType::kI8, {int64_t(maximum.workspace_bytes)});
  Buffer panel(gpu.q, DType::kI8, {k, 128});
  // Explicit large/small/large transitions exercise both arithmetic routes,
  // reusing the W8A8 storage instead of relying on fresh allocator contents.
  for (int m : sequence) {
    CAPTURE(m);
    Buffer in(gpu.q, DType::kF16, {m, k}), out(gpu.q, DType::kF16, {m, n});
    const auto& x = f.Get("input_m" + std::to_string(m)); in.upload(x.data);
    if (m == 128) {
      const auto p = vt::PlanExl3SmallM(m, k, n, bits);
      Buffer had(gpu.q, DType::kF16, {groups, k / 16, p.padded_rows, 16});
      Buffer parts(gpu.q, DType::kF32, {p.splits, m, n});
      vt::Exl3GroupedLinear(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
          svh.tensor, mapping.tensor, had.tensor, parts.tensor, args);
      xpu_test::SameBytes(out.download(), std::vector<unsigned char>(
          f.Get("output_m128").data, f.Get("output_m128").data + f.Get("output_m128").nbytes));
      continue;
    }
    std::vector<unsigned char> poison(scratch.bytes, 0xff); scratch.upload(poison.data());
    vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
        svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args);
    Accuracy(out.download(), f.Get("output_m" + std::to_string(m)), "output");
    const auto& wref = f.Get("last_weight_panel_m" + std::to_string(m));
    xpu_test::SameBytes(panel.download(), std::vector<unsigned char>(wref.data, wref.data + wref.nbytes));
    const auto p = vt::PlanExl3W8A8(m, k, n, groups, bits);
    const auto raw = scratch.download();
    const auto& qref = f.Get("xq_m" + std::to_string(m));
    const auto& sref = f.Get("sx_m" + std::to_string(m));
    bool quantized_exact = true, scales_exact = true, padding_zero = true;
    for (int g = 0; g < groups; ++g) for (int row = 0; row < p.padded_rows; ++row) {
      const auto* a = raw.data() + p.activation_offset + (size_t(g) * p.padded_rows + row) * k;
      const auto* scale = raw.data() + p.row_scale_offset + (size_t(g) * p.padded_rows + row) * 4;
      if (row < m) {
        quantized_exact &= std::memcmp(a, qref.data + (size_t(g) * m + row) * k, k) == 0;
        scales_exact &= std::memcmp(scale, sref.data + (size_t(g) * m + row) * 4, 4) == 0;
      } else {
        padding_zero &= std::all_of(a, a + k, [](unsigned char v) { return v == 0; });
        padding_zero &= vt::LoadUnaligned<float>(scale) == 1.0f;
      }
    }
    CHECK(quantized_exact); CHECK(scales_exact); CHECK(padding_zero);
    const auto ybytes = size_t(m) * n * 2;
    Accuracy(std::vector<unsigned char>(raw.begin() + p.intermediate_offset,
        raw.begin() + p.intermediate_offset + ybytes), f.Get("y_m" + std::to_string(m)), "F16_intermediate");
    if (m == 129) {
      if (groups > 1) {
        // H128 is block-local. Permuting complete packed output blocks, SV
        // and their source IDs must give exactly the same permutation of the
        // original W8A8 output, including a noncontiguous source-group map.
        auto packed_bytes = trellis.download(), sv_bytes = svh.download();
        auto map_bytes = mapping.download();
        const size_t block_bytes = size_t(8) * 32 * bits;
        for (int tile = 0; tile < k / 16; ++tile) {
          const size_t first = size_t(tile) * (n / 16) * 32 * bits;
          const size_t last = first + size_t(n / 128 - 1) * block_bytes;
          std::swap_ranges(packed_bytes.begin() + first, packed_bytes.begin() + first + block_bytes,
                           packed_bytes.begin() + last);
        }
        std::swap_ranges(sv_bytes.begin(), sv_bytes.begin() + 256, sv_bytes.end() - 256);
        for (int i = 0; i < 4; ++i) std::swap(map_bytes[i], map_bytes[map_bytes.size() - 4 + i]);
        REQUIRE(vt::LoadUnaligned<int32_t>(map_bytes.data()) !=
                vt::LoadUnaligned<int32_t>(map_bytes.data() + map_bytes.size() - 4));
        trellis.upload(packed_bytes.data()); svh.upload(sv_bytes.data()); mapping.upload(map_bytes.data());
        const auto& expected = f.Get("output_m129");
        std::vector<unsigned char> permuted(expected.data, expected.data + expected.nbytes);
        for (int row = 0; row < m; ++row) {
          const size_t start = size_t(row) * n * 2;
          std::swap_ranges(permuted.begin() + start, permuted.begin() + start + 256,
                           permuted.begin() + start + size_t(n - 128) * 2);
        }
        vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
            svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args);
        xpu_test::SameBytes(out.download(), permuted);
        trellis.upload(tr.data); svh.upload(sv.data); mapping.upload(map.data);
      }
      const auto before = out.download();
      auto bad = scratch.tensor; bad.shape[0] = int64_t(p.workspace_bytes) - 1;
      CHECK_THROWS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
          svh.tensor, mapping.tensor, bad, panel.tensor, args));
      auto alias = panel.tensor; alias.data = out.tensor.data;
      CHECK_THROWS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
          svh.tensor, mapping.tensor, scratch.tensor, alias, args));
      auto map_raw = mapping.download();
      const int32_t invalid = groups; std::memcpy(map_raw.data(), &invalid, sizeof(invalid));
      mapping.upload(map_raw.data());
      CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
          svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args),
          doctest::Contains("group out of range"), std::runtime_error);
      xpu_test::SameBytes(out.download(), before);
      mapping.upload(map.data);
      const auto original_input = in.download();
      const auto original_svh = svh.download();
      const auto original_suh = suh.download();
      auto reject = [&](const char* message, const std::string& boundary = "nonfinite operand") {
        CAPTURE(boundary);
        CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor,
            suh.tensor, svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args),
            doctest::Contains(message), std::runtime_error);
        xpu_test::SameBytes(out.download(), before);
      };
      for (const uint16_t value : {uint16_t{0x7e00}, uint16_t{0x7c00}, uint16_t{0xfc00}}) {
        CAPTURE(value);
        // First/last blocks exercise the whole parallel SV check, including
        // its tail. The input and metadata failures must not write output.
        for (const size_t offset : {size_t{0}, svh.bytes - 2}) {
          auto bad_svh = original_svh;
          std::memcpy(bad_svh.data() + offset, &value, 2); svh.upload(bad_svh.data());
          reject("finite svh");
        }
        svh.upload(original_svh.data());
        auto bad_input = original_input;
        std::memcpy(bad_input.data() + bad_input.size() - 2, &value, 2);
        in.upload(bad_input.data()); reject("finite inputs");
        auto bad_svh = original_svh;
        std::memcpy(bad_svh.data(), &value, 2); svh.upload(bad_svh.data());
        reject("finite svh");  // Prior error priority when both checks fail.
        in.upload(original_input.data()); svh.upload(original_svh.data());
      }
      // Finite operands can overflow at either FP16 rounding boundary.
      auto bad_input = original_input, bad_suh = original_suh;
      const uint16_t maximum_half = 0x7bff, one_half = 0x3c00;
      std::memcpy(bad_input.data(), &maximum_half, 2);
      std::memcpy(bad_suh.data(), &maximum_half, 2);
      in.upload(bad_input.data()); suh.upload(bad_suh.data()); reject("finite inputs", "FP16 product");
      for (int col = 0; col < 128; ++col) {
        std::memcpy(bad_input.data() + col * 2, &maximum_half, 2);
        std::memcpy(bad_suh.data() + col * 2, &one_half, 2);
      }
      in.upload(bad_input.data()); suh.upload(bad_suh.data()); reject("finite inputs", "FP16 Hadamard");
      in.upload(original_input.data()); suh.upload(original_suh.data());
      // Reuse the same flag storage after failed calls: neither zero flag may
      // leak into the next valid projection.
      vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
          svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args);
      Accuracy(out.download(), f.Get("output_m129"), "after_rejected_operands");
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 W8A8 model seam: grouped and single projection own scratch retirement") {
  const char* env = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!env) std::exit(77);
  auto f = vllm::SafetensorsFile::Open(env);
  const auto& tr = f.Get("merged_trellis");
  const auto& su = f.Get("stacked_suh");
  const int64_t k = tr.shape[0] * 16, n = tr.shape[1] * 16;
  const int bits = int(tr.shape[2] / 16), groups = int(su.shape[0]);
  auto owned = [](const vllm::StTensor& source, DType dtype,
                   std::initializer_list<int64_t> shape) {
    vllm::OwnedTensor t; t.dtype = dtype; t.rank = int(shape.size());
    std::copy(shape.begin(), shape.end(), t.shape);
    t.bytes.resize(source.nbytes); std::memcpy(t.bytes.data(), source.data, source.nbytes);
    return t;
  };
  vllm::Exl3GroupedWeight w; w.name = "model_seam_grouped"; w.codebook = 2;
  w.trellis = owned(tr, DType::kI8, {k / 16, n / 16, 32 * bits});
  w.suh = owned(su, DType::kF16, {groups, k});
  w.svh = owned(f.Get("merged_svh"), DType::kF16, {n});
  w.source_map = owned(f.Get("source_map"), DType::kI32, {n / 128});
  Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  for (int m : {129, 256, 128}) {
    Buffer input(gpu.q, DType::kF16, {m, k}); input.upload(f.Get("input_m" + std::to_string(m)).data);
    auto out = vllm::dense_attn::Exl3GroupedMatmulD(d, input.tensor, w);
    std::vector<unsigned char> raw(size_t(m) * n * 2);
    d.b.Copy(d.q, raw.data(), out.t().data, raw.size()); d.b.Synchronize(d.q);
    Accuracy(raw, f.Get("output_m" + std::to_string(m)), "grouped_model_seam");
    CHECK(w.trellis.bytes.empty()); CHECK(w.suh.bytes.empty());
    if (m > 128) {
      const auto p = vt::PlanExl3W8A8(m, k, n, groups, bits);
      CHECK(vt::xpu::GetMemoryInfo().w8a8_workspace_bytes >= p.workspace_bytes + p.weight_panel_bytes);
    }
  }
  if (groups == 1) {
    vllm::Exl3Weight single; single.name = "model_seam_single"; single.codebook = 2;
    single.trellis = owned(tr, DType::kI8, {k / 16, n / 16, 32 * bits});
    single.suh = owned(su, DType::kF16, {k});
    single.svh = owned(f.Get("merged_svh"), DType::kF16, {n});
    Buffer input(gpu.q, DType::kF16, {129, k}); input.upload(f.Get("input_m129").data);
    auto out = vllm::dense_attn::Exl3MatmulD(d, input.tensor, single, DType::kF16);
    std::vector<unsigned char> raw(129 * n * 2);
    d.b.Copy(d.q, raw.data(), out.t().data, raw.size()); d.b.Synchronize(d.q);
    Accuracy(raw, f.Get("output_m129"), "single_model_seam");
    const auto p = vt::PlanExl3W8A8(129, k, n, 1, bits);
    CHECK(vt::xpu::GetMemoryInfo().w8a8_workspace_bytes >= p.workspace_bytes + p.weight_panel_bytes);
    // Exercise the projection's generated routing metadata under real XPU
    // capture. A per-call zero-filled map must fail backend preflight.
    Buffer decode_input(gpu.q, DType::kF16, {128, k});
    decode_input.upload(f.Get("input_m128").data);
    auto warm = vllm::dense_attn::Exl3MatmulD(d, decode_input.tensor, single, DType::kF16);
    std::vector<unsigned char> eager_bytes(128 * n * 2);
    d.b.Copy(gpu.q, eager_bytes.data(), warm.t().data, eager_bytes.size());
    d.b.Synchronize(d.q);
    warm = {};
    const auto* map_address = single.single_source_map.d_dev.get();
    vt::BreakableGraph graph;
    vllm::dense_attn::DBuf captured;
    {
      vt::GraphCaptureScope scope(d.b, gpu.q, graph, vt::GraphCaptureMode::kFull);
      captured = vllm::dense_attn::Exl3MatmulD(d, decode_input.tensor, single, DType::kF16);
    }
    REQUIRE(graph.captured());
    graph.Replay(gpu.q);
    raw.resize(128 * n * 2);
    d.b.Copy(gpu.q, raw.data(), captured.t().data, raw.size());
    d.b.Synchronize(gpu.q);
    Accuracy(raw, f.Get("output_m128"), "single_model_graph");
    CHECK(raw == eager_bytes);
    CHECK(single.single_source_map.d_dev.get() == map_address);
    // Changed input contents reach the same graph and persistent map.
    d.b.Memset(gpu.q, decode_input.tensor.data, 0, decode_input.tensor.Bytes());
    graph.Replay(gpu.q);
    d.b.Copy(gpu.q, raw.data(), captured.t().data, raw.size());
    d.b.Synchronize(gpu.q);
    bool all_zero = true;
    for (size_t i = 0; i < raw.size(); i += 2)
      all_zero &= vt::F16ToF32(vt::LoadUnaligned<uint16_t>(raw.data() + i)) == 0.0f;
    CHECK(all_zero);
    graph.Reset();
  }
  const auto info = vt::xpu::GetMemoryInfo();
  std::cout << "P2_MODEL_POOL retained_bytes=" << info.w8a8_workspace_bytes
            << " device_peak_bytes=" << info.peak_allocated_bytes << '\n';
  CHECK(vt::GetReferenceTierHits() == 0);
}
