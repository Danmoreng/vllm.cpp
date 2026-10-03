#include "xpu_test_helpers.h"
#include "exl3_fixture.h"
#include "vt/exl3_grouped.h"
#include "vt/unaligned.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/dense_exl3_linear.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;

double Relative(const std::vector<float>& got, const std::vector<float>& expected) {
  REQUIRE(got.size() == expected.size());
  double error = 0, norm = 0;
  bool finite = true;
  for (size_t i = 0; i < got.size(); ++i) {
    finite &= std::isfinite(got[i]) && std::isfinite(expected[i]);
    const double delta = double(got[i]) - expected[i];
    error += delta * delta;
    norm += double(expected[i]) * expected[i];
  }
  REQUIRE(finite);
  REQUIRE(norm > 0);
  return std::sqrt(error / norm);
}

struct Scratch {
  vt::Exl3SmallMPlan plan;
  Buffer had, parts;
  Scratch(vt::Queue& q, int m, int k, int n, int bits, int groups)
      : plan(vt::PlanExl3SmallM(m, k, n, bits)),
        had(q, DType::kF16, {groups, k / 16, plan.padded_rows, 16}),
        parts(q, DType::kF32, {plan.splits, m, n}) {
    // Stale/poisoned allocations must be fully overwritten, including DPAS
    // padded rows which the caller has not promised to initialize.
    std::vector<uint16_t> poison(had.bytes / 2, 0x7e00);
    had.upload(poison.data());
    parts.put(std::vector<float>(parts.bytes / 4,
                                 std::numeric_limits<float>::quiet_NaN()));
  }
};

void CheckPadding(Scratch& scratch, int m, int k, int groups) {
  const auto raw = scratch.had.download();
  bool zero = true;
  for (int g = 0; g < groups; ++g) for (int tile = 0; tile < k / 16; ++tile)
    for (int row = m; row < scratch.plan.padded_rows; ++row)
      for (int col = 0; col < 16; ++col) {
        const size_t offset = (((size_t(g) * (k / 16) + tile) *
                                 scratch.plan.padded_rows + row) * 16 + col) * 2;
        zero &= vt::LoadUnaligned<uint16_t>(raw.data() + offset) == 0;
      }
  CHECK(zero);
}

std::vector<float> Floats(const vllm::StTensor& t) {
  REQUIRE((t.dtype == "F16" || t.dtype == "F32"));
  const size_t size = t.dtype == "F16" ? 2 : 4;
  std::vector<float> result(t.nbytes / size);
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = size == 2 ? vt::F16ToF32(vt::LoadUnaligned<uint16_t>(t.data + i * size))
                          : vt::LoadUnaligned<float>(t.data + i * size);
  return result;
}
}

TEST_CASE("XPU EXL3 producer SmallM: separate group transforms, padding and refusals") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int k = 256, n = 384, groups = 2;
  const std::vector<int32_t> mapping{1, 0, 1};
  for (int bits : {4, 6}) for (int m : {1, 4, 12, 16, 128}) {
    CAPTURE(bits);
    CAPTURE(m);
    auto f = exl3_test::MakeFixture(k, n, bits, 0x9827u + bits);
    auto u = f.suh;
    // Group1 is not merely the same transform under another index.
    for (int i = 0; i < k; ++i)
      u.push_back(vt::F32ToF16((i % 3 ? 0.31f : -0.62f) * vt::F16ToF32(f.suh[i])));
    Buffer input(gpu.q, DType::kF16, {m, k}), packed(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
    Buffer scales(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
    Buffer shard(gpu.q, DType::kI32, {n / 128}), output(gpu.q, DType::kF16, {m, n});
    input.put(xpu_test::Values(m * k, 13, 0.007f));
    packed.upload(f.trellis.data()); scales.upload(u.data()); svh.upload(f.svh.data());
    shard.upload(mapping.data());
    Scratch scratch(gpu.q, m, k, n, bits, groups);
    const vt::Exl3GroupedLinearArgs args{bits, 2, "synthetic_grouped"};
    vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed.tensor,
                          scales.tensor, svh.tensor, shard.tensor,
                          scratch.had.tensor, scratch.parts.tensor, args);
    const auto got = output.floats();
    CheckPadding(scratch, m, k, groups);
    std::vector<float> expected(m * n);
    const auto input_bytes = input.download();
    // Independent CPU-order reference: one 128-output block at a time with
    // the input transform selected by its source group. Use F32 output before
    // the final F16 rounding, as in the producer's FP32 HadOut/scale tail.
    for (int nb = 0; nb < n / 128; ++nb) {
      Buffer ca(cpu.q, DType::kF16, {m, k}), cb(cpu.q, DType::kI8, {k / 16, 8, 32 * bits});
      Buffer cu(cpu.q, DType::kF16, {k}), cv(cpu.q, DType::kF16, {128});
      Buffer ch(cpu.q, DType::kF16, {m, k}), co(cpu.q, DType::kF32, {m, 128});
      std::vector<uint8_t> panel(cb.bytes);
      for (int tile = 0; tile < k / 16; ++tile)
        std::memcpy(panel.data() + size_t(tile) * 8 * 32 * bits,
                    reinterpret_cast<const uint8_t*>(f.trellis.data()) +
                        (size_t(tile) * (n / 16) + nb * 8) * 32 * bits,
                    8 * 32 * bits);
      ca.upload(input_bytes.data()); cb.upload(panel.data());
      cu.upload(u.data() + mapping[nb] * k); cv.upload(f.svh.data() + nb * 128);
      vt::Exl3Gemm(cpu.q, co.tensor, ca.tensor, cb.tensor, cu.tensor, cv.tensor, ch.tensor,
                   vt::Exl3GemmArgs{bits, 2});
      const auto panel_out = co.floats();
      for (int row = 0; row < m; ++row) for (int col = 0; col < 128; ++col)
        expected[row * n + nb * 128 + col] =
            vt::F16ToF32(vt::F32ToF16(panel_out[row * 128 + col]));
    }
    const double relative = Relative(got, expected);
    std::cout << "GROUPED_SMALLM bits=" << bits << " M=" << m << " relative=" << relative << '\n';
    CHECK(relative < 2e-3);
    if (m == 4 && bits == 4) {
      const auto before = output.download();
      auto bad = args; bad.codebook = 1;
      CHECK_THROWS(vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed.tensor,
          scales.tensor, svh.tensor, shard.tensor, scratch.had.tensor, scratch.parts.tensor, bad));
      auto alias = scratch.parts.tensor; alias.data = output.tensor.data;
      CHECK_THROWS(vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed.tensor,
          scales.tensor, svh.tensor, shard.tensor, scratch.had.tensor, alias, args));
      auto wrong = scratch.had.tensor; --wrong.shape[2];
      CHECK_THROWS(vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed.tensor,
          scales.tensor, svh.tensor, shard.tensor, wrong, scratch.parts.tensor, args));
      const int32_t invalid[] = {1, 0, groups}; shard.upload(invalid);
      CHECK_THROWS_WITH_AS(vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed.tensor,
          scales.tensor, svh.tensor, shard.tensor, scratch.had.tensor, scratch.parts.tensor, args),
          doctest::Contains("group out of range"), std::runtime_error);
      xpu_test::SameBytes(output.download(), before);
    }
  }
  CHECK_THROWS(vt::PlanExl3SmallM(129, k, n, 4));
  CHECK_THROWS(vt::PlanExl3SmallM(1, k, n, 3));
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 producer SmallM: real M1/M4 captures and first/last head blocks") {
  const char* env = std::getenv("VT_B70_EXL3_S0B_FIXTURES");
  if (!env) {
    std::cerr << "SKIP: set VT_B70_EXL3_S0B_FIXTURES to pinned local captures.\n";
    std::exit(77);
  }
  Queue gpu(vt::DeviceType::kXPU);
  for (const std::string family : {"mlp_gate", "head_first", "head_last"}) {
    CAPTURE(family);
    const std::filesystem::path root(env);
    auto f = vllm::SafetensorsFile::Open((root / (family + ".safetensors")).string());
    auto oracle = vllm::SafetensorsFile::Open((root / (family + "_oracle_v2.safetensors")).string());
    const auto& tr = f.Get("trellis");
    REQUIRE(tr.dtype == "I16"); REQUIRE(tr.shape.size() == 3);
    const int k = int(tr.shape[0] * 16), n = int(tr.shape[1] * 16), bits = int(tr.shape[2] / 16);
    REQUIRE(vt::LoadUnaligned<uint32_t>(f.Get("mul1").data) == 0x83DCD12Du);
    Buffer packed(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
    Buffer u(gpu.q, DType::kF16, {1, k}), v(gpu.q, DType::kF16, {n});
    Buffer shard(gpu.q, DType::kI32, {n / 128});
    REQUIRE(packed.bytes == tr.nbytes);
    packed.upload(tr.data); u.upload(f.Get("suh").data); v.upload(f.Get("svh").data);
    std::vector<int32_t> map(n / 128, 0); shard.upload(map.data());
    for (int m : {1, 4}) {
      CAPTURE(m);
      const std::string prefix = "m" + std::to_string(m);
      Buffer input(gpu.q, DType::kF16, {m, k}), output(gpu.q, DType::kF16, {m, n});
      const auto& x = f.Get("activation_" + prefix + "_fp16");
      REQUIRE(input.bytes == x.nbytes); input.upload(x.data);
      Scratch scratch(gpu.q, m, k, n, bits, 1);
      const auto& parts = oracle.Get(prefix + "_production_split_parts_f32");
      REQUIRE(parts.shape == std::vector<int64_t>{scratch.plan.splits, m, n});
      vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed.tensor,
          u.tensor, v.tensor, shard.tensor, scratch.had.tensor, scratch.parts.tensor,
          {bits, 2, family.c_str()});
      const auto had = scratch.had.download();
      std::vector<unsigned char> active(size_t(m) * k * 2);
      for (int row = 0; row < m; ++row) for (int tile = 0; tile < k / 16; ++tile)
        std::memcpy(active.data() + size_t(row * k + tile * 16) * 2,
                    had.data() + size_t(tile * scratch.plan.padded_rows + row) * 32, 32);
      const auto& expected_had = oracle.Get(prefix + "_production_input_hadamard_f16");
      xpu_test::SameBytes(active, std::vector<unsigned char>(expected_had.data,
                                                           expected_had.data + expected_had.nbytes));
      CheckPadding(scratch, m, k, 1);
      const double part_relative = Relative(scratch.parts.floats(), Floats(parts));
      const auto& expected = oracle.Get(prefix + "_production_output_f16");
      const double output_relative = Relative(output.floats(), Floats(expected));
      std::cout << "REAL_SMALLM family=" << family << " M=" << m
                << " splits=" << scratch.plan.splits << " padded_M=" << scratch.plan.padded_rows
                << " part_relative=" << part_relative << " output_relative=" << output_relative << '\n';
      CHECK(part_relative < 2e-3);
      CHECK(output_relative < 2e-3);
      // The pinned header and launch/math configuration should preserve every
      // byte of the actual producer Partials/Output, stronger than the gate.
      xpu_test::SameBytes(scratch.parts.download(),
          std::vector<unsigned char>(parts.data, parts.data + parts.nbytes));
      xpu_test::SameBytes(output.download(),
          std::vector<unsigned char>(expected.data, expected.data + expected.nbytes));
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 producer SmallM: real grouped Gate/Up M1/4/12/16/128") {
  const char* sources = std::getenv("VT_B70_EXL3_S0B_FIXTURES");
  const char* captures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  if (!sources || !captures) {
    std::cerr << "SKIP: set VT_B70_EXL3_S0B_FIXTURES and VT_B70_EXL3_S1_FIXTURES.\n";
    std::exit(77);
  }
  Queue gpu(vt::DeviceType::kXPU);
  auto gate_file = vllm::SafetensorsFile::Open((std::filesystem::path(sources) / "mlp_gate.safetensors").string());
  auto up_file = vllm::SafetensorsFile::Open((std::filesystem::path(captures) / "mlp_up.safetensors").string());
  auto oracle = vllm::SafetensorsFile::Open((std::filesystem::path(captures) / "mlp_gate_up_oracle.safetensors").string());
  const auto load = [](const vllm::SafetensorsFile& file) {
    const auto get = [&](const std::string& name) -> const vllm::StTensor& {
      return file.Get(name.substr(2));
    };
    const auto has = [&](const std::string& name) {
      const auto key = name.substr(2);
      return std::find(file.Names().begin(), file.Names().end(), key) != file.Names().end();
    };
    return vllm::dense_loaders::LoadExl3(get, has, "p");
  };
  auto gate = load(gate_file), up = load(up_file);
  auto grouped = vllm::MergeExl3Weights({&gate, &up}, "real_gate_up");
  REQUIRE(gate.InFeatures() == 5120);
  REQUIRE(gate.OutFeatures() == 17408);
  REQUIRE(up.InFeatures() == 5120);
  REQUIRE(up.OutFeatures() == 17408);
  REQUIRE(gate.Bits() == 4);
  REQUIRE(up.Bits() == 4);
  constexpr int k = 5120, n = 34816, bits = 4, groups = 2;
  for (const auto& pair : {
           std::pair{&grouped.trellis, "merged_trellis"},
           std::pair{&grouped.suh, "stacked_suh"},
           std::pair{&grouped.svh, "merged_svh"},
           std::pair{&grouped.source_map, "source_map"}}) {
    const auto& expected = oracle.Get(pair.second);
    REQUIRE(pair.first->bytes.size() == expected.nbytes);
    CHECK(std::memcmp(pair.first->bytes.data(), expected.data, expected.nbytes) == 0);
  }
  CHECK(std::memcmp(gate.suh.bytes.data(), up.suh.bytes.data(), k * 2) != 0);
  Buffer packed(gpu.q, DType::kI8, {k/16, n/16, 32*bits});
  Buffer suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
  Buffer map(gpu.q, DType::kI32, {n/128});
  packed.upload(grouped.trellis.bytes.data());
  suh.upload(grouped.suh.bytes.data());
  svh.upload(grouped.svh.bytes.data());
  map.upload(grouped.source_map.bytes.data());
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  const void* resident = nullptr;
  for (int m : {1, 4, 12, 16, 128}) {
    CAPTURE(m);
    const std::string prefix = "m" + std::to_string(m);
    Buffer input(gpu.q, DType::kF16, {m, k}), output(gpu.q, DType::kF16, {m, n});
    input.upload(oracle.Get("activation_" + prefix + "_fp16").data);
    Scratch scratch(gpu.q, m, k, n, bits, groups);
    const auto& expected_parts = oracle.Get(prefix + "_production_split_parts_f32");
    REQUIRE(expected_parts.shape == std::vector<int64_t>{scratch.plan.splits, m, n});
    vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed.tensor,
        suh.tensor, svh.tensor, map.tensor, scratch.had.tensor, scratch.parts.tensor,
        {bits, 2, "real_gate_up_direct"});
    const auto had = scratch.had.download();
    std::vector<unsigned char> active(size_t(groups) * m * k * 2);
    for (int s = 0; s < groups; ++s) for (int row = 0; row < m; ++row)
      for (int tile = 0; tile < k/16; ++tile)
        std::memcpy(active.data() + (size_t(s*m + row)*k + tile*16)*2,
                    had.data() + ((size_t(s)*(k/16) + tile)*scratch.plan.padded_rows + row)*32, 32);
    const auto& expected_had = oracle.Get(prefix + "_production_input_hadamard_f16");
    xpu_test::SameBytes(active, std::vector<unsigned char>(expected_had.data, expected_had.data + expected_had.nbytes));
    CheckPadding(scratch, m, k, groups);
    const auto& expected = oracle.Get(prefix + "_production_output_f16");
    const auto got = output.floats(), want = Floats(expected);
    const double relative = Relative(got, want);
    float max_error = 0;
    for (size_t i = 0; i < got.size(); ++i) max_error = std::max(max_error, std::abs(got[i] - want[i]));
    std::cout << "REAL_GROUPED_GATE_UP M=" << m << " splits=" << scratch.plan.splits
              << " padded_M=" << scratch.plan.padded_rows << " relative=" << relative
              << " max_error=" << max_error << '\n';
    CHECK(relative < 2e-3);
    xpu_test::SameBytes(scratch.parts.download(), std::vector<unsigned char>(expected_parts.data, expected_parts.data + expected_parts.nbytes));
    xpu_test::SameBytes(output.download(), std::vector<unsigned char>(expected.data, expected.data + expected.nbytes));
    // Exercise the actual model-resident wrapper with the same independently
    // captured output, including first-upload host release and reuse across M.
    auto model = vllm::dense_attn::Exl3GroupedMatmulD(d, input.tensor, grouped);
    std::vector<unsigned char> model_bytes(expected.nbytes);
    model.Download(d, model_bytes.data());
    xpu_test::SameBytes(model_bytes, std::vector<unsigned char>(expected.data, expected.data + expected.nbytes));
    CHECK(grouped.trellis.host_released);
    CHECK(grouped.source_map.host_released);
    if (resident) CHECK(grouped.trellis.d_dev.get() == resident);
    resident = grouped.trellis.d_dev.get();
  }
  CHECK(gate.trellis.d_dev == nullptr);
  CHECK(up.trellis.d_dev == nullptr);
  CHECK(vt::GetReferenceTierHits() == 0);
}

namespace {
void CheckRealAttentionGroup(const std::string& family,
                             const std::vector<std::string>& source_names,
                             const std::vector<int>& widths) {
  const char* env = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  if (!env) {
    std::cerr << "SKIP: set VT_B70_EXL3_S1_FIXTURES for real QKVZ/QKV captures.\n";
    std::exit(77);
  }
  const std::filesystem::path root(env);
  Queue gpu(vt::DeviceType::kXPU);
  auto oracle = vllm::SafetensorsFile::Open((root / (family + "_oracle.safetensors")).string());
  std::vector<vllm::Exl3Weight> weights;
  std::vector<const vllm::Exl3Weight*> sources;
  int n = 0;
  constexpr int k = 5120, bits = 4;
  const int groups = static_cast<int>(source_names.size());
  REQUIRE(widths.size() == source_names.size());
  for (size_t s = 0; s < source_names.size(); ++s) {
    auto file = vllm::SafetensorsFile::Open((root / (source_names[s] + ".safetensors")).string());
    const auto get = [&](const std::string& name) -> const vllm::StTensor& { return file.Get(name.substr(2)); };
    const auto has = [&](const std::string& name) {
      const auto key = name.substr(2);
      return std::find(file.Names().begin(), file.Names().end(), key) != file.Names().end();
    };
    auto weight = vllm::dense_loaders::LoadExl3(get, has, "p");
    weight.name = family + "." + source_names[s];
    REQUIRE(weight.InFeatures() == k);
    REQUIRE(weight.OutFeatures() == widths[s]);
    REQUIRE(weight.Bits() == bits);
    n += widths[s];
    weights.push_back(std::move(weight));
    // Source mappings must stay alive after their SafetensorsFile owners die.
  }
  for (const auto& weight : weights) sources.push_back(&weight);
  auto merged = vllm::MergeExl3Weights(sources, family);
  for (const auto& pair : {std::pair{&merged.trellis, "merged_trellis"},
                          std::pair{&merged.suh, "stacked_suh"},
                          std::pair{&merged.svh, "merged_svh"},
                          std::pair{&merged.source_map, "source_map"}}) {
    const auto& expected = oracle.Get(pair.second);
    REQUIRE(pair.first->bytes.size() == expected.nbytes);
    CHECK(std::memcmp(pair.first->bytes.data(), expected.data, expected.nbytes) == 0);
  }
  Buffer packed(gpu.q, DType::kI8, {k/16, n/16, 32*bits});
  Buffer suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
  Buffer map(gpu.q, DType::kI32, {n/128});
  packed.upload(merged.trellis.bytes.data()); suh.upload(merged.suh.bytes.data());
  svh.upload(merged.svh.bytes.data()); map.upload(merged.source_map.bytes.data());
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  vllm::Exl3GroupedWeight model_cache;
  const void* resident = nullptr;
  for (int m : {1, 4, 12, 16, 128}) {
    CAPTURE(family);
    CAPTURE(m);
    const std::string prefix = "m" + std::to_string(m);
    Buffer input(gpu.q, DType::kF16, {m, k}), output(gpu.q, DType::kF16, {m, n});
    const auto& x = oracle.Get("activation_" + prefix + "_fp16");
    REQUIRE(input.bytes == x.nbytes); input.upload(x.data);
    Scratch scratch(gpu.q, m, k, n, bits, groups);
    const auto& parts = oracle.Get(prefix + "_production_split_parts_f32");
    REQUIRE(parts.shape == std::vector<int64_t>{scratch.plan.splits, m, n});
    vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed.tensor,
        suh.tensor, svh.tensor, map.tensor, scratch.had.tensor, scratch.parts.tensor,
        {bits, 2, family.c_str()});
    const auto had = scratch.had.download();
    std::vector<unsigned char> active(size_t(groups) * m * k * 2);
    for (int s = 0; s < groups; ++s) for (int row = 0; row < m; ++row)
      for (int tile = 0; tile < k/16; ++tile)
        std::memcpy(active.data() + (size_t(s*m + row)*k + tile*16)*2,
                    had.data() + ((size_t(s)*(k/16) + tile)*scratch.plan.padded_rows + row)*32, 32);
    const auto& expected_had = oracle.Get(prefix + "_production_input_hadamard_f16");
    xpu_test::SameBytes(active, std::vector<unsigned char>(expected_had.data, expected_had.data + expected_had.nbytes));
    CheckPadding(scratch, m, k, groups);
    const auto& expected = oracle.Get(prefix + "_production_output_f16");
    const auto got = output.floats(), want = Floats(expected);
    const double relative = Relative(got, want);
    float max_error = 0;
    for (size_t i = 0; i < got.size(); ++i) max_error = std::max(max_error, std::abs(got[i] - want[i]));
    std::cout << "REAL_ATTENTION_GROUP family=" << family << " M=" << m
              << " splits=" << scratch.plan.splits << " padded_M=" << scratch.plan.padded_rows
              << " relative=" << relative << " max_error=" << max_error << '\n';
    CHECK(relative < 2e-3);
    xpu_test::SameBytes(scratch.parts.download(), std::vector<unsigned char>(parts.data, parts.data + parts.nbytes));
    xpu_test::SameBytes(output.download(), std::vector<unsigned char>(expected.data, expected.data + expected.nbytes));
    // The translation-unit seam used by actual model QKVZ/QKV initializes its
    // own empty model-lifetime cache, then reuses it across all physical M.
    auto model = vllm::dense_exl3::GroupedLinear(d, input.tensor, sources, model_cache);
    std::vector<unsigned char> model_bytes(expected.nbytes);
    model.Download(d, model_bytes.data());
    xpu_test::SameBytes(model_bytes, std::vector<unsigned char>(expected.data, expected.data + expected.nbytes));
    REQUIRE(model_cache.suh.shape[0] == groups);
    CHECK(model_cache.trellis.host_released);
    CHECK(model_cache.source_map.host_released);
    if (resident) CHECK(model_cache.trellis.d_dev.get() == resident);
    resident = model_cache.trellis.d_dev.get();
  }
  for (const auto& source : weights) CHECK(source.trellis.d_dev == nullptr);
  CHECK(vt::GetReferenceTierHits() == 0);
}
}  // namespace

TEST_CASE("XPU EXL3 producer SmallM: real attention groups QKVZ/QKV M1/4/12/16/128") {
  CheckRealAttentionGroup("gdn_qkvz", {"gdn_qkv", "gdn_z"}, {10240, 6144});
  CheckRealAttentionGroup("attn_qkv", {"attn_q", "attn_k", "attn_v"}, {12288, 1024, 1024});
}

TEST_CASE("XPU EXL3 producer SmallM: full 6bpw head and exact GPU upload") {
  const char* env = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  if (!env) {
    std::cerr << "SKIP: set VT_B70_EXL3_S1_FIXTURES for full target head captures.\n";
    std::exit(77);
  }
  const std::filesystem::path root(env);
  Queue gpu(vt::DeviceType::kXPU);
  auto file = vllm::SafetensorsFile::Open((root / "full_head.safetensors").string());
  auto oracle = vllm::SafetensorsFile::Open((root / "full_head_oracle.safetensors").string());
  const auto get = [&](const std::string& name) -> const vllm::StTensor& { return file.Get(name.substr(2)); };
  const auto has = [&](const std::string& name) {
    const auto key = name.substr(2);
    return std::find(file.Names().begin(), file.Names().end(), key) != file.Names().end();
  };
  auto weight = vllm::dense_loaders::LoadExl3(get, has, "p");
  weight.name = "lm_head";
  constexpr int k = 5120, n = 248320, bits = 6;
  REQUIRE(weight.InFeatures() == k);
  REQUIRE(weight.OutFeatures() == n);
  REQUIRE(weight.Bits() == bits);
  REQUIRE(weight.codebook == 2);
  REQUIRE(weight.trellis.bytes.size() == 953548800);
  const auto& expected_packed = oracle.Get("merged_trellis");
  REQUIRE(expected_packed.nbytes == weight.trellis.bytes.size());
  CHECK(std::memcmp(weight.trellis.bytes.data(), expected_packed.data, expected_packed.nbytes) == 0);
  for (const auto& pair : {std::pair{&weight.suh, "stacked_suh"}, std::pair{&weight.svh, "merged_svh"}}) {
    const auto& expected = oracle.Get(pair.second);
    REQUIRE(expected.nbytes == pair.first->bytes.size());
    CHECK(std::memcmp(pair.first->bytes.data(), expected.data, expected.nbytes) == 0);
  }
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  const auto packed = vllm::dense_attn::ResidentWeight(d, weight.trellis);
  const auto suh = vllm::dense_attn::Reshape(vllm::dense_attn::ResidentWeight(d, weight.suh), {1, k});
  const auto svh = vllm::dense_attn::ResidentWeight(d, weight.svh);
  Buffer map(gpu.q, DType::kI32, {n/128});
  const std::vector<int32_t> zero_map(n/128, 0);
  const auto& expected_map = oracle.Get("source_map");
  REQUIRE(expected_map.nbytes == zero_map.size()*4);
  CHECK(std::memcmp(zero_map.data(), expected_map.data, expected_map.nbytes) == 0);
  map.upload(zero_map.data());
  // Verify every uploaded packed byte in bounded chunks, without creating a
  // second GPU packed owner or an entire-head host download temporary.
  const auto& original = file.Get("trellis");
  constexpr size_t chunk_size = 64*1024*1024;
  std::vector<unsigned char> chunk(chunk_size);
  for (size_t offset = 0; offset < original.nbytes; offset += chunk_size) {
    const size_t size = std::min(chunk_size, original.nbytes - offset);
    d.b.Copy(d.q, chunk.data(), static_cast<const unsigned char*>(packed.data) + offset, size);
    d.b.Synchronize(d.q);
    CHECK(std::memcmp(chunk.data(), original.data + offset, size) == 0);
  }
  const void* resident = weight.trellis.d_dev.get();
  REQUIRE(resident != nullptr);
  vllm::OwnedTensor no_dense_head;
  for (int m : {1, 4, 12, 16, 128}) {
    CAPTURE(m);
    const std::string prefix = "m" + std::to_string(m);
    Buffer input(gpu.q, DType::kF16, {m, k}), output(gpu.q, DType::kF16, {m, n});
    const auto& x = oracle.Get("activation_" + prefix + "_fp16");
    REQUIRE(input.bytes == x.nbytes); input.upload(x.data);
    Scratch scratch(gpu.q, m, k, n, bits, 1);
    const auto& parts = oracle.Get(prefix + "_production_split_parts_f32");
    REQUIRE(parts.shape == std::vector<int64_t>{scratch.plan.splits, m, n});
    vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed, suh, svh,
        map.tensor, scratch.had.tensor, scratch.parts.tensor, {bits, 2, "full_head_direct"});
    const auto had = scratch.had.download();
    std::vector<unsigned char> active(size_t(m)*k*2);
    for (int row = 0; row < m; ++row) for (int tile = 0; tile < k/16; ++tile)
      std::memcpy(active.data() + size_t(row*k + tile*16)*2,
                  had.data() + size_t(tile*scratch.plan.padded_rows + row)*32, 32);
    const auto& expected_had = oracle.Get(prefix + "_production_input_hadamard_f16");
    xpu_test::SameBytes(active, std::vector<unsigned char>(expected_had.data, expected_had.data + expected_had.nbytes));
    CheckPadding(scratch, m, k, 1);
    const auto& expected = oracle.Get(prefix + "_production_output_f16");
    const auto got = output.floats(), want = Floats(expected);
    const double relative = Relative(got, want);
    float max_error = 0;
    for (size_t i = 0; i < got.size(); ++i) max_error = std::max(max_error, std::abs(got[i] - want[i]));
    std::cout << "FULL_HEAD_SMALLM M=" << m << " N=" << n << " bits=" << bits
              << " splits=" << scratch.plan.splits << " padded_M=" << scratch.plan.padded_rows
              << " relative=" << relative << " max_error=" << max_error << '\n';
    CHECK(relative < 2e-3);
    xpu_test::SameBytes(scratch.parts.download(), std::vector<unsigned char>(parts.data, parts.data + parts.nbytes));
    xpu_test::SameBytes(output.download(), std::vector<unsigned char>(expected.data, expected.data + expected.nbytes));
    auto model = vllm::dense_exl3::Linear(d, input.tensor, no_dense_head, weight, DType::kF16);
    std::vector<unsigned char> model_bytes(expected.nbytes);
    model.Download(d, model_bytes.data());
    xpu_test::SameBytes(model_bytes, std::vector<unsigned char>(expected.data, expected.data + expected.nbytes));
    CHECK(weight.trellis.d_dev.get() == resident);
    CHECK(no_dense_head.Empty());
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real Gate/Up SwiGLU: pinned eager activation and model seam") {
  const char* sources = std::getenv("VT_B70_EXL3_S0B_FIXTURES");
  const char* captures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  if (!sources || !captures) {
    std::cerr << "SKIP: set VT_B70_EXL3_S0B_FIXTURES and VT_B70_EXL3_S1_FIXTURES.\n";
    std::exit(77);
  }
  const std::filesystem::path root(captures);
  Queue gpu(vt::DeviceType::kXPU);
  auto gate_file = vllm::SafetensorsFile::Open((std::filesystem::path(sources) / "mlp_gate.safetensors").string());
  auto up_file = vllm::SafetensorsFile::Open((root / "mlp_up.safetensors").string());
  auto projection = vllm::SafetensorsFile::Open((root / "mlp_gate_up_oracle.safetensors").string());
  auto oracle = vllm::SafetensorsFile::Open((root / "mlp_swiglu_oracle.safetensors").string());
  const auto load = [](const vllm::SafetensorsFile& file) {
    const auto get = [&](const std::string& name) -> const vllm::StTensor& { return file.Get(name.substr(2)); };
    const auto has = [&](const std::string& name) {
      const auto key = name.substr(2);
      return std::find(file.Names().begin(), file.Names().end(), key) != file.Names().end();
    };
    return vllm::dense_loaders::LoadExl3(get, has, "p");
  };
  auto gate = load(gate_file), up = load(up_file);
  gate.name = "swiglu.gate"; up.name = "swiglu.up";
  vllm::Exl3GroupedWeight grouped;
  vllm::OwnedTensor no_dense_gate_up;
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  constexpr int k = 5120, intermediate = 17408;
  const void* resident = nullptr;
  for (int m : {1, 4, 12, 16, 128}) {
    CAPTURE(m);
    const std::string prefix = "m" + std::to_string(m);
    const auto& gu = oracle.Get(prefix + "_gate_up_f16");
    const auto& original_gu = projection.Get(prefix + "_production_output_f16");
    REQUIRE(gu.shape == std::vector<int64_t>{m, 2*intermediate});
    REQUIRE(gu.dtype == "F16");
    REQUIRE(gu.nbytes == original_gu.nbytes);
    CHECK(std::memcmp(gu.data, original_gu.data, gu.nbytes) == 0);
    const auto& expected = oracle.Get(prefix + "_production_swiglu_f16");
    REQUIRE(expected.shape == std::vector<int64_t>{m, intermediate});
    REQUIRE(expected.dtype == "F16");
    Buffer packed_output(gpu.q, DType::kF16, {m, 2*intermediate});
    Buffer activation(gpu.q, DType::kF16, {m, intermediate});
    packed_output.upload(gu.data);
    vt::SiluAndMul(gpu.q, activation.tensor, packed_output.tensor);
    const auto got = activation.floats(), want = Floats(expected);
    const double relative = Relative(got, want);
    float max_error = 0;
    for (size_t i = 0; i < got.size(); ++i) max_error = std::max(max_error, std::abs(got[i] - want[i]));
    std::cout << "REAL_SWIGLU M=" << m << " I=" << intermediate
              << " relative=" << relative << " max_error=" << max_error << '\n';
    CHECK(relative < 2e-3);
    xpu_test::SameBytes(activation.download(), std::vector<unsigned char>(expected.data, expected.data + expected.nbytes));
    Buffer input(gpu.q, DType::kF16, {m, k});
    const auto& x = projection.Get("activation_" + prefix + "_fp16");
    REQUIRE(input.bytes == x.nbytes); input.upload(x.data);
    auto model = vllm::dense_exl3::GateUp(d, input.tensor, no_dense_gate_up,
                                        gate, up, intermediate, &grouped);
    std::vector<unsigned char> model_bytes(expected.nbytes);
    model.Download(d, model_bytes.data());
    xpu_test::SameBytes(model_bytes, std::vector<unsigned char>(expected.data, expected.data + expected.nbytes));
    REQUIRE(grouped.trellis.d_dev != nullptr);
    if (resident) CHECK(grouped.trellis.d_dev.get() == resident);
    else resident = grouped.trellis.d_dev.get();
    CHECK(gate.trellis.d_dev == nullptr);
    CHECK(up.trellis.d_dev == nullptr);
    CHECK(no_dense_gate_up.Empty());
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}
