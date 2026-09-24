#include "exl3_fixture.h"
#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include <array>

namespace {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;
using xpu_test::SameBytes;

std::vector<uint16_t> Input(int64_t elements) {
  exl3_test::Rng rng;
  std::vector<uint16_t> result(elements);
  for (auto& v : result) v = vt::F32ToF16(rng.next(0.15f));
  return result;
}
std::vector<unsigned char> Gemm(vt::Queue& q, const exl3_test::Exl3Fixture& fixture,
                                int rows, DType output, bool alias, int cb) {
  const auto k = fixture.k, n = fixture.n;
  Buffer a(q, DType::kF16, {rows, k}), ah(q, DType::kF16, {rows, k});
  Buffer b(q, DType::kI8, {k / 16, n / 16, 32 * fixture.bits});
  Buffer suh(q, DType::kF16, {k}), svh(q, DType::kF16, {n});
  Buffer c(q, output, {rows, n});
  a.upload(Input(rows * k).data());
  b.upload(fixture.trellis.data(), true);
  suh.upload(fixture.suh.data(), true); svh.upload(fixture.svh.data(), true);
  auto& scratch = alias ? a.tensor : ah.tensor;
  vt::Exl3Gemm(q, c.tensor, a.tensor, b.tensor, suh.tensor, svh.tensor, scratch,
               vt::Exl3GemmArgs{fixture.bits, cb});
  if (output == DType::kF32) {
    Buffer bf(q, DType::kBF16, {rows, n});
    vt::CastBf16(q, bf.tensor, c.tensor);
    auto result = c.download();
    const auto converted = bf.download();
    result.insert(result.end(), converted.begin(), converted.end());
    return result;
  }
  return c.download();
}
}

TEST_CASE("XPU EXL3 Had128: F16/F32, pre/post scales, in-place, model width") {
  Queue host(vt::DeviceType::kCPU), device(vt::DeviceType::kXPU);
  for (auto dtype : {DType::kF16, DType::kF32}) for (int rows : {1, 3})
    for (int cols : {128, 256, 5120}) for (int mode = 0; mode < 3; ++mode)
      for (bool alias : {false, true}) {
        CAPTURE(dtype);
        CAPTURE(rows);
        CAPTURE(cols);
        CAPTURE(mode);
        CAPTURE(alias);
        const auto input = Input(rows * cols);
        std::vector<float> input32(input.size());
        for (size_t i = 0; i < input.size(); ++i) input32[i] = vt::F16ToF32(input[i]);
        std::vector<uint16_t> scales(cols);
        for (int i = 0; i < cols; ++i) scales[i] = vt::F32ToF16(i % 3 ? 0.513f : -0.739f);
        std::vector<unsigned char> expected;
        for (auto* q : {&host.q, &device.q}) {
          Buffer in(*q, dtype, {rows, cols}), out(*q, dtype, {rows, cols}), sc(*q, DType::kF16, {cols});
          in.upload(dtype == DType::kF16 ? static_cast<const void*>(input.data()) : input32.data());
          sc.upload(scales.data(), true);
          vt::Exl3HadArgs args{mode == 1 ? &sc.tensor : nullptr, mode == 2 ? &sc.tensor : nullptr, 0.73f};
          vt::Exl3HadR128(*q, alias ? in.tensor : out.tensor, in.tensor, args);
          auto actual = (alias ? in : out).download();
          if (q == &host.q) expected = actual; else SameBytes(actual, expected);
        }
      }
}

TEST_CASE("XPU EXL3 GEMM: bits 1-8, tail M, scratch alias, F16 and F32 to BF16") {
  Queue host(vt::DeviceType::kCPU), device(vt::DeviceType::kXPU);
  for (int bits = 1; bits <= 8; ++bits) for (int rows : {1, 2, 17})
    for (auto dtype : {DType::kF16, DType::kF32}) {
      CAPTURE(bits);
      CAPTURE(rows);
      CAPTURE(dtype);
      auto fixture = exl3_test::MakeFixture(256, 384, bits, 0x542113u + bits);
      const auto expected = Gemm(host.q, fixture, rows, dtype, false, 2);
      SameBytes(Gemm(device.q, fixture, rows, dtype, rows % 2 != 0, 2), expected);
    }
  for (int cb : {0, 1}) {
    CAPTURE(cb);
    auto fixture = exl3_test::MakeFixture(128, 128, 4, 0x673523u);
    SameBytes(Gemm(device.q, fixture, 3, DType::kF32, true, cb),
              Gemm(host.q, fixture, 3, DType::kF32, false, cb));
  }
  CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, device.q.device.type));
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 0);
}
