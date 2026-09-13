// The EXL3 reconstruct dispatch through the production seam — QUANT-EXL3,
// ISSUE-LOCAL-01M2BYPW7YTC2B2MY023ETTKQ2 item 1.
//
// #3150 ported exllamav3's `AUTO_RECONSTRUCT_THRESHOLD` (exl3.py:10,135) into
// `dense_attn::Exl3MatmulD`: M > 144 routes to `vt::Exl3ReconstructGemm`. That
// op is registered for CUDA only (`src/vt/cuda/cuda_exl3.cu`), while
// `vt::Exl3Gemm` is registered for CPU, ROCm and Vulkan too. A device-blind
// threshold therefore refused every EXL3 prefill above 144 tokens on those
// backends, which served every M through `Exl3Gemm` before #3150. Upstream's
// reconstruct path is CUDA-only as well, so the correct mirror on a backend with
// no reconstruct kernel is the cooperative kernel at every M.
//
// This suite enters through `Exl3MatmulD` on a CPU queue, which is the seam
// `layers::Exl3LinearMethod::Apply` delegates to, so it measures the dispatch a
// model forward reaches and not the kernels by hand. CPU-only; runs in CI.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

#include "exl3_fixture.h"

namespace {

using exl3_test::Exl3ChainF64;
using exl3_test::Exl3Fixture;
using exl3_test::MakeFixture;
using exl3_test::Rms;
using exl3_test::Rng;
using exl3_test::UlpF16;

vllm::Exl3Weight WrapFixture(const Exl3Fixture& f) {
  vllm::Exl3Weight w;
  w.codebook = 1;  // the codebook Exl3ChainF64 decodes by default
  const auto bytes_of = [](const std::vector<uint16_t>& v) {
    return vllm::OwnedBytes(std::vector<uint8_t>(
        reinterpret_cast<const uint8_t*>(v.data()),
        reinterpret_cast<const uint8_t*>(v.data()) + v.size() * 2));
  };
  w.trellis.dtype = vt::DType::kI8;
  w.trellis.rank = 3;
  w.trellis.shape[0] = f.k / 16;
  w.trellis.shape[1] = f.n / 16;
  w.trellis.shape[2] = 32 * f.bits;
  w.trellis.bytes = bytes_of(f.trellis);
  w.suh.dtype = vt::DType::kF16;
  w.suh.rank = 1;
  w.suh.shape[0] = f.k;
  w.suh.bytes = bytes_of(f.suh);
  w.svh.dtype = vt::DType::kF16;
  w.svh.rank = 1;
  w.svh.shape[0] = f.n;
  w.svh.bytes = bytes_of(f.svh);
  return w;
}

}  // namespace

TEST_CASE("exl3 dispatch: M > 144 on a backend with no reconstruct kernel serves through Exl3Gemm") {
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = b.CreateQueue();
  vllm::dense_attn::Dev d{b, q};

  // The premise, asserted rather than assumed: if a CPU reconstruct kernel is
  // ever registered, this case no longer tests the fallback and must be revised.
  REQUIRE_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCPU));
  REQUIRE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kCPU));

  // 145 is the first M upstream sends to reconstruct_hgemm (`rows <= 144` keeps
  // the cooperative kernel, exl3.py:135).
  const int64_t m = 145, k = 128, n = 256;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x3150A11u);
  const vllm::Exl3Weight w = WrapFixture(f);

  Rng rng;
  rng.s = 0x0C0FFEEu;
  std::vector<float> x(static_cast<size_t>(m * k));
  // Through fp16 first: the seam stages the activation to fp16, and the
  // reference must read the same values.
  for (auto& v : x) v = vt::F16ToF32(vt::F32ToF16(rng.next(1.0f)));

  vt::EnableOpProviderCallStats(true);
  const unsigned long long before =
      vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kCPU).selections;

  vllm::dense_attn::DBuf xb(d, vt::DType::kF32, {m, k}, x.data());
  std::vector<float> got(static_cast<size_t>(m * n), 0.0f);
  std::string refusal;
  try {
    vllm::dense_attn::DBuf out = vllm::dense_attn::Exl3MatmulD(d, xb.t(), w, vt::DType::kF32);
    out.Download(d, got.data());
  } catch (const std::exception& e) {
    refusal = e.what();
  }
  INFO("Exl3MatmulD refusal: " << refusal);
  REQUIRE(refusal.empty());

  // Positive signal that the cooperative kernel served the call, not merely
  // that something returned numbers.
  const unsigned long long after =
      vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kCPU).selections;
  vt::EnableOpProviderCallStats(false);
  CHECK(after > before);

  const std::vector<double> ref = Exl3ChainF64(f, x, m);
  const double rms = Rms(ref);
  REQUIRE(rms > 0.0);
  double sq = 0.0, worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double dlt = static_cast<double>(got[i]) - ref[i];
    sq += dlt * dlt;
    worst = std::max(worst, std::fabs(dlt));
  }
  const double rel_rms = std::sqrt(sq / static_cast<double>(ref.size())) / rms;
  MESSAGE("Exl3MatmulD M=145 cpu vs f64: rel_rms=", rel_rms, " worst=", worst);
  // The CPU arm's own tier-3 bound (test_exl3_gemm.cpp, spec `## W2 design` §1).
  CHECK(rel_rms <= 1.0e-3);
  CHECK(worst <= 8.0 * UlpF16(rms));

  b.DestroyQueue(q);
}

// ─── QUANT-EXL3 W7: one reconstruct scratch per stream ──────────────────────
//
// `.agents/specs/quant-exl3-recon-scratch.md` §7, ISSUE-LOCAL-01M2DW8CXYEWWMJSZZ6GRH48SZ.
// #3150 drew the reconstruct weight scratch, fp16 [K, min(N, 32768)], from the
// DevicePool on every M > 144 call. A CUDA-graph decode step PINS every block its
// eager step demanded, so each lazily captured verify width pinned one block of
// every scratch class (about 786 MiB per slot on Qwen3.8-27B EXL3) and the GB10
// host ran out of memory at c = 32. These cases enter through `Exl3MatmulD`, the
// seam `layers::Exl3LinearMethod::Apply` delegates to, on a real CUDA queue.
// They SKIP without a CUDA reconstruct kernel, and still assert, so a CPU run
// never reports a skip as a pass.

namespace {

bool HasReconstructCuda() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kCUDA);
    return vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCUDA);
  } catch (const std::runtime_error&) {
    return false;
  }
}

struct W7Shape {
  int64_t k;
  int64_t n;
};

// Every (K, N) a W7 case uses, including one N > 32768 so the sliced scratch
// ([K, 32768], smaller than the weight) is covered. All are multiples of 128.
constexpr W7Shape kW7Shapes[] = {{256, 512}, {512, 256}, {384, 1024}, {256, 32896}};

std::vector<uint16_t> RandomF16(size_t count, uint32_t seed) {
  Rng rng;
  rng.s = seed;
  std::vector<uint16_t> v(count);
  for (auto& x : v) x = vt::F32ToF16(rng.next(1.0f));
  return v;
}

int64_t ScratchCols(int64_t n) { return n <= 32768 ? n : 32768; }

std::vector<uint8_t> Download(vllm::dense_attn::Dev d, vllm::dense_attn::DBuf& buf) {
  std::vector<uint8_t> out(buf.bytes());
  buf.Download(d, out.data());
  return out;
}

}  // namespace

TEST_CASE("exl3 dispatch cuda: the reconstruct scratch draws no DevicePool block") {
  if (!HasReconstructCuda()) {
    MESSAGE("SKIPPED, no CUDA reconstruct kernel: QUANT-EXL3 W7 test (1) is PENDING. "
            "Reproduce with: rc run -d dgx:gpu0 -- ctest -R test_exl3_matmul_dispatch -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCUDA));
    return;
  }
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q = cb.CreateQueue();
  vllm::dense_attn::Dev d{cb, q};
  const int64_t m = 145;  // the first M upstream sends to reconstruct_hgemm

  // The measurement is only meaningful if no OTHER buffer of the step shares a
  // class with a scratch: the x input and a_had are [M, K] f16 and the output is
  // [M, N] f16. The premise is asserted, not assumed.
  std::vector<size_t> other_classes, scratch_classes;
  for (const W7Shape& s : kW7Shapes) {
    other_classes.push_back(vllm::DevicePool::SizeClassForTest(static_cast<size_t>(m * s.k * 2)));
    other_classes.push_back(vllm::DevicePool::SizeClassForTest(static_cast<size_t>(m * s.n * 2)));
    scratch_classes.push_back(
        vllm::DevicePool::SizeClassForTest(static_cast<size_t>(s.k * ScratchCols(s.n) * 2)));
  }
  for (size_t sc : scratch_classes)
    for (size_t oc : other_classes) REQUIRE(sc != oc);

  std::vector<Exl3Fixture> fixtures;
  std::vector<vllm::Exl3Weight> weights;
  for (const W7Shape& s : kW7Shapes) fixtures.push_back(MakeFixture(s.k, s.n, 3, 0x3150B07u));
  for (const Exl3Fixture& f : fixtures) weights.push_back(WrapFixture(f));

  const auto run_all = [&]() {
    for (size_t i = 0; i < weights.size(); ++i) {
      const std::vector<uint16_t> x = RandomF16(static_cast<size_t>(m * kW7Shapes[i].k), 7u + i);
      vllm::dense_attn::DBuf xb(d, vt::DType::kF16, {m, kW7Shapes[i].k}, x.data());
      vllm::dense_attn::DBuf out =
          vllm::dense_attn::Exl3MatmulD(d, xb.t(), weights[i], vt::DType::kF16);
      cb.Synchronize(q);
    }
  };

  vllm::DevicePool& pool = vllm::ActivePool(cb);
  run_all();  // the eager warm step: uploads, and grows whatever it grows
  pool.MarkStepBoundary();
  run_all();
  run_all();
  const vllm::DevicePool::StepDemand demand = pool.StepDemandProfile();

  for (const auto& entry : demand) {
    for (size_t i = 0; i < scratch_classes.size(); ++i) {
      INFO("shape K=" << kW7Shapes[i].k << " N=" << kW7Shapes[i].n << ": the pool's demand "
                      "profile holds the reconstruct scratch class " << entry.first
                      << " with peak " << entry.second);
      CHECK(entry.first != scratch_classes[i]);
    }
  }
  cb.DestroyQueue(q);
}

TEST_CASE("exl3 dispatch cuda: the persistent scratch leaves the reconstruct output byte-identical") {
  if (!HasReconstructCuda()) {
    MESSAGE("SKIPPED, no CUDA reconstruct kernel: QUANT-EXL3 W7 test (2) is PENDING. "
            "Reproduce with: rc run -d dgx:gpu0 -- ctest -R test_exl3_matmul_dispatch -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCUDA));
    return;
  }
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q = cb.CreateQueue();
  vllm::dense_attn::Dev d{cb, q};
  using vllm::dense_attn::DBuf;

  // M = 145 and 300 take the UNFUSED reconstruct (M < 1024), and 1024 the FUSED
  // one (exl3.py:176-184), so both sub-paths read the scratch.
  for (const int64_t m : {int64_t{145}, int64_t{300}, int64_t{1024}}) {
    for (size_t i = 0; i < std::size(kW7Shapes); ++i) {
      const int64_t k = kW7Shapes[i].k, n = kW7Shapes[i].n;
      if (m == 1024 && n > 32768) continue;  // keeps the f16 [M, N] output small
      CAPTURE(m);
      CAPTURE(k);
      CAPTURE(n);
      const Exl3Fixture f = MakeFixture(k, n, 3, 0x3150B07u + static_cast<uint32_t>(i));
      const vllm::Exl3Weight w = WrapFixture(f);
      const std::vector<uint16_t> x = RandomF16(static_cast<size_t>(m * k), 0xB07u + i);
      DBuf xb(d, vt::DType::kF16, {m, k}, x.data());

      // Through the production seam.
      DBuf got = vllm::dense_attn::Exl3MatmulD(d, xb.t(), w, vt::DType::kF16);
      const std::vector<uint8_t> got_bytes = Download(d, got);

      // The pre-change call, spelled out: the same op with a scratch DBuf drawn
      // for this call, exactly as `Exl3MatmulD` did at #3150.
      vt::Tensor trellis = vllm::dense_attn::ResidentWeight(d, w.trellis);
      vt::Tensor suh = vllm::dense_attn::ResidentWeight(d, w.suh);
      vt::Tensor svh = vllm::dense_attn::ResidentWeight(d, w.svh);
      vt::Exl3GemmArgs args;
      args.bits = w.Bits();
      args.codebook = w.codebook;
      DBuf a_had(d, vt::DType::kF16, {m, k});
      DBuf w_scratch(d, vt::DType::kF16, {k, ScratchCols(n)});
      DBuf old(d, vt::DType::kF16, {m, n});
      vt::Exl3ReconstructGemm(q, old.t(), xb.t(), trellis, suh, svh, a_had.t(), w_scratch.t(),
                              args);
      const std::vector<uint8_t> old_bytes = Download(d, old);
      CHECK(got_bytes == old_bytes);

      // And an INDEPENDENT reference within the bound the existing gates hold
      // each sub-path to (test_exl3_gemm.cpp, rel RMS 1.0e-3). The unfused path
      // is checked against the cooperative kernel at the same M, as that file's
      // cross-path gate does. The FUSED path is checked against the f64 chain, as
      // that file's fused gate does: measured on thor:gpu0 at base 909125d16, the
      // fused path sits 1.01e-3 from `Exl3Gemm` at M = 1024 on every shape here,
      // because the two errors do not share a sign, while each is within 1.0e-3
      // of f64. That is a property of the kernels before this change, not of it.
      std::vector<double> ref;
      if (m < 1024) {
        DBuf a_had2(d, vt::DType::kF16, {m, k});
        DBuf coop(d, vt::DType::kF32, {m, n});
        vt::Exl3GemmArgs cargs = args;
        cargs.force_gemv = 0;
        vt::Exl3Gemm(q, coop.t(), xb.t(), trellis, suh, svh, a_had2.t(), cargs);
        std::vector<float> coop_h(static_cast<size_t>(m * n));
        coop.Download(d, coop_h.data());
        ref.assign(coop_h.begin(), coop_h.end());
      } else {
        std::vector<float> xf(x.size());
        for (size_t j = 0; j < x.size(); ++j) xf[j] = vt::F16ToF32(x[j]);
        ref = Exl3ChainF64(f, xf, m);
      }
      double num = 0.0, den = 0.0;
      for (size_t j = 0; j < ref.size(); ++j) {
        uint16_t h = 0;
        std::memcpy(&h, got_bytes.data() + 2 * j, 2);
        const double dl = static_cast<double>(vt::F16ToF32(h)) - ref[j];
        num += dl * dl;
        den += ref[j] * ref[j];
      }
      REQUIRE(den > 0.0);
      const double rel = std::sqrt(num / den);
      MESSAGE("W7 M=", m, " K=", k, " N=", n, ": reconstruct via seam vs ",
              std::string(m < 1024 ? "Exl3Gemm" : "f64 chain"), " rel_rms=", rel);
      CHECK(rel <= 1.0e-3);
    }
  }
  cb.DestroyQueue(q);
}

// LAST IN THE FILE ON PURPOSE: it opens a capture on the stream, and a defect in
// the code under test could leave that stream unusable for a later case.
TEST_CASE("exl3 dispatch cuda: growing the reconstruct scratch while capturing is refused by name") {
  if (!HasReconstructCuda()) {
    MESSAGE("SKIPPED, no CUDA reconstruct kernel: QUANT-EXL3 W7 test (3) is PENDING. "
            "Reproduce with: rc run -d dgx:gpu0 -- ctest -R test_exl3_matmul_dispatch -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCUDA));
    return;
  }
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kCUDA);
  REQUIRE(cb.SupportsGraphCapture());
  vt::Queue q = cb.CreateQueue();
  vllm::dense_attn::Dev d{cb, q};
  using vllm::dense_attn::DBuf;

  // Larger than every scratch the earlier cases asked for ([256, 32768] is the
  // largest), so the capture has to GROW whatever this stream inherited even if
  // the driver hands this queue a recycled stream handle.
  const int64_t m = 145, k = 1024, n = 32896;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x3150C4Du);
  const vllm::Exl3Weight w = WrapFixture(f);
  const std::vector<uint16_t> x = RandomF16(static_cast<size_t>(m * k), 0xC4Du);
  DBuf xb(d, vt::DType::kF16, {m, k}, x.data());

  // Everything the captured call needs that is NOT the scratch is made ready
  // eagerly, so the capture can only fail on the scratch: the weight upload
  // (an M <= 144 call takes the cooperative kernel and allocates no scratch), and
  // one free pool block for each of the call's [M, K] and [M, N] buffers.
  {
    DBuf x4(d, vt::DType::kF16, {4, k});
    x4.Zero(d);
    DBuf warm = vllm::dense_attn::Exl3MatmulD(d, x4.t(), w, vt::DType::kF16);
    DBuf p1(d, vt::DType::kF16, {m, k});
    DBuf p2(d, vt::DType::kF16, {m, n});
    // At #3150 the scratch was a pool block; give it a free one too, so the
    // pre-change code reaches the end of the call rather than a driver malloc.
    DBuf p3(d, vt::DType::kF16, {k, ScratchCols(n)});
    cb.Synchronize(q);
  }

  std::string refusal;
  cb.BeginCapture(q);
  try {
    DBuf out = vllm::dense_attn::Exl3MatmulD(d, xb.t(), w, vt::DType::kF16);
  } catch (const std::exception& e) {
    refusal = e.what();
  }
  void* graph = nullptr;
  try {
    graph = cb.EndCaptureGraph(q);
  } catch (const std::exception& e) {
    MESSAGE("EndCaptureGraph after the call: ", e.what());
  }
  if (graph != nullptr) cb.DestroyGraph(graph);

  INFO("Exl3MatmulD under capture: '" << refusal << "'");
  CHECK(refusal.find("exl3 reconstruct scratch") != std::string::npos);
  CHECK(refusal.find("CUDA graph capture") != std::string::npos);

  // The refusal is recoverable: the same call runs eagerly afterwards.
  DBuf out = vllm::dense_attn::Exl3MatmulD(d, xb.t(), w, vt::DType::kF16);
  cb.Synchronize(q);
  cb.DestroyQueue(q);
}
