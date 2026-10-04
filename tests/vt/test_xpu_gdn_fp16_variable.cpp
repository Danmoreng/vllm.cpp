#include "xpu_test_helpers.h"
#include "vt/gdn_fp16_plan.h"
#include "vt/unaligned.h"
#include "vt/xpu.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include <cstdlib>
#include <iostream>

namespace {
using xpu_test::Buffer;
using vt::DType;
void Same(const Buffer& buffer, const vllm::StTensor& expected) {
  xpu_test::SameBytes(buffer.download(),
      std::vector<unsigned char>(expected.data, expected.data + expected.nbytes));
}
void State(const Buffer& owner, const vllm::StTensor& expected) {
  const auto bytes = owner.download();
  constexpr size_t slot_bytes = size_t(48) * 128 * 128 * 4;
  REQUIRE(expected.nbytes == slot_bytes);
  xpu_test::SameBytes(std::vector<unsigned char>(bytes.begin() + 4 * slot_bytes, bytes.end()),
      std::vector<unsigned char>(expected.data, expected.data + expected.nbytes));
  bool unchanged = true;
  for (size_t i = 0; i < 4 * slot_bytes; i += 4) unchanged &= vt::LoadUnaligned<float>(bytes.data() + i) == 7.0f;
  CHECK(unchanged);
}
void ConvState(const Buffer& owner, const vllm::StTensor& expected) {
  const auto bytes = owner.download();
  constexpr size_t slot_words = 10240 * 3;
  REQUIRE(expected.nbytes == slot_words * 2);
  bool active = true, untouched = true;
  for (size_t i = 0; i < 4 * slot_words; ++i)
    untouched &= vt::LoadUnaligned<uint16_t>(bytes.data() + i * 2) == vt::F32ToF16(-2.0f);
  for (size_t c = 0; c < 10240; ++c) for (size_t j = 0; j < 3; ++j)
    active &= vt::LoadUnaligned<uint16_t>(bytes.data() + (4 * slot_words + c * 3 + j) * 2) ==
              vt::LoadUnaligned<uint16_t>(expected.data + (j * 10240 + c) * 2);
  CHECK(active); CHECK(untouched);
}
}

TEST_CASE("XPU GDN FP16 variable C1: exact prefill tails continuation and stable call order") {
  const char* env = std::getenv("VT_B70_EXL3_VARIABLE_GDN_FIXTURE");
  if (!env) { std::cerr << "SKIP: VT_B70_EXL3_VARIABLE_GDN_FIXTURE required\n"; std::exit(77); }
  auto f = vllm::SafetensorsFile::Open(env);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer weights(gpu.q, DType::kF16, {10240, 4}), alog(gpu.q, DType::kF32, {48});
  Buffer bias(gpu.q, DType::kF32, {48});
  weights.upload(f.Get("weight_conv1d_weight").data); alog.upload(f.Get("weight_A_log").data);
  std::vector<float> bias_values(48);
  for (size_t h = 0; h < 48; ++h)
    bias_values[h] = vt::F16ToF32(vt::LoadUnaligned<uint16_t>(f.Get("weight_dt_bias").data + 2 * h));
  bias.put(bias_values);
  Buffer state(gpu.q, DType::kF32, {5, 48, 128, 128});
  Buffer conv(gpu.q, DType::kF16, {5, 10240, 3});
  Buffer qsl(gpu.q, DType::kI32, {2}), initial(gpu.q, DType::kI32, {1});
  Buffer index(gpu.q, DType::kI32, {1});
  const int32_t slot = 4; index.upload(&slot);
  auto active_state = state.tensor.Slice(0, 4, 5);
  auto active_conv = conv.tensor.Slice(0, 4, 5);

  // Decode first: an exactly zero update has zero output/state regardless of
  // the finite gate values. It must reserve the full 4K capacity immediately.
  state.put(std::vector<float>(state.bytes / 4, 0));
  Buffer zero(gpu.q, DType::kF16, {1, 10240}), gates(gpu.q, DType::kF16, {1, 48});
  Buffer warm_out(gpu.q, DType::kF16, {1, 48, 128});
  zero.put(std::vector<float>(10240, 0)); gates.put(std::vector<float>(48, 0));
  vt::GdnPackedDecode(gpu.q, warm_out.tensor, zero.tensor, gates.tensor, gates.tensor,
      alog.tensor, bias.tensor, state.tensor, index.tensor, {1.0f / std::sqrt(128.0f)});
  const auto warm = warm_out.floats();
  CHECK(std::all_of(warm.begin(), warm.end(), [](float v) { return v == 0; }));
  const auto reserved = vt::xpu::GetMemoryInfo().native_gdn_workspace_bytes;
  REQUIRE(reserved == vt::kGdnFp16ReservationBytes);

  for (int length : {127, 128, 129, 256, 4096}) {
    auto state_values = std::vector<float>(state.bytes / 4, 7);
    std::fill(state_values.begin() + 4 * 48 * 128 * 128, state_values.end(), 0);
    state.put(state_values);
    std::vector<float> conv_values(conv.bytes / 2, -2);
    std::fill(conv_values.begin() + 4 * 10240 * 3, conv_values.end(), 0); conv.put(conv_values);
    for (const std::string suffix : {"", "_append3", "_d1"}) {
      const std::string label = "p" + std::to_string(length) + suffix;
      CAPTURE(label);
      const bool prefill = suffix != "_d1";
      const int rows = suffix.empty() ? length : prefill ? 3 : 1;
      Buffer qkvz(gpu.q, DType::kF16, {rows, 16384}), ba(gpu.q, DType::kF16, {rows, 96});
      qkvz.upload(f.Get(label + "_qkvz").data); ba.upload(f.Get(label + "_ba").data);
      auto mixed = qkvz.tensor.Slice(1, 0, 10240);
      auto raw_b = ba.tensor.Slice(1, 0, 48), raw_a = ba.tensor.Slice(1, 48, 96);
      Buffer output(gpu.q, DType::kF16, {rows, 48, 128});
      Buffer postconv(gpu.q, prefill ? DType::kF32 : DType::kF16, {rows, 10240});
      if (prefill) {
        const int32_t offsets[] = {0, rows}, keep = suffix.empty() ? 0 : 1;
        qsl.upload(offsets); initial.upload(&keep);
        vt::CausalConv1dFwd(gpu.q, postconv.tensor, mixed, weights.tensor, nullptr,
            active_conv, qsl.tensor, initial.tensor, {true});
        Buffer q(gpu.q, DType::kF16, {rows, 16, 128}), k(gpu.q, DType::kF16, {rows, 16, 128});
        Buffer v(gpu.q, DType::kF16, {rows, 48, 128});
        Buffer g(gpu.q, DType::kF32, {rows, 48}), beta(gpu.q, DType::kF32, {rows, 48});
        vt::GdnPostConv(gpu.q, q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor,
            postconv.tensor, raw_a, raw_b, alog.tensor, bias.tensor, {1e-6f, true});
        Same(q, f.Get(label + "_q")); Same(k, f.Get(label + "_k"));
        Same(v, f.Get(label + "_v")); Same(beta, f.Get(label + "_beta"));
        vt::GdnPrefillRawGate(gpu.q, output.tensor, q.tensor, k.tensor, v.tensor, raw_a,
            beta.tensor, alog.tensor, bias.tensor, active_state, qsl.tensor, {1.0f});
      } else {
        vt::CausalConv1dUpdate(gpu.q, postconv.tensor, mixed, weights.tensor, nullptr,
            conv.tensor, {true}, &index.tensor);
        Same(postconv, f.Get(label + "_decode_mixed"));
        vt::GdnPackedDecode(gpu.q, output.tensor, postconv.tensor, raw_a, raw_b,
            alog.tensor, bias.tensor, state.tensor, index.tensor, {1.0f / std::sqrt(128.0f)});
      }
      Same(output, f.Get(label + "_core"));
      State(state, f.Get(label + "_state")); ConvState(conv, f.Get(label + "_conv_state"));
      CHECK(vt::xpu::GetMemoryInfo().native_gdn_workspace_bytes == reserved);
      std::cout << "VARIABLE_GDN_NATIVE_EXACT " << label << " reservation_bytes=" << reserved << '\n';
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}
