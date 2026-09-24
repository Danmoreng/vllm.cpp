#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include <cstdlib>
#include <string>

namespace {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;
using xpu_test::Values;
using xpu_test::Close;
using xpu_test::SameBytes;
struct SequentialReference {
  const bool had = std::getenv("VT_GDN_CHUNKED") != nullptr;
  const std::string old = had ? std::getenv("VT_GDN_CHUNKED") : "";
  SequentialReference() { setenv("VT_GDN_CHUNKED", "0", 1); }
  ~SequentialReference() { if (had) setenv("VT_GDN_CHUNKED", old.c_str(), 1); else unsetenv("VT_GDN_CHUNKED"); }
};
vt::Tensor Rows(const vt::Tensor& tensor, int64_t first, int64_t count) {
  auto view = tensor;
  view.data = static_cast<char*>(tensor.data) + first * tensor.stride[0] * vt::SizeOf(tensor.dtype);
  view.shape[0] = count;
  return view;
}
std::vector<float> Normalized(size_t count, int width, int salt) {
  auto data = Values(count, salt);
  for (size_t start = 0; start < count; start += width) {
    float sum = 1e-6f;
    for (int j = 0; j < width; ++j) sum += data[start + j] * data[start + j];
    const float inv = 1.0f / std::sqrt(sum);
    for (int j = 0; j < width; ++j) data[start + j] *= inv;
  }
  return data;
}
}

TEST_CASE("XPU conv prefill: raw history, lengths 0/1/2/3/4/63/64/65, dtype and row strides") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  const std::vector<int32_t> offsets{0, 0, 1, 3, 6, 10, 73, 137, 202};
  const int rows = 202, sequences = 8;
  for (int channels : {17, 10240}) for (auto dtype : {DType::kF32, DType::kBF16, DType::kF16})
    for (bool silu : {false, true}) {
      CAPTURE(channels);
      CAPTURE(dtype);
      CAPTURE(silu);
      std::vector<float> expected;
      std::vector<unsigned char> expected_state;
      for (auto* q : {&cpu.q, &gpu.q}) {
        Buffer x(*q, dtype, {rows, channels + 3}), w(*q, dtype, {channels, 4});
        Buffer out(*q, dtype == DType::kF16 ? DType::kF32 : dtype, {rows, channels});
        Buffer state(*q, DType::kF32, {sequences, channels, 3});
        Buffer qsl(*q, DType::kI32, {9}), init(*q, DType::kI8, {8}), bias(*q, DType::kF32, {channels});
        x.put(Values(rows * (channels + 3))); w.put(Values(channels * 4, 2));
        state.put(Values(sequences * channels * 3, 8)); bias.put(Values(channels, 6));
        const int8_t flags[] = {0, 1, 0, 1, 0, 1, 1, 0}; init.upload(flags); qsl.upload(offsets.data());
        x.tensor.shape[1] = channels;
        vt::CausalConv1dFwd(*q, out.tensor, x.tensor, w.tensor, &bias.tensor, state.tensor,
                             qsl.tensor, init.tensor, vt::CausalConv1dArgs{silu});
        if (q == &cpu.q) { expected = out.floats(); expected_state = state.download(); }
        else { Close(out.floats(), expected, dtype == DType::kBF16 ? 0.008f : 2e-6f);
               SameBytes(state.download(), expected_state); }
      }
    }
}

TEST_CASE("XPU conv decode: permuted/null slots, wider history, in-place output and invalid metadata") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (auto dtype : {DType::kF32, DType::kBF16}) {
    std::vector<float> expected;
    std::vector<unsigned char> expected_state;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer x(*q, dtype, {3, 17}), w(*q, dtype, {17, 4}), out(*q, dtype, {3, 17});
      Buffer state(*q, DType::kF32, {4, 17, 6}), idx(*q, DType::kI32, {3});
      const int32_t indices[] = {2, -1, 0}; idx.upload(indices);
      x.put(Values(51)); out.put(Values(51)); w.put(Values(68, 3)); state.put(Values(4 * 17 * 6, 4));
      auto& target = q == &gpu.q ? x.tensor : out.tensor;
      vt::CausalConv1dUpdate(*q, target, x.tensor, w.tensor, nullptr, state.tensor,
                            vt::CausalConv1dArgs{}, &idx.tensor);
      if (q == &cpu.q) { expected = out.floats(); expected_state = state.download(); }
      else {
        Close(x.floats(), expected, dtype == DType::kBF16 ? 0.008f : 2e-6f);
        SameBytes(state.download(), expected_state);
        for (const auto& invalid : {std::vector<int32_t>{4, -1, 0}, std::vector<int32_t>{2, 2, 0}}) {
          idx.upload(invalid.data());
          CHECK_THROWS(vt::CausalConv1dUpdate(*q, out.tensor, x.tensor, w.tensor, nullptr,
                        state.tensor, vt::CausalConv1dArgs{}, &idx.tensor));
        }
      }
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 0);
}

TEST_CASE("XPU conv full prefill and in-place prefill equal split prefill plus decode") {
  Queue gpu(vt::DeviceType::kXPU);
  for (int length : {1, 2, 3, 4, 63, 64, 65}) {
    constexpr int channels = 17;
    Buffer input(gpu.q, DType::kBF16, {length, channels}), all(gpu.q, DType::kBF16, {length, channels});
    Buffer split(gpu.q, DType::kBF16, {length, channels}), weights(gpu.q, DType::kBF16, {channels, 4});
    Buffer state(gpu.q, DType::kF32, {1, channels, 3}), qsl(gpu.q, DType::kI32, {2}), init(gpu.q, DType::kI32, {1});
    const auto original = Values(length * channels, 3), original_state = Values(channels * 3, 7);
    input.put(original); weights.put(Values(channels * 4)); state.put(original_state);
    int32_t offsets[] = {0, length}, flags[] = {1}; qsl.upload(offsets); init.upload(flags);
    vt::CausalConv1dFwd(gpu.q, all.tensor, input.tensor, weights.tensor, nullptr,
                         state.tensor, qsl.tensor, init.tensor, {});
    const auto expected_state = state.download();
    state.put(original_state);
    vt::CausalConv1dFwd(gpu.q, input.tensor, input.tensor, weights.tensor, nullptr,
                         state.tensor, qsl.tensor, init.tensor, {});
    SameBytes(input.download(), all.download()); SameBytes(state.download(), expected_state);
    input.put(original); state.put(original_state);
    const int prefix = std::min(length, 3); offsets[1] = prefix; qsl.upload(offsets);
    auto out = Rows(split.tensor, 0, prefix);
    vt::CausalConv1dFwd(gpu.q, out, Rows(input.tensor, 0, prefix), weights.tensor, nullptr,
                         state.tensor, qsl.tensor, init.tensor, {});
    for (int t = prefix; t < length; ++t) {
      out = Rows(split.tensor, t, 1);
      vt::CausalConv1dUpdate(gpu.q, out, Rows(input.tensor, t, 1), weights.tensor, nullptr, state.tensor, {});
    }
    SameBytes(split.download(), all.download()); SameBytes(state.download(), expected_state);
  }
}

TEST_CASE("XPU GDN post-conv and gated RMSNorm: actual heads, strided gates, one normalization") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (auto dtype : {DType::kF32, DType::kBF16}) for (bool sigmoid : {false, true}) {
    constexpr int t = 3, hk = 16, hv = 48, dk = 128, dv = 128, channels = 10240;
    std::vector<std::vector<float>> expected;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer conv(*q, dtype, {t, channels}), a(*q, dtype, {t, hv + 3}), b(*q, dtype, {t, hv + 3});
      Buffer al(*q, DType::kF32, {hv}), dt(*q, DType::kF32, {hv});
      Buffer qo(*q, dtype, {t, hk, dk}), ko(*q, dtype, {t, hk, dk}), vo(*q, dtype, {t, hv, dv});
      Buffer go(*q, DType::kF32, {t, hv}), bo(*q, DType::kF32, {t, hv});
      conv.put(Values(t * channels)); a.put(Values(t * (hv + 3), 7, 2)); b.put(Values(t * (hv + 3), 3));
      al.put(Values(hv, 3)); dt.put(Values(hv, 9)); a.tensor.shape[1] = b.tensor.shape[1] = hv;
      vt::GdnPostConv(*q, qo.tensor, ko.tensor, vo.tensor, go.tensor, bo.tensor, conv.tensor,
                      a.tensor, b.tensor, al.tensor, dt.tensor, {1e-6f});
      std::vector<std::vector<float>> actual{qo.floats(), ko.floats(), vo.floats(), go.floats(), bo.floats()};
      Buffer gate(*q, dtype, {t, hv + 1, dv}), weight(*q, dtype, {dv});
      Buffer out(*q, dtype, {t, hv, dv}); gate.put(Values(t * (hv + 1) * dv, 3)); weight.put(Values(dv, 5));
      gate.tensor.shape[1] = hv;
      vt::RmsNormGated(*q, out.tensor, vo.tensor, gate.tensor, weight.tensor, {1e-6f, sigmoid});
      actual.push_back(out.floats());
      if (q == &cpu.q) expected = actual;
      else for (size_t i = 0; i < actual.size(); ++i)
        Close(actual[i], expected[i], dtype == DType::kBF16 && (i < 3 || i == 5) ? 0.008f : 3e-6f);
    }
  }
}

TEST_CASE("XPU GDN recurrence: varlen, Hv/Hk=3, complete output and F32 state") {
  SequentialReference mode;
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (auto dtype : {DType::kF32, DType::kBF16}) for (int length : {1, 2, 3, 4, 63, 64, 65}) {
    const int hk = length == 4 ? 16 : 2, hv = 3 * hk;
    const int dk = length == 4 ? 128 : 9, dv = length == 4 ? 128 : 7, tokens = length + 1;
    CAPTURE(dtype);
    CAPTURE(length);
    std::vector<float> expected, expected_state;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer qi(*q, dtype, {tokens, hk, dk}), ki(*q, dtype, {tokens, hk, dk});
      Buffer vi(*q, dtype, {tokens, hv, dv}), out(*q, dtype, {tokens, hv, dv});
      Buffer g(*q, DType::kF32, {tokens, hv}), beta(*q, DType::kF32, {tokens, hv});
      Buffer state(*q, DType::kF32, {3, hv, dv, dk}), qsl(*q, DType::kI32, {4});
      qi.put(Normalized(tokens * hk * dk, dk, 0)); ki.put(Normalized(tokens * hk * dk, dk, 3));
      vi.put(Values(tokens * hv * dv, 4)); g.put(std::vector<float>(tokens * hv, -0.13f));
      beta.put(std::vector<float>(tokens * hv, 0.61f)); state.put(Values(3 * hv * dv * dk, 5));
      const int32_t offsets[] = {0, 1, tokens, tokens}; qsl.upload(offsets);
      vt::GdnPrefill(*q, out.tensor, qi.tensor, ki.tensor, vi.tensor, g.tensor, beta.tensor,
                     state.tensor, qsl.tensor, {1.0f / std::sqrt(float(dk))});
      if (q == &cpu.q) { expected = out.floats(); expected_state = state.floats(); }
      else {
        Close(out.floats(), expected, dtype == DType::kBF16 ? 0.008f : 2e-5f);
        Close(state.floats(), expected_state, 2e-5f);
        const int32_t invalid[] = {0, tokens, 1, tokens}; qsl.upload(invalid);
        CHECK_THROWS(vt::GdnPrefill(*q, out.tensor, qi.tensor, ki.tensor, vi.tensor, g.tensor,
                      beta.tensor, state.tensor, qsl.tensor, {1.0f}));
      }
    }
  }
}

TEST_CASE("XPU GDN full prefill equals split prefill plus decode, including every state value") {
  Queue gpu(vt::DeviceType::kXPU);
  for (int length : {1, 2, 3, 4, 63, 64, 65}) {
    constexpr int hk = 2, hv = 6, dk = 9, dv = 7;
    Buffer qi(gpu.q, DType::kBF16, {length, hk, dk}), ki(gpu.q, DType::kBF16, {length, hk, dk});
    Buffer vi(gpu.q, DType::kBF16, {length, hv, dv}), all(gpu.q, DType::kBF16, {length, hv, dv});
    Buffer split(gpu.q, DType::kBF16, {length, hv, dv});
    Buffer g(gpu.q, DType::kF32, {length, hv}), beta(gpu.q, DType::kF32, {length, hv});
    Buffer state(gpu.q, DType::kF32, {1, hv, dv, dk}), qsl(gpu.q, DType::kI32, {2});
    qi.put(Normalized(length * hk * dk, dk, 0)); ki.put(Normalized(length * hk * dk, dk, 3));
    vi.put(Values(length * hv * dv, 4)); g.put(std::vector<float>(length * hv, -0.13f));
    beta.put(std::vector<float>(length * hv, 0.61f)); state.put(Values(hv * dv * dk, 5));
    int32_t offsets[] = {0, length}; qsl.upload(offsets);
    vt::GdnPrefill(gpu.q, all.tensor, qi.tensor, ki.tensor, vi.tensor, g.tensor, beta.tensor,
                   state.tensor, qsl.tensor, {1.0f / 3.0f});
    const auto expected_state = state.download(); state.put(Values(hv * dv * dk, 5));
    const int prefix = std::min(length, 3); offsets[1] = prefix; qsl.upload(offsets);
    auto o = Rows(split.tensor, 0, prefix);
    vt::GdnPrefill(gpu.q, o, Rows(qi.tensor, 0, prefix), Rows(ki.tensor, 0, prefix),
                   Rows(vi.tensor, 0, prefix), Rows(g.tensor, 0, prefix), Rows(beta.tensor, 0, prefix),
                   state.tensor, qsl.tensor, {1.0f / 3.0f});
    for (int t = prefix; t < length; ++t) {
      o = Rows(split.tensor, t, 1);
      vt::GdnDecode(gpu.q, o, Rows(qi.tensor, t, 1), Rows(ki.tensor, t, 1), Rows(vi.tensor, t, 1),
                    Rows(g.tensor, t, 1), Rows(beta.tensor, t, 1), state.tensor, {1.0f / 3.0f});
    }
    SameBytes(split.download(), all.download()); SameBytes(state.download(), expected_state);
  }
}

TEST_CASE("XPU GDN state gather/scatter and decode: permutation, null slot, preserved cache rows") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (auto cache_type : {DType::kF32, DType::kBF16}) {
    std::vector<float> expected_cache, expected_work, expected_output, expected_state;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer cache(*q, cache_type, {4, 6, 6}), work(*q, DType::kF32, {3, 6, 3});
      Buffer ids(*q, DType::kI32, {3}), flags(*q, DType::kI32, {3});
      const int32_t indices[] = {2, 0, 3}, initial[] = {1, 0, 1}; ids.upload(indices); flags.upload(initial);
      cache.put(Values(4 * 6 * 6));
      vt::GdnStateGather(*q, work.tensor, cache.tensor, ids.tensor, &flags.tensor);
      const auto gathered = work.floats(); work.put(Values(3 * 6 * 3, 9));
      vt::GdnStateScatter(*q, cache.tensor, work.tensor, ids.tensor);
      Buffer qi(*q, DType::kBF16, {3, 2, 9}), ki(*q, DType::kBF16, {3, 2, 9});
      Buffer vi(*q, DType::kBF16, {3, 6, 7}), out(*q, DType::kBF16, {3, 6, 7});
      Buffer g(*q, DType::kF32, {3, 6}), beta(*q, DType::kF32, {3, 6}), state(*q, DType::kF32, {4, 6, 7, 9});
      qi.put(Normalized(3 * 2 * 9, 9, 0)); ki.put(Normalized(3 * 2 * 9, 9, 3)); vi.put(Values(3 * 6 * 7));
      g.put(std::vector<float>(18, -0.13f)); beta.put(std::vector<float>(18, 0.61f)); state.put(Values(4 * 6 * 7 * 9));
      const int32_t slots[] = {2, -1, 0}; ids.upload(slots);
      vt::GdnDecode(*q, out.tensor, qi.tensor, ki.tensor, vi.tensor, g.tensor, beta.tensor, state.tensor,
                    {1.0f / 3.0f}, &ids.tensor);
      if (q == &cpu.q) {
        expected_cache = cache.floats(); expected_work = gathered;
        expected_output = out.floats(); expected_state = state.floats();
      } else {
        CHECK(cache.floats() == expected_cache); CHECK(gathered == expected_work);
        Close(out.floats(), expected_output, 0.008f); Close(state.floats(), expected_state, 2e-5f);
      }
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 0);
}

TEST_CASE("XPU compressed conv: direct BF16 cache equals the F32 working-copy path") {
  Queue gpu(vt::DeviceType::kXPU);
  for (bool prefill : {false, true}) {
    constexpr auto cache_type = DType::kBF16;
    constexpr int channels = 17, slots = 4, tokens = 4;
    Buffer x(gpu.q, DType::kF32, {tokens, channels}), weight(gpu.q, DType::kF32, {channels, 4});
    Buffer state(gpu.q, cache_type, {slots, channels, 6});
    Buffer working(gpu.q, DType::kF32, {slots, channels, 6}), stored(gpu.q, cache_type, {slots, channels, 6});
    Buffer out(gpu.q, DType::kF32, {tokens, channels}), reference(gpu.q, DType::kF32, {tokens, channels});
    Buffer indices(gpu.q, DType::kI32, {tokens}), qsl(gpu.q, DType::kI32, {slots + 1}), flags(gpu.q, DType::kI8, {slots});
    state.put(Values(slots * channels * 6, 2)); weight.put(Values(channels * 4, 3));
    out.put(std::vector<float>(tokens * channels, -1)); reference.put(std::vector<float>(tokens * channels, -1));
    const int32_t offsets[] = {0, 0, 1, 3, 4}; const int8_t initial[] = {1, 0, 1, 1};
    qsl.upload(offsets); flags.upload(initial);
    for (int step = 0; step < 8; ++step) {
      x.put(Values(tokens * channels, step, .071f));
      int32_t ids[tokens]; for (int i = 0; i < tokens; ++i) ids[i] = i == step % slots ? -1 : (i + step) % slots;
      indices.upload(ids); vt::Copy(gpu.q, working.tensor, state.tensor);
      if (prefill) {
        vt::CausalConv1dFwd(gpu.q, out.tensor, x.tensor, weight.tensor, nullptr, state.tensor, qsl.tensor, flags.tensor, {});
        vt::CausalConv1dFwd(gpu.q, reference.tensor, x.tensor, weight.tensor, nullptr, working.tensor, qsl.tensor, flags.tensor, {});
      } else {
        vt::CausalConv1dUpdate(gpu.q, out.tensor, x.tensor, weight.tensor, nullptr, state.tensor, {}, &indices.tensor);
        vt::CausalConv1dUpdate(gpu.q, reference.tensor, x.tensor, weight.tensor, nullptr, working.tensor, {}, &indices.tensor);
      }
      vt::Copy(gpu.q, stored.tensor, working.tensor);
      SameBytes(out.download(), reference.download()); SameBytes(state.download(), stored.download());
    }
  }
}
