#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/xpu/xpu_common.h"
#include "vt/xpu/xpu_gptq4.h"

namespace {

struct QueueOwner {
  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  ~QueueOwner() { vt::DestroyQueue(queue); }
};

struct Buffer {
  vt::Queue& queue;
  vt::Tensor tensor;
  size_t bytes;

  Buffer(vt::Queue& q, vt::DType dtype, std::initializer_list<int64_t> shape)
      : queue(q), tensor(vt::Tensor::Contiguous(nullptr, dtype, q.device, shape)),
        bytes(tensor.Bytes()) {
    tensor.data = vt::Alloc(q.device, bytes);
  }
  ~Buffer() { vt::Free(queue.device, tensor.data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  void Upload(const void* data, size_t length) {
    REQUIRE(length == bytes);
    vt::GetBackend(queue.device).Copy(queue, tensor.data, data, length);
    vt::GetBackend(queue.device).Synchronize(queue);
  }
  std::vector<uint8_t> Read() {
    std::vector<uint8_t> result(bytes);
    vt::GetBackend(queue.device).Copy(queue, result.data(), tensor.data, bytes);
    vt::GetBackend(queue.device).Synchronize(queue);
    return result;
  }
};

float ReadHalf(const uint8_t* bytes, size_t index) {
  uint16_t bits;
  std::memcpy(&bits, bytes + index * sizeof(bits), sizeof(bits));
  return vt::F16ToF32(bits);
}

void RequireClose(const std::vector<uint8_t>& actual,
                  const std::vector<float>& expected) {
  REQUIRE(actual.size() == expected.size() * sizeof(uint16_t));
  for (size_t i = 0; i < expected.size(); ++i) {
    CAPTURE(i);
    CHECK(std::abs(ReadHalf(actual.data(), i) - expected[i]) <=
          0.02f + 0.01f * std::abs(expected[i]));
  }
}

}  // namespace

TEST_CASE("XPU GPTQ W4A16 packed nibbles, scales and runtime reuse") {
  QueueOwner owner;
  auto& q = owner.queue;
  constexpr int m = 1, k = 5120, n = 1024, group = 128;
  Buffer activation(q, vt::DType::kF16, {m, k});
  Buffer qweight(q, vt::DType::kI32, {n, k / 8});
  Buffer scales(q, vt::DType::kF16, {k / group, n});
  Buffer zero_point(q, vt::DType::kI8, {1});
  Buffer output(q, vt::DType::kF16, {m, n});

  std::vector<uint16_t> a_bits(m * k);
  std::vector<float> a(m * k);
  for (int i = 0; i < m * k; ++i) {
    a[i] = (static_cast<int>(i % 17) - 8) / 16.0f;
    a_bits[i] = vt::F32ToF16(a[i]);
    a[i] = vt::F16ToF32(a_bits[i]);
  }
  std::vector<uint32_t> packed(n * (k / 8));
  std::vector<int> codes(n * k);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < k; ++col) {
      const int code = 5 + (row * 3 + col * 5) % 7;
      codes[row * k + col] = code;
      packed[row * (k / 8) + col / 8] |=
          static_cast<uint32_t>(code) << (4 * (col % 8));
    }
  }
  std::vector<uint16_t> scale_bits((k / group) * n);
  std::vector<float> scale_values((k / group) * n);
  for (int group_id = 0; group_id < k / group; ++group_id) {
    for (int row = 0; row < n; ++row) {
      const auto offset = group_id * n + row;
      scale_values[offset] = vt::F16ToF32(vt::F32ToF16(
          0.0625f + 0.015625f * static_cast<float>(row % 7)));
      scale_bits[offset] = vt::F32ToF16(scale_values[offset]);
    }
  }
  const int8_t zp = 8;
  activation.Upload(a_bits.data(), a_bits.size() * sizeof(a_bits[0]));
  qweight.Upload(packed.data(), packed.size() * sizeof(packed[0]));
  scales.Upload(scale_bits.data(), scale_bits.size() * sizeof(scale_bits[0]));
  zero_point.Upload(&zp, sizeof(zp));

  std::vector<float> expected(m * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      for (int inner = 0; inner < k; ++inner) {
        const int group_id = inner / group;
        const float weight = (codes[col * k + inner] - zp) *
                             scale_values[group_id * n + col];
        expected[row * n + col] += a[row * k + inner] * weight;
      }
    }
  }

  const auto before = vt::xpu::GetGptq4RuntimeStats(q.device.index);
  for (int iteration = 0; iteration < 5; ++iteration)
    vt::MatmulGptq4W4A16(q, output.tensor, activation.tensor, qweight.tensor,
                         scales.tensor, zero_point.tensor, group);
  vt::GetBackend(q.device).Synchronize(q);
  RequireClose(output.Read(), expected);
  const auto after = vt::xpu::GetGptq4RuntimeStats(q.device.index);
  CHECK(after.engine_count == 1);
  CHECK(after.primitive_count == before.primitive_count + 1);
  CHECK(after.scratchpad_allocation_count >= before.scratchpad_allocation_count);
  CHECK(after.primitive_count ==
        vt::xpu::GetGptq4RuntimeStats(q.device.index).primitive_count);
}

TEST_CASE("XPU GPTQ dense F16 matmul with optional bias") {
  QueueOwner owner;
  auto& q = owner.queue;
  constexpr int m = 1, k = 5120, n = 96;
  Buffer activation(q, vt::DType::kF16, {m, k});
  Buffer weight(q, vt::DType::kF16, {n, k});
  Buffer bias(q, vt::DType::kF16, {n});
  Buffer output(q, vt::DType::kF16, {m, n});
  Buffer output_no_bias(q, vt::DType::kF16, {m, n});
  std::vector<uint16_t> a_bits(m * k), w_bits(n * k), b_bits(n);
  for (size_t i = 0; i < a_bits.size(); ++i)
    a_bits[i] = vt::F32ToF16((static_cast<int>(i % 23) - 11) / 16.0f);
  for (size_t i = 0; i < w_bits.size(); ++i)
    w_bits[i] = vt::F32ToF16((static_cast<int>(i % 19) - 9) / 32.0f);
  for (size_t i = 0; i < b_bits.size(); ++i)
    b_bits[i] = vt::F32ToF16((static_cast<int>(i % 7) - 3) / 8.0f);
  activation.Upload(a_bits.data(), a_bits.size() * sizeof(a_bits[0]));
  weight.Upload(w_bits.data(), w_bits.size() * sizeof(w_bits[0]));
  bias.Upload(b_bits.data(), b_bits.size() * sizeof(b_bits[0]));

  std::vector<float> expected(m * n), expected_no_bias(m * n);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int inner = 0; inner < k; ++inner)
        sum += vt::F16ToF32(a_bits[row * k + inner]) *
               vt::F16ToF32(w_bits[col * k + inner]);
      expected_no_bias[row * n + col] = sum;
      expected[row * n + col] = sum + vt::F16ToF32(b_bits[col]);
    }
  }
  vt::MatmulDenseF16(q, output.tensor, activation.tensor, weight.tensor,
                     &bias.tensor);
  vt::GetBackend(q.device).Synchronize(q);
  RequireClose(output.Read(), expected);
  vt::MatmulDenseF16(q, output_no_bias.tensor, activation.tensor, weight.tensor);
  vt::GetBackend(q.device).Synchronize(q);
  RequireClose(output_no_bias.Read(), expected_no_bias);
  const auto stats = vt::xpu::GetGptq4RuntimeStats(q.device.index);
  CHECK(stats.engine_count == 1);
  CHECK(stats.primitive_count == 3);
}
