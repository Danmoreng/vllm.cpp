#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gptq4_weight.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
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

TEST_CASE("XPU GPTQ captured attention QKV merged equals split and Python") {
  const char* fixture_dir = std::getenv("VLLM_CPP_GPTQ4_ORACLE_DIR");
  if (fixture_dir == nullptr) {
    MESSAGE("Set VLLM_CPP_GPTQ4_ORACLE_DIR for the captured QKV operation test");
    return;
  }
  const auto weights_file = vllm::SafetensorsFile::Open(
      std::string(fixture_dir) + "/attention_qkv_weights.safetensors");
  constexpr int64_t k = 5120;
  constexpr int64_t n = 14336;
  constexpr int64_t qn = 12288;
  constexpr int64_t kn = 1024;
  constexpr int64_t vn = 1024;
  vllm::Gptq4Weight merged;
  merged.k = k;
  merged.n = n;
  merged.qweight = vllm::dense_loaders::MakeOwned(vt::DType::kI32,
                                                   {n, k / 8});
  merged.scales = vllm::dense_loaders::MakeOwned(vt::DType::kF16,
                                                  {k / 128, n});
  merged.zero_point = vllm::dense_loaders::MakeOwned(vt::DType::kI8, {1});
  const auto& packed = weights_file.Get("qweight_nt_int32");
  const auto& scales = weights_file.Get("scales_f16");
  const auto& zero = weights_file.Get("effective_zero_point_i8");
  REQUIRE(packed.nbytes == merged.qweight.bytes.size());
  REQUIRE(scales.nbytes == merged.scales.bytes.size());
  REQUIRE(zero.nbytes == 1);
  std::memcpy(merged.qweight.bytes.data(), packed.data, packed.nbytes);
  std::memcpy(merged.scales.bytes.data(), scales.data, scales.nbytes);
  merged.zero_point.bytes.data()[0] = zero.data[0];

  const auto split = [&merged](int64_t begin, int64_t width) {
    vllm::Gptq4Weight part;
    part.k = k;
    part.n = width;
    part.qweight = vllm::dense_loaders::MakeOwned(vt::DType::kI32,
                                                   {width, k / 8});
    part.scales = vllm::dense_loaders::MakeOwned(vt::DType::kF16,
                                                  {k / 128, width});
    part.zero_point = vllm::dense_loaders::MakeOwned(vt::DType::kI8, {1});
    std::memcpy(part.qweight.bytes.data(),
                merged.qweight.bytes.data() + begin * (k / 8) * 4,
                part.qweight.bytes.size());
    for (int64_t group = 0; group < k / 128; ++group)
      std::memcpy(part.scales.bytes.data() + group * width * 2,
                  merged.scales.bytes.data() + (group * n + begin) * 2,
                  width * 2);
    part.zero_point.bytes.data()[0] = merged.zero_point.bytes.data()[0];
    return part;
  };
  auto qpart = split(0, qn);
  auto kpart = split(qn, kn);
  auto vpart = split(qn + kn, vn);
  QueueOwner owner;
  auto& queue = owner.queue;
  const auto merged_gpu = vllm::PrepareGptq4Resident(merged, queue);
  const auto qgpu = vllm::PrepareGptq4Resident(qpart, queue);
  const auto kgpu = vllm::PrepareGptq4Resident(kpart, queue);
  const auto vgpu = vllm::PrepareGptq4Resident(vpart, queue);
  for (const auto* fixture : {"attention_qkv_m1.safetensors",
                              "attention_qkv_m16.safetensors"}) {
    const auto sample = vllm::SafetensorsFile::Open(
        std::string(fixture_dir) + "/" + fixture);
    const auto& input = sample.Get("activation_fp16");
    const auto& expected = sample.Get("output_fp16");
    REQUIRE(input.shape.size() == 2);
    const int64_t m = input.shape[0];
    REQUIRE(input.shape[1] == k);
    REQUIRE(expected.shape.size() == 2);
    REQUIRE(expected.shape[0] == m);
    REQUIRE(expected.shape[1] == n);
    Buffer activation(queue, vt::DType::kF16, {m, k});
    Buffer combined(queue, vt::DType::kF16, {m, n});
    Buffer qout(queue, vt::DType::kF16, {m, qn});
    Buffer kout(queue, vt::DType::kF16, {m, kn});
    Buffer vout(queue, vt::DType::kF16, {m, vn});
    activation.Upload(input.data, input.nbytes);
    vt::MatmulGptq4W4A16(queue, combined.tensor, activation.tensor,
                         merged_gpu.qweight, merged_gpu.scales,
                         merged_gpu.zero_point, 128);
    vt::MatmulGptq4W4A16(queue, qout.tensor, activation.tensor,
                         qgpu.qweight, qgpu.scales, qgpu.zero_point, 128);
    vt::MatmulGptq4W4A16(queue, kout.tensor, activation.tensor,
                         kgpu.qweight, kgpu.scales, kgpu.zero_point, 128);
    vt::MatmulGptq4W4A16(queue, vout.tensor, activation.tensor,
                         vgpu.qweight, vgpu.scales, vgpu.zero_point, 128);
    vt::GetBackend(queue.device).Synchronize(queue);
    const auto merged_result = combined.Read();
    const auto q_result = qout.Read();
    const auto k_result = kout.Read();
    const auto v_result = vout.Read();
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const float actual = ReadHalf(merged_result.data(), row * n + col);
        const float reference = ReadHalf(expected.data, row * n + col);
        const float separate = col < qn
            ? ReadHalf(q_result.data(), row * qn + col)
            : col < qn + kn
                ? ReadHalf(k_result.data(), row * kn + col - qn)
                : ReadHalf(v_result.data(), row * vn + col - qn - kn);
        const float tolerance = 0.02f + 0.01f * std::abs(reference);
        CHECK(std::abs(actual - reference) <= tolerance);
        CHECK(std::abs(actual - separate) <= tolerance);
      }
    }
  }
}

namespace {

void CheckCapturedTwoWayMerge(vt::Queue& queue, const std::string& directory,
                              const std::string& family,
                              int64_t first_width, int64_t second_width) {
  constexpr int64_t k = 5120;
  const int64_t n = first_width + second_width;
  const auto captured = vllm::SafetensorsFile::Open(
      directory + "/" + family + "_weights.safetensors");
  const auto& packed = captured.Get("qweight_nt_int32");
  const auto& scales = captured.Get("scales_f16");
  const auto& zero = captured.Get("effective_zero_point_i8");
  vllm::Gptq4Weight merged;
  merged.k = k;
  merged.n = n;
  merged.qweight = vllm::dense_loaders::MakeOwned(vt::DType::kI32,
                                                  {n, k / 8});
  merged.scales = vllm::dense_loaders::MakeOwned(vt::DType::kF16,
                                                 {k / 128, n});
  merged.zero_point = vllm::dense_loaders::MakeOwned(vt::DType::kI8, {1});
  REQUIRE(packed.nbytes == merged.qweight.bytes.size());
  REQUIRE(scales.nbytes == merged.scales.bytes.size());
  REQUIRE(zero.nbytes == 1);
  std::memcpy(merged.qweight.bytes.data(), packed.data, packed.nbytes);
  std::memcpy(merged.scales.bytes.data(), scales.data, scales.nbytes);
  merged.zero_point.bytes.data()[0] = zero.data[0];
  const auto slice = [&merged, n](int64_t begin, int64_t width) {
    vllm::Gptq4Weight part;
    part.k = k;
    part.n = width;
    part.qweight = vllm::dense_loaders::MakeOwned(vt::DType::kI32,
                                                  {width, k / 8});
    part.scales = vllm::dense_loaders::MakeOwned(vt::DType::kF16,
                                                 {k / 128, width});
    part.zero_point = vllm::dense_loaders::MakeOwned(vt::DType::kI8, {1});
    std::memcpy(part.qweight.bytes.data(),
                merged.qweight.bytes.data() + begin * (k / 8) * 4,
                part.qweight.bytes.size());
    for (int64_t group = 0; group < k / 128; ++group)
      std::memcpy(part.scales.bytes.data() + group * width * 2,
                  merged.scales.bytes.data() + (group * n + begin) * 2,
                  width * 2);
    part.zero_point.bytes.data()[0] = merged.zero_point.bytes.data()[0];
    return part;
  };
  auto first = slice(0, first_width);
  auto second = slice(first_width, second_width);
  const auto merged_gpu = vllm::PrepareGptq4Resident(merged, queue);
  const auto first_gpu = vllm::PrepareGptq4Resident(first, queue);
  const auto second_gpu = vllm::PrepareGptq4Resident(second, queue);
  CHECK(merged.ResidentBytes() == first.ResidentBytes() +
                                    second.ResidentBytes() - 1);
  CHECK(!merged.qweight.HasHostBytes());
  for (const char* batch : {"m1", "m16"}) {
    const auto sample = vllm::SafetensorsFile::Open(
        directory + "/" + family + "_" + batch + ".safetensors");
    const auto& input = sample.Get("activation_fp16");
    const auto& expected = sample.Get("output_fp16");
    REQUIRE(input.shape.size() == 2);
    const int64_t m = input.shape[0];
    REQUIRE(input.shape[1] == k);
    REQUIRE(expected.shape.size() == 2);
    REQUIRE(expected.shape[0] == m);
    REQUIRE(expected.shape[1] == n);
    Buffer activation(queue, vt::DType::kF16, {m, k});
    Buffer combined(queue, vt::DType::kF16, {m, n});
    Buffer first_out(queue, vt::DType::kF16, {m, first_width});
    Buffer second_out(queue, vt::DType::kF16, {m, second_width});
    activation.Upload(input.data, input.nbytes);
    vt::MatmulGptq4W4A16(queue, combined.tensor, activation.tensor,
                         merged_gpu.qweight, merged_gpu.scales,
                         merged_gpu.zero_point, 128);
    vt::MatmulGptq4W4A16(queue, first_out.tensor, activation.tensor,
                         first_gpu.qweight, first_gpu.scales,
                         first_gpu.zero_point, 128);
    vt::MatmulGptq4W4A16(queue, second_out.tensor, activation.tensor,
                         second_gpu.qweight, second_gpu.scales,
                         second_gpu.zero_point, 128);
    vt::GetBackend(queue.device).Synchronize(queue);
    const auto actual = combined.Read();
    const auto first_result = first_out.Read();
    const auto second_result = second_out.Read();
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const float reference = ReadHalf(expected.data, row * n + col);
        const float combined_value = ReadHalf(actual.data(), row * n + col);
        const float separate = col < first_width
            ? ReadHalf(first_result.data(), row * first_width + col)
            : ReadHalf(second_result.data(),
                       row * second_width + col - first_width);
        const float tolerance = 0.02f + 0.01f * std::abs(reference);
        CHECK(std::abs(combined_value - reference) <= tolerance);
        CHECK(std::abs(combined_value - separate) <= tolerance);
      }
    }
  }
}

}  // namespace

TEST_CASE("XPU GPTQ captured GDN and MLP merges equal split and Python") {
  const char* directory = std::getenv("VLLM_CPP_GPTQ4_ORACLE_DIR");
  if (directory == nullptr) {
    MESSAGE("Set VLLM_CPP_GPTQ4_ORACLE_DIR for captured merge tests");
    return;
  }
  QueueOwner owner;
  CheckCapturedTwoWayMerge(owner.queue, directory, "gdn_qkvz", 10240, 6144);
  CheckCapturedTwoWayMerge(owner.queue, directory, "mlp_gate_up", 17408, 17408);
}

TEST_CASE("XPU GPTQ full checkpoint streams one resident per merge") {
  const char* checkpoint = std::getenv("VLLM_CPP_GPTQ4_CHECKPOINT_DIR");
  if (checkpoint == nullptr) {
    MESSAGE("Set VLLM_CPP_GPTQ4_CHECKPOINT_DIR for full XPU staging");
    return;
  }
  QueueOwner owner;
  const auto config = vllm::LoadHfConfig(
      std::string(checkpoint) + "/config.json");
  std::vector<vllm::SafetensorsFile> shards;
  for (int i = 1; i <= 5; ++i)
    shards.push_back(vllm::SafetensorsFile::Open(
        std::string(checkpoint) + "/model-0000" + std::to_string(i) +
        "-of-00005.safetensors"));
  const auto model = vllm::LoadQwen3_5Dense(shards, config, &owner.queue);
  REQUIRE(model.gptq4_checkpoint);
  REQUIRE(model.layers.size() == 64);
  size_t resident_bytes = 0;
  for (const auto& layer : model.layers) {
    resident_bytes += layer.gptq4.ResidentBytes();
    const auto check = [](const vllm::Gptq4Weight& weight) {
      if (weight.k == 0) return;
      CHECK(weight.resident);
      CHECK(weight.qweight.d_dev != nullptr);
      CHECK(weight.scales.d_dev != nullptr);
      CHECK(weight.zero_point.d_dev != nullptr);
      CHECK(!weight.qweight.HasHostBytes());
      CHECK(!weight.scales.HasHostBytes());
      CHECK(!weight.zero_point.HasHostBytes());
    };
    check(layer.gptq4.gdn_qkvz);
    check(layer.gptq4.gdn_out);
    check(layer.gptq4.attn_qkv);
    check(layer.gptq4.attn_out);
    check(layer.gptq4.mlp_gate_up);
    check(layer.gptq4.mlp_down);
  }
  CHECK(resident_bytes > 10ULL * 1024 * 1024 * 1024);
  CHECK(resident_bytes < 20ULL * 1024 * 1024 * 1024);
  CHECK(model.Gptq4ResidentBytes() == resident_bytes);
  const size_t uploaded = vllm::load_stats::Snapshot().device_upload_bytes;
  for (const auto& layer : model.layers) layer.gptq4.PrepareResident(owner.queue);
  CHECK(vllm::load_stats::Snapshot().device_upload_bytes == uploaded);
  CHECK(model.lm_head.dtype == vt::DType::kF16);
  CHECK(model.embed_tokens.dtype == vt::DType::kF16);
}
