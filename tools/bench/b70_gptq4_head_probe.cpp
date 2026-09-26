#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/xpu.h"

namespace {

struct QueueOwner {
  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  std::vector<void*> allocations;
  void* Allocate(size_t bytes) {
    void* data = vt::Alloc(queue.device, bytes);
    allocations.push_back(data);
    return data;
  }
  ~QueueOwner() {
    vt::GetBackend(queue.device).Synchronize(queue);
    for (void* data : allocations) vt::Free(queue.device, data);
    vt::DestroyQueue(queue);
  }
};

int Run(const std::string& shard_path, const std::string& activation_path,
        const std::string& route) {
  if (route != "reference_matmul_bt" && route != "onednn_f16_cast")
    throw std::runtime_error("route must be reference_matmul_bt or onednn_f16_cast");
  const bool onednn = route == "onednn_f16_cast";
  const auto shard = vllm::SafetensorsFile::Open(shard_path);
  const auto operation = vllm::SafetensorsFile::Open(activation_path);
  const auto& weight = shard.Get("lm_head.weight");
  const auto& activation = operation.Get("activation_fp16");
  const auto& reference = operation.Get("output_reference");
  constexpr int64_t k = 5120, n = 248320;
  if (weight.dtype != "F16" || weight.shape != std::vector<int64_t>{n, k} ||
      activation.dtype != "F16" ||
      activation.shape != std::vector<int64_t>{1, k} ||
      reference.dtype != "F16" ||
      reference.shape != std::vector<int64_t>{1, n})
    throw std::runtime_error("unexpected dense-head fixture dtype or shape");
  QueueOwner owner;
  auto& queue = owner.queue;
  auto& backend = vt::GetBackend(queue.device);
  vt::Tensor device_weight = vt::Tensor::Contiguous(
      owner.Allocate(weight.nbytes), vt::DType::kF16, queue.device, {n, k});
  vt::Tensor device_activation = vt::Tensor::Contiguous(
      owner.Allocate(activation.nbytes), vt::DType::kF16, queue.device, {1, k});
  vt::Tensor device_output = vt::Tensor::Contiguous(
      owner.Allocate(static_cast<size_t>(n) * sizeof(float)),
      vt::DType::kF32, queue.device, {1, n});
  vt::Tensor device_output_f16;
  if (onednn)
    device_output_f16 = vt::Tensor::Contiguous(
        owner.Allocate(static_cast<size_t>(n) * sizeof(uint16_t)),
        vt::DType::kF16, queue.device, {1, n});
  backend.Copy(queue, device_weight.data, weight.data, weight.nbytes);
  backend.Copy(queue, device_activation.data, activation.data,
               activation.nbytes);
  backend.Synchronize(queue);

  const auto invoke = [&] {
    if (onednn) {
      vt::MatmulDenseF16(queue, device_output_f16,
                         device_activation, device_weight);
      vt::CastF32(queue, device_output, device_output_f16);
    } else {
      vt::MatmulBT(queue, device_output, device_activation, device_weight);
    }
  };
  constexpr int warmup_calls = 8;
  for (int index = 0; index < warmup_calls; ++index) invoke();
  backend.Synchronize(queue);
  std::vector<float> output(static_cast<size_t>(n));
  backend.Copy(queue, output.data(), device_output.data,
               output.size() * sizeof(float));
  backend.Synchronize(queue);
  double max_abs_error = 0.0;
  size_t outside_tolerance = 0;
  for (int64_t index = 0; index < n; ++index) {
    uint16_t bits;
    std::memcpy(&bits, reference.data + index * 2, sizeof(bits));
    const double expected = vt::F16ToF32(bits);
    const double actual = output[static_cast<size_t>(index)];
    const double error = std::abs(actual - expected);
    max_abs_error = std::max(max_abs_error, error);
    if (!std::isfinite(actual) || error > 0.02 + 0.01 * std::abs(expected))
      ++outside_tolerance;
  }
  std::vector<double> synchronized_ms;
  for (int sample = 0; sample < 15; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    invoke();
    backend.Synchronize(queue);
    const auto end = std::chrono::steady_clock::now();
    synchronized_ms.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }
  std::sort(synchronized_ms.begin(), synchronized_ms.end());
  std::cout << nlohmann::json({
      {"operation", route},
      {"M", 1}, {"K", k}, {"N", n},
      {"weight_dtype", "f16"}, {"output_dtype", "f32"},
      {"temporary_f16_output_bytes", onednn ? static_cast<size_t>(n) * 2 : 0},
      {"warmup_calls", warmup_calls},
      {"median_synchronized_ms", synchronized_ms[synchronized_ms.size() / 2]},
      {"min_synchronized_ms", synchronized_ms.front()},
      {"max_synchronized_ms", synchronized_ms.back()},
      {"max_abs_error_vs_f16_oracle", max_abs_error},
      {"values_outside_rtol_0_01_atol_0_02", outside_tolerance},
      {"weight_source", shard_path},
      {"activation_source", activation_path}}).dump() << '\n';
  return outside_tolerance == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: b70_gptq4_head_probe SHARD ACTIVITY_CAPTURE "
                 "{reference_matmul_bt|onednn_f16_cast}\n";
    return 2;
  }
  try {
    return Run(argv[1], argv[2], argv[3]);
  } catch (const std::exception& error) {
    std::cerr << "b70_gptq4_head_probe: " << error.what() << '\n';
    return 1;
  }
}
