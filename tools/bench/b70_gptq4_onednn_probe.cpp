// Focused GPTQ-01 operation benchmark and fixture replay.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/xpu/xpu_common.h"
#include "vt/xpu/xpu_gptq4.h"

namespace {
using Json = nlohmann::json;

struct TimerAnchor {
  void operator()() const {}
};

struct FixtureTensor {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

class Fixture {
 public:
  explicit Fixture(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open fixture: " + path);
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(input)), {});
    if (file.size() < sizeof(uint64_t)) throw std::runtime_error("short safetensors file");
    uint64_t header_bytes = 0;
    for (int i = 0; i < 8; ++i) header_bytes |= uint64_t{file[i]} << (8 * i);
    if (header_bytes > file.size() - 8) throw std::runtime_error("invalid safetensors header size");
    const auto* begin = reinterpret_cast<const char*>(file.data() + 8);
    header_ = Json::parse(begin, begin + header_bytes);
    data_start_ = 8 + static_cast<size_t>(header_bytes);
    file_ = std::move(file);
  }

  FixtureTensor Get(const std::string& name) const {
    if (!header_.contains(name)) throw std::runtime_error("missing fixture tensor: " + name);
    const Json& entry = header_.at(name);
    FixtureTensor result;
    result.dtype = entry.at("dtype").get<std::string>();
    result.shape = entry.at("shape").get<std::vector<int64_t>>();
    const auto offsets = entry.at("data_offsets").get<std::array<uint64_t, 2>>();
    if (offsets[0] > offsets[1] || offsets[1] > file_.size() - data_start_)
      throw std::runtime_error("invalid safetensors offsets for " + name);
    const size_t first = data_start_ + static_cast<size_t>(offsets[0]);
    const size_t last = data_start_ + static_cast<size_t>(offsets[1]);
    result.bytes.assign(file_.begin() + first, file_.begin() + last);
    return result;
  }

  bool Has(const std::string& name) const { return header_.contains(name); }

 private:
  size_t data_start_ = 0;
  std::vector<uint8_t> file_;
  Json header_;
};

struct QueueOwner {
  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  ~QueueOwner() { vt::DestroyQueue(queue); }
};

class DeviceBuffer {
 public:
  DeviceBuffer(vt::Queue& queue, vt::DType dtype,
               std::initializer_list<int64_t> shape)
      : queue_(queue), tensor_(vt::Tensor::Contiguous(nullptr, dtype, queue.device, shape)),
        bytes_(tensor_.Bytes()) {
    tensor_.data = vt::Alloc(queue.device, bytes_);
  }
  ~DeviceBuffer() { vt::Free(queue_.device, tensor_.data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  vt::Tensor& tensor() { return tensor_; }

  void Upload(const std::vector<uint8_t>& bytes) {
    if (bytes.size() != bytes_) throw std::runtime_error("fixture tensor byte size mismatch");
    vt::GetBackend(queue_.device).Copy(queue_, tensor_.data, bytes.data(), bytes.size());
    vt::GetBackend(queue_.device).Synchronize(queue_);
  }

  std::vector<uint8_t> Read() {
    std::vector<uint8_t> bytes(bytes_);
    vt::GetBackend(queue_.device).Copy(queue_, bytes.data(), tensor_.data, bytes.size());
    vt::GetBackend(queue_.device).Synchronize(queue_);
    return bytes;
  }

 private:
  vt::Queue& queue_;
  vt::Tensor tensor_;
  size_t bytes_;
};

void RequireTensor(const FixtureTensor& tensor, const char* name,
                   const char* dtype, const std::vector<int64_t>& shape,
                   size_t element_bytes) {
  if (tensor.dtype != dtype || tensor.shape != shape)
    throw std::runtime_error(std::string("unexpected dtype/shape for ") + name);
  size_t elements = 1;
  for (const auto dim : shape) elements *= static_cast<size_t>(dim);
  if (tensor.bytes.size() != elements * element_bytes)
    throw std::runtime_error(std::string("unexpected byte count for ") + name);
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

Json PointerAlignment(const void* pointer) {
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  return {{"mod_64", address % 64}, {"mod_128", address % 128},
          {"mod_256", address % 256}};
}

int Run(const std::string& mode, const std::string& operation_path,
        const std::string& weights_path) {
  QueueOwner owner;
  auto& queue = owner.queue;
  Fixture operation(operation_path);
  Fixture weights(weights_path);
  auto activation = operation.Get("activation_fp16");
  if (activation.dtype != "F16" || activation.shape.size() != 2)
    throw std::runtime_error("activation must be an F16 [M,K] fixture tensor");
  const int64_t m = activation.shape[0], k = activation.shape[1];

  FixtureTensor expected = operation.Has("output_fp16")
      ? operation.Get("output_fp16") : operation.Get("output_reference");
  if (expected.dtype != "F16" || expected.shape.size() != 2 || expected.shape[0] != m)
    throw std::runtime_error("expected output must be F16 [M,N]");
  const int64_t n = expected.shape[1];
  RequireTensor(activation, "activation_fp16", "F16", {m, k}, 2);
  RequireTensor(expected, "expected output", "F16", {m, n}, 2);

  DeviceBuffer device_activation(queue, vt::DType::kF16, {m, k});
  DeviceBuffer device_output(queue, vt::DType::kF16, {m, n});
  device_activation.Upload(activation.bytes);

  std::unique_ptr<DeviceBuffer> device_weights;
  std::unique_ptr<DeviceBuffer> device_scales;
  std::unique_ptr<DeviceBuffer> device_zero_points;
  std::unique_ptr<DeviceBuffer> device_bias;
  int group_size = 128;
  if (mode == "gptq4") {
    auto qweight = weights.Get("qweight_nt_int32");
    auto scales = weights.Get("scales_f16");
    auto zero_points = weights.Get("effective_zero_point_i8");
    RequireTensor(qweight, "qweight_nt_int32", "I32", {n, k / 8}, 4);
    RequireTensor(scales, "scales_f16", "F16", {k / group_size, n}, 2);
    RequireTensor(zero_points, "effective_zero_point_i8", "I8", {1}, 1);
    device_weights = std::make_unique<DeviceBuffer>(queue, vt::DType::kI32,
                                                    std::initializer_list<int64_t>{n, k / 8});
    device_scales = std::make_unique<DeviceBuffer>(queue, vt::DType::kF16,
                                                   std::initializer_list<int64_t>{k / group_size, n});
    device_zero_points = std::make_unique<DeviceBuffer>(queue, vt::DType::kI8,
                                                       std::initializer_list<int64_t>{1});
    device_weights->Upload(qweight.bytes);
    device_scales->Upload(scales.bytes);
    device_zero_points->Upload(zero_points.bytes);
    qweight.bytes.clear();
    scales.bytes.clear();
    zero_points.bytes.clear();
    if (weights.Has("bias_fp16")) {
      auto bias = weights.Get("bias_fp16");
      RequireTensor(bias, "bias_fp16", "F16", {n}, 2);
      device_bias = std::make_unique<DeviceBuffer>(queue, vt::DType::kF16,
                                                   std::initializer_list<int64_t>{n});
      device_bias->Upload(bias.bytes);
    }
  } else {
    std::string weight_name;
    for (const auto& candidate : {"weight_fp16_nk", "weight_full_fp16_nk"})
      if (weights.Has(candidate)) { weight_name = candidate; break; }
    if (weight_name.empty()) throw std::runtime_error("missing dense FP16 [N,K] weight");
    auto dense = weights.Get(weight_name);
    RequireTensor(dense, weight_name.c_str(), "F16", {n, k}, 2);
    device_weights = std::make_unique<DeviceBuffer>(queue, vt::DType::kF16,
                                                    std::initializer_list<int64_t>{n, k});
    device_weights->Upload(dense.bytes);
    dense.bytes.clear();
    if (weights.Has("bias_fp16")) {
      auto bias = weights.Get("bias_fp16");
      RequireTensor(bias, "bias_fp16", "F16", {n}, 2);
      device_bias = std::make_unique<DeviceBuffer>(queue, vt::DType::kF16,
                                                   std::initializer_list<int64_t>{n});
      device_bias->Upload(bias.bytes);
    }
  }

  auto invoke = [&] {
    if (mode == "gptq4")
      vt::MatmulGptq4W4A16(queue, device_output.tensor(), device_activation.tensor(),
                           device_weights->tensor(), device_scales->tensor(),
                           device_zero_points->tensor(), group_size,
                           device_bias ? &device_bias->tensor() : nullptr);
    else
      vt::MatmulDenseF16(queue, device_output.tensor(), device_activation.tensor(),
                         device_weights->tensor(),
                         device_bias ? &device_bias->tensor() : nullptr);
  };

  auto& native = vt::xpu::NativeQueue(queue);
  constexpr size_t kWarmupCalls = 768;
  auto warmup_begin = native.single_task<TimerAnchor>(TimerAnchor{});
  for (size_t warmup_count = 0; warmup_count < kWarmupCalls;
       ++warmup_count) {
    invoke();
    if ((warmup_count + 1) % 64 == 0)
      vt::GetBackend(queue.device).Synchronize(queue);
  }
  auto warmup_end = native.single_task<TimerAnchor>(TimerAnchor{});
  vt::GetBackend(queue.device).Synchronize(queue);
  const auto warmup_begin_ns = warmup_begin.get_profiling_info<
      sycl::info::event_profiling::command_start>();
  const auto warmup_end_ns = warmup_end.get_profiling_info<
      sycl::info::event_profiling::command_end>();
  const double warmup_gpu_ms =
      static_cast<double>(warmup_end_ns - warmup_begin_ns) / 1.0e6;
  const auto after_warmup = vt::xpu::GetGptq4RuntimeStats(queue.device.index);
  const auto actual = device_output.Read();
  double max_error = 0.0, squared_error = 0.0;
  size_t outside_screen = 0;
  for (size_t i = 0; i < expected.bytes.size() / 2; ++i) {
    uint16_t bits;
    std::memcpy(&bits, expected.bytes.data() + i * 2, 2);
    const double ref = vt::F16ToF32(bits);
    const double got = vt::F16ToF32(*reinterpret_cast<const uint16_t*>(actual.data() + i * 2));
    const double error = std::abs(got - ref);
    max_error = std::max(max_error, error);
    squared_error += error * error;
    if (!std::isfinite(got) || !std::isfinite(ref) || error > 0.02 + 0.01 * std::abs(ref))
      ++outside_screen;
  }

  constexpr size_t kSamples = 15;
  std::vector<double> host_enqueue_us, gpu_interval_us;
  host_enqueue_us.reserve(kSamples);
  gpu_interval_us.reserve(kSamples);
  for (size_t i = 0; i < kSamples; ++i) {
    auto start = native.single_task<TimerAnchor>(TimerAnchor{});
    const auto host_start = std::chrono::steady_clock::now();
    invoke();
    const auto host_end = std::chrono::steady_clock::now();
    auto end = native.single_task<TimerAnchor>(TimerAnchor{});
    end.wait_and_throw();
    host_enqueue_us.push_back(
        std::chrono::duration<double, std::micro>(host_end - host_start).count());
    const auto start_ns = start.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end_ns = end.get_profiling_info<sycl::info::event_profiling::command_end>();
    gpu_interval_us.push_back(static_cast<double>(end_ns - start_ns) / 1000.0);
  }
  const auto after_timing = vt::xpu::GetGptq4RuntimeStats(queue.device.index);
  const bool steady_state = after_timing.engine_count == 1 &&
      after_timing.primitive_count == after_warmup.primitive_count &&
      after_timing.scratchpad_allocation_count == after_warmup.scratchpad_allocation_count;

  Json report = {
      {"mode", mode}, {"M", m}, {"K", k}, {"N", n},
      {"warmup_count", kWarmupCalls}, {"warmup_gpu_ms", warmup_gpu_ms},
      {"group_size", mode == "gptq4" ? group_size : 0},
      {"reference_tolerance", "rtol=0.01, atol=0.02"},
      {"max_abs_error", max_error},
      {"rms_error", std::sqrt(squared_error / (expected.bytes.size() / 2))},
      {"values_outside_tolerance", outside_screen},
      {"median_host_enqueue_us", Median(host_enqueue_us)},
      {"median_gpu_interval_us", Median(gpu_interval_us)},
      {"gpu_interval_timer", "SYCL single-task queue markers; includes host submission gaps"},
      {"host_enqueue_us", host_enqueue_us}, {"gpu_interval_us", gpu_interval_us},
      {"engine_count", after_timing.engine_count},
      {"primitive_count", after_timing.primitive_count},
      {"scratchpad_allocation_count", after_timing.scratchpad_allocation_count},
      {"scratchpad_capacity_bytes", after_timing.scratchpad_capacity_bytes},
      {"steady_state_cache_stable", steady_state},
      {"pointer_alignment", {
          {"activation", PointerAlignment(device_activation.tensor().data)},
          {"output", PointerAlignment(device_output.tensor().data)},
          {"weights", PointerAlignment(device_weights->tensor().data)},
          {"scales", device_scales ? PointerAlignment(device_scales->tensor().data) : Json(nullptr)},
          {"zero_points", device_zero_points ? PointerAlignment(device_zero_points->tensor().data) : Json(nullptr)}}},
      {"weights_repacked", false}, {"cpu_compute_path", false}};
  std::cout << std::setw(2) << report << '\n';
  return outside_screen == 0 && steady_state ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 4 ||
        (std::string(argv[1]) != "gptq4" && std::string(argv[1]) != "dense")) {
      std::cerr << "usage: b70_gptq4_probe {gptq4|dense} OP.safetensors "
                   "[WEIGHTS.safetensors]\n";
      return 2;
    }
#ifndef _WIN32
    setenv("VT_XPU_PROFILE", "1", 1);
#endif
    const std::string weights = argc == 4 ? argv[3] : argv[2];
    return Run(argv[1], argv[2], weights);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
