// Platform values follow vLLM platforms/xpu.py @ e126687a9a. Kernels and USM
// ownership are native VT implementations; Torch's memory allocator is not used.
#include "vllm/platforms/interface.h"
#include "vt/xpu.h"

namespace vllm::platforms {
namespace {
class XpuPlatform final : public Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kXPU; }
  Backend& backend() const override { return vt::GetBackend(DeviceType::kXPU); }
  DeviceCapability get_device_capability() const override { return {}; }
  std::vector<DType> supported_dtypes() const override { return {DType::kBF16, DType::kF16, DType::kF32}; }
  bool needs_weight_staging() const override { return true; }
  ResidencyPolicy residency_policy() const override {
    ResidencyPolicy p;
    p.release_host_weights_after_upload = true;
    p.device_memory_total_bytes = vt::xpu::GetMemoryInfo().total_bytes;
    return p;
  }
  // PR03-06 add quantization, recurrent kernels, attention, and model execution.
  bool supports_model_architecture(std::string_view) const override { return false; }
};
struct Registrar {
  Registrar() {
    if (vt::xpu::DeviceCount() == 0) return;
    static XpuPlatform platform;
    RegisterPlatform(DeviceType::kXPU, &platform);
  }
} registrar;
}  // namespace
}  // namespace vllm::platforms
