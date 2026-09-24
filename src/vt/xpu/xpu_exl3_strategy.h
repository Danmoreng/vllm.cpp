#pragma once
#include "vt/tensor.h"
#include <map>
#include <string>
#include <tuple>

namespace vt::xpu::exl3 {
enum class Strategy { kAuto, kReference, kPacked, kFused };
inline constexpr char kKernelVersion[] = "exl3-packed-fused-v1";

struct StrategyDomain {
  int device_id;
  std::string driver, runtime, compiler, kernel;
  auto operator<=>(const StrategyDomain&) const = default;
};
using Shape = std::tuple<int, int64_t, int64_t, int64_t, DType>;  // bits, K, N, M, output

inline Strategy MeasuredStrategy(const StrategyDomain& domain, const Shape& shape) {
  if (domain.device_id != 57891 || domain.driver != "1.17.39758+10" || domain.runtime != "1.17" ||
      domain.compiler != "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)" ||
      domain.kernel != kKernelVersion) return Strategy::kPacked;
  const auto [bits, k, n, m, dtype] = shape;
  if (dtype != DType::kF32 || m < 1 || m > 20) return Strategy::kPacked;
  constexpr uint32_t large = (1u << 16) | (1u << 20);
  // Two independent five-sample medians per strategy after >=200 ms warmup.
  // Admit fusion only for >=10% mean improvement, with neither fused median
  // >5% slower than either packed median. Unmeasured regimes keep packed.
  struct Row { int bits; int64_t k, n; uint32_t rows; };
  constexpr Row winners[] = {
    {3, 5120, 17408, (1u << 2) | (1u << 4) | (1u << 5) | (1u << 8) | large},
    {3, 17408, 5120, (1u << 1) | (1u << 8) | large},
    {4, 5120, 1024, (1u << 1)},
    {4, 5120, 6144, (1u << 1) | large},
    {4, 5120, 10240, (1u << 5) | (1u << 8) | large},
    {4, 5120, 12288, (1u << 4) | (1u << 5) | (1u << 8) | large},
    {4, 5120, 17408, (1u << 2) | (1u << 4) | (1u << 5) | (1u << 8) | large},
    {4, 6144, 5120, (1u << 1) | large},
    {4, 17408, 5120, (1u << 1) | large},
    {5, 6144, 5120, (1u << 1) | (1u << 8) | large},
    {6, 5120, 248320, (1u << 2) | (1u << 4) | (1u << 5) | (1u << 8) | large},
  };
  for (const auto& row : winners)
    if (row.bits == bits && row.k == k && row.n == n && (row.rows & (1u << m)))
      return Strategy::kFused;
  return Strategy::kPacked;
}

// Host-only, bounded memoization. The caller serializes access. Domain changes
// cannot reuse a choice made for another device, driver, compiler or kernel.
class StrategyCache {
 public:
  Strategy Get(const StrategyDomain& domain, const Shape& shape) {
    const auto key = std::make_pair(domain, shape);
    if (const auto found = cache_.find(key); found != cache_.end()) return found->second;
    const auto choice = MeasuredStrategy(domain, shape);
    if (cache_.size() == 512) cache_.clear();
    cache_.emplace(key, choice);
    return choice;
  }
  size_t Size() const { return cache_.size(); }
 private:
  std::map<std::pair<StrategyDomain, Shape>, Strategy> cache_;
};
}  // namespace vt::xpu::exl3
