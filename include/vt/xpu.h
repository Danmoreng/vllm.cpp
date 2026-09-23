// Native SYCL/Level Zero resource queries. No vendor types cross this interface.
#pragma once
#include <cstddef>
#include <string>

namespace vt::xpu {
struct MemoryInfo {
  size_t total_bytes = 0;       // device capacity, never process allocation count
  size_t budget_bytes = 0;      // process allocation ceiling, <= total
  size_t allocated_bytes = 0;   // live device USM owned by this backend
  size_t pinned_bytes = 0;      // host USM; excluded from device usage
  size_t free_bytes = 0;        // live driver report, valid only if free_known
  bool free_known = false;
};
int DeviceCount() noexcept;
MemoryInfo GetMemoryInfo(int index = 0);
std::string DeviceDescription(int index = 0);
}  // namespace vt::xpu
