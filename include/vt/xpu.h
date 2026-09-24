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
  size_t exl3_workspace_bytes = 0;  // included in allocated_bytes, shared by all queues
  size_t gdn_workspace_bytes = 0;   // included in allocated_bytes, chunk-64 scratch
  size_t sampling_workspace_bytes = 0; // included in allocated_bytes, top-k/top-p scratch
  size_t attention_workspace_bytes = 0; // included in allocated_bytes, split-KV scratch
  size_t peak_allocated_bytes = 0; // backend-tracked device high-water mark (excludes driver allocations)
  size_t graph_count = 0, graph_nodes = 0;
  size_t graph_device_bytes = 0; // validation buffers plus SYCL-reported graph memory; excludes driver command lists
};
int DeviceCount() noexcept;
MemoryInfo GetMemoryInfo(int index = 0);
std::string DeviceDescription(int index = 0);
}  // namespace vt::xpu
