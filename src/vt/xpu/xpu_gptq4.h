#pragma once

#include <cstddef>
#include <cstdint>

#include "vt/device.h"

namespace vt::xpu {

struct Gptq4RuntimeStats {
  size_t engine_count = 0;
  size_t primitive_count = 0;
  size_t scratchpad_allocation_count = 0;
  size_t scratchpad_capacity_bytes = 0;
};

Gptq4RuntimeStats GetGptq4RuntimeStats(int device_index);
void ReleaseGptq4Queue(const Queue& queue);

}  // namespace vt::xpu
