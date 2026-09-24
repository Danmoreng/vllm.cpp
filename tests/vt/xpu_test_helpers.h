#pragma once
#include <doctest/doctest.h>
#include "vt/backend.h"
#include "vt/ops.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace xpu_test {
struct Queue {
  vt::Queue q;
  explicit Queue(vt::DeviceType type) : q(vt::CreateQueue({type, 0})) {}
  ~Queue() { vt::DestroyQueue(q); }
};
struct Buffer {
  vt::Queue& q;
  vt::Tensor tensor;
  size_t bytes;
  Buffer(vt::Queue& queue, vt::DType type, std::initializer_list<int64_t> shape) : q(queue) {
    tensor = vt::Tensor::Contiguous(nullptr, type, q.device, shape);
    bytes = tensor.Bytes();
    tensor.data = vt::Alloc(q.device, std::max(bytes, size_t{1}));
  }
  ~Buffer() { vt::Free(q.device, tensor.data); }
  Buffer(const Buffer&) = delete;
  void upload(const void* source, bool odd_address = false) {
    std::vector<unsigned char> unaligned;
    if (odd_address) {
      unaligned.resize(bytes + 1);
      std::memcpy(unaligned.data() + 1, source, bytes);
      source = unaligned.data() + 1;
    }
    auto& backend = vt::GetBackend(q.device);
    backend.Copy(q, tensor.data, source, bytes);
    backend.Synchronize(q);
  }
  std::vector<unsigned char> download() const {
    std::vector<unsigned char> result(bytes);
    auto& backend = vt::GetBackend(q.device);
    backend.Copy(q, result.data(), tensor.data, bytes);
    backend.Synchronize(q);
    return result;
  }
};
inline void SameBytes(const std::vector<unsigned char>& actual,
                      const std::vector<unsigned char>& expected) {
  REQUIRE(actual.size() == expected.size());
  const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
  const auto offset = mismatch.first - actual.begin();
  CAPTURE(offset);
  REQUIRE(mismatch.first == actual.end());
}
}  // namespace xpu_test
