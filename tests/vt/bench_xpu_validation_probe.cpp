// Isolated P3 probe: one checked readback per repeated consumer versus one
// checked readback for the same immutable, host-provided metadata. This is not
// a model throughput benchmark or a substitute for a validated descriptor.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

bool Check(sycl::queue& q, const int32_t* slots, int32_t bound) {
  auto* status = sycl::malloc_device<int32_t>(1, q);
  auto* host = sycl::malloc_host<int32_t>(1, q);
  if (!status || !host) throw std::bad_alloc();
  q.single_task([=] { status[0] = slots[0] >= 0 && slots[0] < bound; });
  q.memcpy(host, status, sizeof(int32_t)).wait_and_throw();
  const bool valid = host[0] != 0;
  sycl::free(status, q);
  sycl::free(host, q);
  return valid;
}

void Work(sycl::queue& q, int32_t* output, int iteration) {
  q.single_task([=] { output[iteration] = iteration + 1; });
}

double Run(sycl::queue& q, const int32_t* slots, int32_t* output,
           int32_t bound, int repeats, bool bundled) {
  const auto start = Clock::now();
  if (bundled) {
    if (!Check(q, slots, bound)) throw std::runtime_error("invalid slot");
    for (int i = 0; i < repeats; ++i) Work(q, output, i);
  } else {
    for (int i = 0; i < repeats; ++i) {
      if (!Check(q, slots, bound)) throw std::runtime_error("invalid slot");
      Work(q, output, i);
    }
  }
  q.wait_and_throw();
  const auto end = Clock::now();
  int32_t last = 0;
  q.memcpy(&last, output + repeats - 1, sizeof(last)).wait_and_throw();
  if (last != repeats) throw std::runtime_error("work output mismatch");
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool RejectsInvalid(sycl::queue& q, const int32_t* slots, int32_t* output,
                    int32_t bound, int repeats, bool bundled) {
  try {
    Run(q, slots, output, bound, repeats, bundled);
    return false;
  } catch (const std::runtime_error&) {
    return true;
  }
}
}  // namespace

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    auto* slots = sycl::malloc_device<int32_t>(1, q);
    auto* output = sycl::malloc_device<int32_t>(192, q);
    if (!slots || !output) throw std::bad_alloc();
    const int32_t valid = 0, invalid = 1;
    for (int repeats : {96, 192}) {
      q.memcpy(slots, &valid, sizeof(valid)).wait_and_throw();
      for (int warmup = 0; warmup < 2; ++warmup) {
        Run(q, slots, output, 1, repeats, false);
        Run(q, slots, output, 1, repeats, true);
      }
      std::vector<double> separate, bundled;
      for (int session = 0; session < 7; ++session) {
        if (session % 2) {
          bundled.push_back(Run(q, slots, output, 1, repeats, true));
          separate.push_back(Run(q, slots, output, 1, repeats, false));
        } else {
          separate.push_back(Run(q, slots, output, 1, repeats, false));
          bundled.push_back(Run(q, slots, output, 1, repeats, true));
        }
      }
      q.memcpy(slots, &invalid, sizeof(invalid)).wait_and_throw();
      const bool separate_rejects = RejectsInvalid(q, slots, output, 1, repeats, false);
      const bool bundled_rejects = RejectsInvalid(q, slots, output, 1, repeats, true);
      if (!separate_rejects || !bundled_rejects)
        throw std::runtime_error("invalid metadata was accepted");
      std::cout << "{\"repeated_checks\":" << repeats << ",\"separate_ms\":[";
      for (size_t i = 0; i < separate.size(); ++i)
        std::cout << (i ? "," : "") << separate[i];
      std::cout << "],\"bundled_ms\":[";
      for (size_t i = 0; i < bundled.size(); ++i)
        std::cout << (i ? "," : "") << bundled[i];
      std::cout << "],\"invalid_rejected\":true}" << std::endl;
    }
    sycl::free(slots, q);
    sycl::free(output, q);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
