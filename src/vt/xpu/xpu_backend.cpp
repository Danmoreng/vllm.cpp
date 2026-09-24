// Initial correctness backend: in-order queues, device USM, conservative frees.
#include "xpu_common.h"
#include "vt/xpu.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

namespace vt::xpu {
namespace {
struct Devices {
  std::vector<sycl::device> gpu;
  std::string error;
  Devices() noexcept {
    try {
      for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        // OpenCL exposes the same card again. One Level Zero device = one context.
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) gpu.push_back(d);
      }
    } catch (const std::exception& e) { error = e.what(); }
  }
};
Devices& AllDevices() { static Devices d; return d; }
const sycl::device& DeviceAt(int index) {
  const auto& all = AllDevices();
  VT_CHECK(index >= 0 && static_cast<size_t>(index) < all.gpu.size(),
           "XPU: no Level Zero GPU at index " + std::to_string(index) +
           (all.error.empty() ? " (CPU fallback is disabled)" : ": " + all.error));
  return all.gpu[index];
}
size_t Budget(size_t total) {
  const char* env = std::getenv("VT_XPU_MEMORY_BUDGET_BYTES");
  if (env == nullptr) return total;
  char* end = nullptr;
  errno = 0;
  const auto bytes = std::strtoull(env, &end, 10);
  VT_CHECK(*env >= '0' && *env <= '9' && end != env && *end == '\0' && errno == 0
               && bytes > 0 && bytes <= total,
           "VT_XPU_MEMORY_BUDGET_BYTES must be positive and no larger than device memory");
  return static_cast<size_t>(bytes);
}
struct Context {
  sycl::device device;
  sycl::context context;
  std::mutex mutex;
  std::unordered_map<sycl::queue*, std::unique_ptr<sycl::queue>> queues;
  std::unordered_map<void*, size_t> allocations, pinned;
  size_t total, budget, allocated = 0, pinned_bytes = 0;
  explicit Context(int index) : device(DeviceAt(index)), context(device),
      total(device.get_info<sycl::info::device::global_mem_size>()), budget(Budget(total)) {
    VT_CHECK(device.has(sycl::aspect::usm_device_allocations)
                 && device.has(sycl::aspect::usm_host_allocations),
             "XPU requires device and host USM allocations");
  }
  // Callers hold mutex. Waiting every owned queue also covers kernel submissions,
  // which do not carry allocation lifetime metadata in VT's current ABI.
  void Drain() { for (auto& [_, q] : queues) q->wait_and_throw(); }
  ~Context() {
    try { Drain(); } catch (const std::exception& e) {
      std::fprintf(stderr, "[vt xpu] shutdown wait failed: %s\n", e.what());
    }
    for (const auto& [p, _] : allocations) sycl::free(p, context);
    for (const auto& [p, _] : pinned) sycl::free(p, context);
  }
};
Context& GetContext(int index) {
  // Stable per-device storage; construction is lazy and failures reach the caller.
  static std::mutex mutex;
  static std::unique_ptr<Context> contexts[kMaxDevicesPerType];
  VT_CHECK(index >= 0 && static_cast<size_t>(index) < kMaxDevicesPerType, "XPU device index out of range");
  std::lock_guard<std::mutex> lock(mutex);
  if (!contexts[index]) contexts[index] = std::make_unique<Context>(index);
  return *contexts[index];
}
struct EventState {
  int index;
  sycl::event event;
  bool recorded = false;
};
EventState& CheckEvent(Event& e, int index) {
  VT_CHECK(e.device == (Device{DeviceType::kXPU, index}) && e.handle != nullptr,
           "XPU event owner mismatch or destroyed event");
  auto& state = *static_cast<EventState*>(e.handle);
  VT_CHECK(state.index == index, "XPU event context mismatch");
  return state;
}
class XpuBackend final : public Backend {
  int index_;
  Context& ctx() const { return GetContext(index_); }
  sycl::queue& queue(Queue& q) const {
    VT_CHECK(q.device == (Device{DeviceType::kXPU, index_}), "XPU queue owner mismatch");
    return NativeQueue(q);
  }
 public:
  explicit XpuBackend(int index) : index_(index) {}
  void* Alloc(size_t bytes) override {
    auto& c = ctx();
    bytes = std::max(bytes, size_t{1});
    std::lock_guard<std::mutex> lock(c.mutex);
    VT_CHECK(bytes <= c.budget - c.allocated, "XPU device allocation exceeds memory budget");
    void* p = sycl::aligned_alloc_device(64, bytes, c.device, c.context);
    VT_CHECK(p != nullptr, "XPU device USM allocation failed");
    try { c.allocations.emplace(p, bytes); } catch (...) { sycl::free(p, c.context); throw; }
    c.allocated += bytes;
    return p;
  }
  void Free(void* p) override {
    if (p == nullptr) return;
    auto& c = ctx();
    std::lock_guard<std::mutex> lock(c.mutex);
    const auto it = c.allocations.find(p);
    VT_CHECK(it != c.allocations.end(), "XPU free: pointer not owned by this device");
    c.Drain();
    sycl::free(p, c.context);
    c.allocated -= it->second;
    c.allocations.erase(it);
  }
  void* AllocPinned(size_t bytes) override {
    auto& c = ctx();
    bytes = std::max(bytes, size_t{1});
    std::lock_guard<std::mutex> lock(c.mutex);
    void* p = sycl::aligned_alloc_host(64, bytes, c.context);
    VT_CHECK(p != nullptr, "XPU pinned host USM allocation failed");
    try { c.pinned.emplace(p, bytes); } catch (...) { sycl::free(p, c.context); throw; }
    c.pinned_bytes += bytes;
    return p;
  }
  void FreePinned(void* p) override {
    if (p == nullptr) return;
    auto& c = ctx();
    std::lock_guard<std::mutex> lock(c.mutex);
    const auto it = c.pinned.find(p);
    VT_CHECK(it != c.pinned.end(), "XPU pinned free: pointer not owned by this device");
    c.Drain();
    sycl::free(p, c.context);
    c.pinned_bytes -= it->second;
    c.pinned.erase(it);
  }
  void Memset(Queue& q, void* p, int value, size_t bytes) override {
    auto& native = queue(q);
    if (bytes) native.memset(p, value, bytes);
  }
  void Copy(Queue& q, void* dst, const void* src, size_t bytes) override {
    auto& native = queue(q);
    if (!bytes) return;
    const auto context = native.get_context();
    const bool host_src = sycl::get_pointer_type(src, context) == sycl::usm::alloc::unknown;
    const bool host_dst = sycl::get_pointer_type(dst, context) == sycl::usm::alloc::unknown;
    if (!host_src && !host_dst) { native.memcpy(dst, src, bytes); return; }
    if (host_src && host_dst) {
      native.wait_and_throw();
      std::memcpy(dst, src, bytes);
      return;
    }
    // Ordinary host pointers include read-only, unaligned safetensors mmaps.
    // Level Zero's direct import of those mappings can fault in the copy engine.
    // Only known USM pointers reach DMA; bound staging independently of weights.
    const size_t chunk = std::min(bytes, size_t{4 * 1024 * 1024});
    void* staging = AllocPinned(chunk);
    try {
      for (size_t offset = 0; offset < bytes; offset += chunk) {
        const size_t count = std::min(chunk, bytes - offset);
        if (host_src) {
          std::memcpy(staging, static_cast<const char*>(src) + offset, count);
          native.memcpy(static_cast<char*>(dst) + offset, staging, count).wait_and_throw();
        } else {
          native.memcpy(staging, static_cast<const char*>(src) + offset, count).wait_and_throw();
          std::memcpy(static_cast<char*>(dst) + offset, staging, count);
        }
      }
    } catch (...) {
      try { FreePinned(staging); } catch (...) { /* retained until context shutdown */ }
      throw;
    }
    FreePinned(staging);
  }
  Queue CreateQueue() override {
    auto& c = ctx();
    auto q = std::make_unique<sycl::queue>(c.context, c.device,
        [](sycl::exception_list errors) { for (auto e : errors) std::rethrow_exception(e); },
        sycl::property_list{sycl::property::queue::in_order{}});
    auto* handle = q.get();
    std::lock_guard<std::mutex> lock(c.mutex);
    c.queues.emplace(handle, std::move(q));
    return Queue{Device{DeviceType::kXPU, index_}, handle};
  }
  void DestroyQueue(Queue& q) override {
    if (q.handle == nullptr) return;
    queue(q).wait_and_throw();
    auto& c = ctx();
    std::lock_guard<std::mutex> lock(c.mutex);
    c.queues.erase(static_cast<sycl::queue*>(q.handle));
    q.handle = nullptr;
  }
  void Synchronize(Queue& q) override { queue(q).wait_and_throw(); }
  bool UnifiedMemory() const override { return false; }
  bool DeviceMemoryIsHostAddressable() const override { return false; }
  bool DeviceMemoryInfo(size_t* free, size_t* total) const override {
    const auto info = GetMemoryInfo(index_);
    if (!info.free_known) return false;
    if (free) *free = std::min(info.free_bytes, info.budget_bytes - info.allocated_bytes);
    if (total) *total = info.total_bytes;
    return true;
  }
  Event CreateEvent(bool = false) override {
    (void)ctx();
    return Event{Device{DeviceType::kXPU, index_}, new EventState{index_, {}, false}};
  }
  void DestroyEvent(Event& e) override {
    if (!e.handle) return;
    auto& state = CheckEvent(e, index_);
    if (state.recorded) state.event.wait_and_throw();
    delete &state;
    e.handle = nullptr;
  }
  void RecordEvent(Event& e, Queue& q) override {
    auto& state = CheckEvent(e, index_);
    state.event = queue(q).ext_oneapi_submit_barrier();
    state.recorded = true;
  }
  void SynchronizeEvent(Event& e) override {
    auto& state = CheckEvent(e, index_);
    if (state.recorded) state.event.wait_and_throw();
  }
  bool QueryEvent(Event& e) override {
    auto& state = CheckEvent(e, index_);
    return !state.recorded || state.event.get_info<sycl::info::event::command_execution_status>()
                                 == sycl::info::event_command_status::complete;
  }
  void QueueWaitEvent(Queue& q, Event& e) override {
    auto& state = CheckEvent(e, index_);
    if (state.recorded) queue(q).ext_oneapi_submit_barrier({state.event});
  }
};
struct Registrar {
  Registrar() {
    // Register a refusing index-0 backend even without hardware: explicit XPU
    // resource requests get a diagnostic and can never resolve to a CPU queue.
    static std::vector<std::unique_ptr<XpuBackend>> backends;
    const int count = std::max(1, std::min(DeviceCount(), static_cast<int>(kMaxDevicesPerType)));
    for (int i = 0; i < count; ++i) {
      backends.push_back(std::make_unique<XpuBackend>(i));
      RegisterBackend(Device{DeviceType::kXPU, i}, backends.back().get());
    }
  }
};
[[maybe_unused]] Registrar registrar;
}  // namespace

int DeviceCount() noexcept { return static_cast<int>(AllDevices().gpu.size()); }
sycl::queue& NativeQueue(Queue& q) {
  VT_CHECK(q.device.type == DeviceType::kXPU && q.handle != nullptr, "XPU requires a live native queue");
  auto& c = GetContext(q.device.index);
  std::lock_guard<std::mutex> lock(c.mutex);
  auto it = c.queues.find(static_cast<sycl::queue*>(q.handle));
  VT_CHECK(it != c.queues.end(), "XPU queue does not belong to this context");
  return *it->second;
}
MemoryInfo GetMemoryInfo(int index) {
  auto& c = GetContext(index);
  std::lock_guard<std::mutex> lock(c.mutex);
  MemoryInfo info{c.total, c.budget, c.allocated, c.pinned_bytes, 0, false};
  if (c.device.has(sycl::aspect::ext_intel_free_memory)) {
    info.free_bytes = c.device.get_info<sycl::ext::intel::info::device::free_memory>();
    info.free_known = true;
  }
  return info;
}
std::string DeviceDescription(int index) {
  const auto& d = DeviceAt(index);
  nlohmann::json info = {
      {"index", index}, {"name", d.get_info<sycl::info::device::name>()},
      {"driver", d.get_info<sycl::info::device::driver_version>()},
      {"runtime", d.get_platform().get_info<sycl::info::platform::version>()},
      {"compiler", __VERSION__}, {"sycl_language", SYCL_LANGUAGE_VERSION},
      {"subgroups", d.get_info<sycl::info::device::sub_group_sizes>()},
      {"total_bytes", d.get_info<sycl::info::device::global_mem_size>()}};
  if (d.has(sycl::aspect::ext_intel_device_id))
    info["device_id"] = d.get_info<sycl::ext::intel::info::device::device_id>();
  info["matrix_combinations"] = nlohmann::json::array();
  if (d.has(sycl::aspect::ext_intel_matrix)) {
    for (const auto& m : d.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>())
      info["matrix_combinations"].push_back({{"m", m.msize}, {"n", m.nsize}, {"k", m.ksize},
          {"max_m", m.max_msize}, {"max_n", m.max_nsize}, {"max_k", m.max_ksize},
          {"a_type", static_cast<int>(m.atype)}, {"b_type", static_cast<int>(m.btype)},
          {"c_type", static_cast<int>(m.ctype)}, {"d_type", static_cast<int>(m.dtype)}});
  }
  return info.dump();
}
}  // namespace vt::xpu
