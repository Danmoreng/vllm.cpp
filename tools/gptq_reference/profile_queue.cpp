// Small SYCL queue helper for timing the Python reference on the same kind of
// profiling-enabled, in-order queue used by the native operation probe.
#include <memory>
#include <sycl/sycl.hpp>

extern "C" void* gptq4_create_profiled_queue(void* original_queue) {
  const auto* original = static_cast<sycl::queue*>(original_queue);
  return new sycl::queue(
      original->get_context(), original->get_device(),
      sycl::property_list{sycl::property::queue::enable_profiling{},
                          sycl::property::queue::in_order{}});
}

extern "C" void gptq4_destroy_profiled_queue(void* queue) {
  delete static_cast<sycl::queue*>(queue);
}

extern "C" void* gptq4_submit_timer_marker(void* queue) {
  auto* sycl_queue = static_cast<sycl::queue*>(queue);
  return new sycl::event(sycl_queue->single_task([] {}));
}

extern "C" double gptq4_timer_interval_us(void* start, void* end) {
  std::unique_ptr<sycl::event> start_event(static_cast<sycl::event*>(start));
  std::unique_ptr<sycl::event> end_event(static_cast<sycl::event*>(end));
  end_event->wait_and_throw();
  const auto start_ns = start_event->get_profiling_info<
      sycl::info::event_profiling::command_start>();
  const auto end_ns = end_event->get_profiling_info<
      sycl::info::event_profiling::command_end>();
  return static_cast<double>(end_ns - start_ns) / 1000.0;
}
