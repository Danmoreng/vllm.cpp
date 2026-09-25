// Standalone correctness probe for the pinned Python GPTQ W4A16 fixture.
// This creates its own in-order SYCL queue; production integration passes
// vt::xpu::NativeQueue(q) to the same oneDNN operation after this gate passes.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

namespace {
using Json = nlohmann::json;

struct TimerStartKernel {
  void operator()() const {}
};
struct TimerEndKernel {
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
    const std::vector<uint8_t> file((std::istreambuf_iterator<char>(input)), {});
    if (file.size() < sizeof(uint64_t)) throw std::runtime_error("short safetensors file");

    uint64_t header_bytes = 0;
    for (int i = 0; i < 8; ++i) header_bytes |= uint64_t{file[i]} << (8 * i);
    if (header_bytes > file.size() - 8) throw std::runtime_error("invalid safetensors header size");
    const auto header_begin = reinterpret_cast<const char*>(file.data() + 8);
    const Json header = Json::parse(header_begin, header_begin + header_bytes);
    data_start_ = 8 + static_cast<size_t>(header_bytes);
    file_ = file;
    header_ = header;
  }

  FixtureTensor Get(const std::string& name) const {
    if (!header_.contains(name)) throw std::runtime_error("missing fixture tensor: " + name);
    const Json& entry = header_.at(name);
    FixtureTensor tensor;
    tensor.dtype = entry.at("dtype").get<std::string>();
    tensor.shape = entry.at("shape").get<std::vector<int64_t>>();
    const auto offsets = entry.at("data_offsets").get<std::array<uint64_t, 2>>();
    if (offsets[0] > offsets[1] || offsets[1] > file_.size() - data_start_)
      throw std::runtime_error("invalid data offsets for fixture tensor: " + name);
    const size_t first = data_start_ + static_cast<size_t>(offsets[0]);
    const size_t last = data_start_ + static_cast<size_t>(offsets[1]);
    tensor.bytes.assign(file_.begin() + first, file_.begin() + last);
    return tensor;
  }

 private:
  size_t data_start_ = 0;
  std::vector<uint8_t> file_;
  Json header_;
};

class DeviceBuffer {
 public:
  DeviceBuffer(sycl::queue& queue, size_t bytes)
      : context_(queue.get_context()), bytes_(bytes), pointer_(sycl::malloc_device(bytes, queue)) {
    if (bytes_ != 0 && pointer_ == nullptr) throw std::bad_alloc();
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) sycl::free(pointer_, context_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void* data() const { return pointer_; }
  size_t size() const { return bytes_; }
  void Upload(sycl::queue& queue, const std::vector<uint8_t>& bytes) {
    if (bytes.size() != bytes_) throw std::runtime_error("fixture tensor byte size mismatch");
    queue.memcpy(pointer_, bytes.data(), bytes.size());
  }

 private:
  sycl::context context_;
  size_t bytes_;
  void* pointer_;
};

void RequireTensor(const FixtureTensor& tensor, const char* name, const char* dtype,
                   std::initializer_list<int64_t> shape, size_t element_bytes) {
  if (tensor.dtype != dtype || tensor.shape != std::vector<int64_t>(shape))
    throw std::runtime_error(std::string("unexpected dtype/shape for ") + name);
  size_t elements = 1;
  for (const auto dimension : tensor.shape) elements *= static_cast<size_t>(dimension);
  if (tensor.bytes.size() != elements * element_bytes)
    throw std::runtime_error(std::string("unexpected byte count for ") + name);
}

float HalfToFloat(uint16_t bits) {
  sycl::half value;
  std::memcpy(&value, &bits, sizeof(bits));
  return static_cast<float>(value);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: b70_gptq4_onednn_probe FIXTURE.safetensors\n";
      return 2;
    }
    const Fixture fixture(argv[1]);
    const auto activation = fixture.Get("activation_fp16");
    const auto qweight = fixture.Get("qweight_nt_int32");
    const auto scales = fixture.Get("scales_f16");
    const auto zero_point = fixture.Get("effective_zero_point_i8");
    const auto expected = fixture.Get("output_fp16");
    RequireTensor(activation, "activation_fp16", "F16", {1, 5120}, 2);
    RequireTensor(qweight, "qweight_nt_int32", "I32", {1024, 640}, 4);
    RequireTensor(scales, "scales_f16", "F16", {40, 1024}, 2);
    RequireTensor(zero_point, "effective_zero_point_i8", "I8", {1}, 1);
    RequireTensor(expected, "output_fp16", "F16", {1, 1024}, 2);
    if (zero_point.bytes[0] != 8) throw std::runtime_error("fixture zero point is not 8");

    constexpr int64_t m = 1, k = 5120, n = 1024, group = 128;
    sycl::queue queue(sycl::gpu_selector_v,
                      sycl::property_list{sycl::property::queue::in_order{},
                                          sycl::property::queue::enable_profiling{}});
    const auto engine = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto stream = dnnl::sycl_interop::make_stream(engine, queue);

    const auto src_desc = dnnl::memory::desc({m, k}, dnnl::memory::data_type::f16, {k, 1});
    const auto weights_desc = dnnl::memory::desc({k, n}, dnnl::memory::data_type::u4, {1, k});
    const auto dst_desc = dnnl::memory::desc({m, n}, dnnl::memory::data_type::f16, {n, 1});
    if (weights_desc.get_size() != qweight.bytes.size())
      throw std::runtime_error("oneDNN U4 descriptor byte span disagrees with packed fixture");

    dnnl::primitive_attr attr;
    attr.set_scales(DNNL_ARG_WEIGHTS, 3, {group, 1}, dnnl::memory::data_type::f16);
    attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, dnnl::memory::data_type::s8);
    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    const dnnl::matmul::primitive_desc pd(engine, src_desc, weights_desc, dst_desc, attr);
    const dnnl::matmul primitive(pd);

    const auto scale_desc =
        dnnl::memory::desc({n, k / group}, dnnl::memory::data_type::f16, {1, n});
    const auto zero_point_desc = dnnl::memory::desc({1}, dnnl::memory::data_type::s8, {1});
    DeviceBuffer device_a(queue, activation.bytes.size());
    DeviceBuffer device_b(queue, qweight.bytes.size());
    DeviceBuffer device_scales(queue, scales.bytes.size());
    DeviceBuffer device_zero(queue, zero_point.bytes.size());
    DeviceBuffer device_c(queue, expected.bytes.size());
    device_a.Upload(queue, activation.bytes);
    device_b.Upload(queue, qweight.bytes);
    device_scales.Upload(queue, scales.bytes);
    device_zero.Upload(queue, zero_point.bytes);
    queue.wait_and_throw();

    std::vector<uint8_t> actual(expected.bytes.size());
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, dnnl::memory(src_desc, engine, device_a.data())},
        {DNNL_ARG_WEIGHTS, dnnl::memory(weights_desc, engine, device_b.data())},
        {DNNL_ARG_DST, dnnl::memory(dst_desc, engine, device_c.data())},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
         dnnl::memory(scale_desc, engine, device_scales.data())},
        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
         dnnl::memory(zero_point_desc, engine, device_zero.data())},
    };
    if (pd.scratchpad_desc().get_size() != 0)
      throw std::runtime_error("fixture descriptor unexpectedly requires a scratchpad");
    primitive.execute(stream, args);
    stream.wait();  // primitive/JIT warm-up, excluded from timed samples
    primitive.execute(stream, args);
    stream.wait();
    queue.memcpy(actual.data(), device_c.data(), actual.size());
    queue.wait_and_throw();

    std::array<double, 3> host_enqueue_ms{};
    std::array<double, 3> gpu_interval_ms{};
    for (size_t i = 0; i < host_enqueue_ms.size(); ++i) {
      auto start = queue.submit(
          [](sycl::handler& handler) { handler.single_task<TimerStartKernel>([] {}); });
      const auto host_start = std::chrono::steady_clock::now();
      primitive.execute(stream, args);
      const auto host_end = std::chrono::steady_clock::now();
      auto end =
          queue.submit([](sycl::handler& handler) { handler.single_task<TimerEndKernel>([] {}); });
      end.wait_and_throw();
      host_enqueue_ms[i] = std::chrono::duration<double, std::milli>(host_end - host_start).count();
      const auto device_start =
          start.get_profiling_info<sycl::info::event_profiling::command_start>();
      const auto device_end = end.get_profiling_info<sycl::info::event_profiling::command_end>();
      gpu_interval_ms[i] = static_cast<double>(device_end - device_start) * 1e-6;
    }

    double squared_error = 0;
    double max_error = 0;
    double max_expected = 0;
    size_t outside_screen = 0;
    for (size_t i = 0; i < actual.size() / 2; ++i) {
      uint16_t got_bits = 0, expected_bits = 0;
      std::memcpy(&got_bits, actual.data() + 2 * i, 2);
      std::memcpy(&expected_bits, expected.bytes.data() + 2 * i, 2);
      const double got = HalfToFloat(got_bits);
      const double ref = HalfToFloat(expected_bits);
      const double error = std::abs(got - ref);
      max_error = std::max(max_error, error);
      max_expected = std::max(max_expected, std::abs(ref));
      squared_error += error * error;
      if (!std::isfinite(got) || !std::isfinite(ref) || error > 0.02 + 0.01 * std::abs(ref))
        ++outside_screen;
    }
    const auto* version = dnnl_version();
    std::cout << std::setprecision(10)
              << "device=" << queue.get_device().get_info<sycl::info::device::name>() << '\n'
              << "onednn=" << version->major << '.' << version->minor << '.' << version->patch
              << " hash=" << version->hash << '\n'
              << "impl=" << pd.impl_info_str() << '\n'
              << "shape=M" << m << " K" << k << " N" << n << '\n'
              << "scratchpad_bytes=" << pd.scratchpad_desc().get_size() << '\n'
              << "same_queue=" << (dnnl::sycl_interop::get_queue(stream) == queue) << '\n'
              << "host_enqueue_ms=" << host_enqueue_ms[0] << ',' << host_enqueue_ms[1] << ','
              << host_enqueue_ms[2] << '\n'
              << "gpu_interval_ms=" << gpu_interval_ms[0] << ',' << gpu_interval_ms[1] << ','
              << gpu_interval_ms[2] << '\n'
              << "max_abs_error=" << max_error << '\n'
              << "rms_error=" << std::sqrt(squared_error / (actual.size() / 2)) << '\n'
              << "max_expected_abs=" << max_expected << '\n'
              << "values_outside_rtol_0.01_atol_0.02=" << outside_screen << '\n';
    return outside_screen == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
