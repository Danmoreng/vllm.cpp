// Experimental benchmark only. This reduction does not satisfy the bit-exact
// EXL3 inference gate and is deliberately not registered as a VT operator.
#include "b70_exl3_xmx.h"
#include "vt/xpu/xpu_common.h"
#include "vt/xpu/xpu_exl3.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace vt::xpu {
namespace {
// Opt-in small-M XMX candidate. Stage one quantized tile at a time in SLM;
// only padded output activations use global scratch, never decoded weights.
template<int Bits, int TM, int TN, int TK>
void MatrixGemm(Queue& q, float* out, const sycl::half* in, const uint32_t* packed,
                 int64_t m, int64_t k, int64_t n) {
  namespace mx = sycl::ext::oneapi::experimental::matrix;
  const int64_t tiles_n = n / TN, groups = ((m + TM - 1) / TM) * tiles_n;
  NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<sycl::half> a(TM * TK, h), b(TK * TN, h);
    h.parallel_for(sycl::nd_range<1>(groups * 128, 128), [=](sycl::nd_item<1> item)
                     [[sycl::reqd_sub_group_size(16)]] {
      auto sg = item.get_sub_group();
      const int lane = item.get_local_id(0);
      const int64_t row = (item.get_group(0) / tiles_n) * TM;
      const int64_t column = (item.get_group(0) % tiles_n) * TN;
      mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major> ja;
      mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::row_major> jb;
      mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> jc;
      if (lane < 16) mx::joint_matrix_fill(sg, jc, 0.0f);
      for (int64_t base = 0; base < k; base += TK) {
        for (int i = lane; i < TM * TK; i += 128)
          a[i] = row + i / TK < m ? in[(row + i / TK) * k + base + i % TK] : sycl::half(0);
        #pragma unroll
        for (int i = lane; i < TK * TN; i += 128) {
          const int r = i / TN, c = i % TN;
          const auto* tile = packed + (((base + r) / 16) * (n / 16) + (column + c) / 16) * (8 * Bits);
          b[i] = sycl::half(exl3::Decode(exl3::PackedCodeword<Bits>(tile, r % 16, c % 16), 2));
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (lane < 16) {
          mx::joint_matrix_load(sg, ja, a.template get_multi_ptr<sycl::access::decorated::no>(), TK);
          mx::joint_matrix_load(sg, jb, b.template get_multi_ptr<sycl::access::decorated::no>(), TN);
          mx::joint_matrix_mad(sg, jc, ja, jb, jc);
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      if (lane < 16) mx::joint_matrix_store(sg, jc,
          sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
              out + row * n + column), n, mx::layout::row_major);
    });
  });
}

}
void Exl3MatrixProbe(Queue& q, Tensor& out, const Tensor& in, const Tensor& trellis,
    const Tensor& suh, const Tensor& svh, Tensor& in_had, int bits) {
  const int64_t m = in.shape[0], k = in.shape[1], n = out.shape[1];
  VT_CHECK(bits >= 3 && bits <= 6 && m > 0 && m <= 128 && k % 128 == 0 && n % 128 == 0,
           "Unsupported XMX probe shape");
  VT_CHECK(in.dtype == DType::kF16 && out.dtype == DType::kF32 &&
           reinterpret_cast<uintptr_t>(trellis.data) % alignof(uint32_t) == 0,
           "Unsupported XMX probe storage");
  Exl3HadR128(q, in_had, in, Exl3HadArgs{&suh, nullptr, 1.0f});
  const int64_t padded_m = m == 1 ? 1 : ((m + 7) / 8) * 8;
  Scratch scratch(q.device, padded_m * n * sizeof(float));
  auto raw = Tensor::Contiguous(scratch.data, DType::kF32, q.device, {m, n});
  const auto* words = static_cast<const uint32_t*>(trellis.data);
  const auto* ah = static_cast<const sycl::half*>(in_had.data);
  auto launch = [&]<int Bits>() {
    if (m == 1) MatrixGemm<Bits, 1, 64, 32>(q, static_cast<float*>(raw.data), ah, words, m, k, n);
    else MatrixGemm<Bits, 8, 16, 16>(q, static_cast<float*>(raw.data), ah, words, m, k, n);
  };
  switch (bits) {
    case 3: launch.template operator()<3>(); break;
    case 4: launch.template operator()<4>(); break;
    case 5: launch.template operator()<5>(); break;
    case 6: launch.template operator()<6>(); break;
  }
  Exl3HadR128(q, out, raw, Exl3HadArgs{nullptr, &svh, 1.0f});
  GetBackend(q.device).Synchronize(q);
}
}  // namespace vt::xpu
