#include "xpu_common.h"
#include "xpu_kernels.h"

namespace vt::xpu {
void TraceXpuOp(OpId op, Queue& q, std::initializer_list<const Tensor*> tensors) {
  vt::TraceOpTensors(op, q, tensors);
  const auto mark = [&](size_t i) {
    if (i < tensors.size() && tensors.begin()[i]) {
      const auto& tensor = *tensors.begin()[i]; RecordGraphWrite(q, tensor.data, Span(tensor));
    }
  };
  switch (op) {
    case OpId::kReshapeAndCache: case OpId::kReshapeAndCacheFp8: mark(2); mark(3); break;
    case OpId::kGdnPostConv: for (size_t i = 0; i < 5; ++i) mark(i); break;
    case OpId::kAttnGateSplit: case OpId::kRopeNeox: case OpId::kRopeFromCache: mark(0); mark(1); break;
    case OpId::kGdnDecode: case OpId::kGdnPrefill: mark(0); mark(6); break;
    case OpId::kCausalConv1dFwd: case OpId::kCausalConv1dUpdate: mark(0); mark(4); break;
    case OpId::kRmsNorm: mark(0); mark(3); break;
    case OpId::kExl3Gemm: mark(0); mark(5); break;
    default: mark(0); break;
  }
}
void CopyKernel(Queue& q, Tensor& out, const Tensor& in) {
  TraceXpuOp(OpId::kCopy, q, {&out, &in});
  const auto n = out.Numel();
  if (!n) return;
  const size_t bytes = SizeOf(out.dtype);
  VT_CHECK(out.dtype == in.dtype ||
               ((in.dtype == DType::kF32 || in.dtype == DType::kF16 || in.dtype == DType::kBF16)
                && (out.dtype == DType::kF32 || out.dtype == DType::kF16 || out.dtype == DType::kBF16)),
           "XPU copy: unsupported conversion");
  WithOutput(q, out, {&in}, [&](Tensor& target) {
    const View dst(target), src(in);
    NativeQueue(q).parallel_for(sycl::range<1>(n), [=](sycl::id<1> item) {
      const auto i = item[0];
      const auto di = dst.offset(i), si = src.offset(i);
      if (dst.dtype == src.dtype) {
        for (size_t b = 0; b < bytes; ++b)
          static_cast<char*>(dst.data)[di * bytes + b] = static_cast<const char*>(src.data)[si * bytes + b];
      } else Store(dst, di, Load(src, si));
    });
  });
}
void AddKernel(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  TraceXpuOp(OpId::kAdd, q, {&out, &a, &b});
  FloatTensor(out); FloatTensor(a); FloatTensor(b);
  WithOutput(q, out, {&a, &b}, [&](Tensor& target) {
    const View dst(target), av(a), bv(b);
    const bool broadcast = b.rank == 1 && a.rank != 1;
    const auto width = a.shape[a.rank - 1];
    NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
      const auto i = item[0];
      Store(dst, dst.offset(i), Load(av, av.offset(i)) + Load(bv, bv.offset(broadcast ? i % width : i)));
    });
  });
}
void MoeSiluMulKernel(Queue& q, Tensor& out, const Tensor& gate, const Tensor& up) {
  TraceXpuOp(OpId::kMoeSiluMul, q, {&out, &gate, &up});
  FloatTensor(out); FloatTensor(gate); FloatTensor(up);
  WithOutput(q, out, {&gate, &up}, [&](Tensor& target) {
    const View dst(target), gv(gate), uv(up);
    NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
      const auto i = item[0]; const float g = Load(gv, gv.offset(i));
      const float act = Round(gv.dtype, g / (1.0f + sycl::exp(-g)));
      Store(dst, dst.offset(i), act * Load(uv, uv.offset(i)));
    });
  });
}
void SiluAndMulKernel(Queue& q, Tensor& out, const Tensor& in) {
  TraceXpuOp(OpId::kSiluAndMul, q, {&out, &in});
  FloatTensor(out); FloatTensor(in);
  WithOutput(q, out, {&in}, [&](Tensor& target) {
    const View dst(target), src(in);
    const auto width = out.shape[1];
    NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
      const auto i = item[0]; const auto off = (i / width) * src.stride[0] + i % width;
      const float gate = Load(src, off);
      Store(dst, dst.offset(i), Round(src.dtype, gate / (1.0f + sycl::exp(-gate))) * Load(src, off + width));
    });
  });
}
void SigmoidGateKernel(Queue& q, Tensor& out, const Tensor& attn, const Tensor& gate) {
  TraceXpuOp(OpId::kSigmoidGateBf16, q, {&out, &attn, &gate});
  FloatTensor(out); FloatTensor(attn); FloatTensor(gate);
  WithOutput(q, out, {&attn, &gate}, [&](Tensor& target) {
    const View dst(target), av(attn), gv(gate);
    NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
      const auto i = item[0];
      Store(dst, dst.offset(i), Load(av, av.offset(i)) * (1.0f / (1.0f + sycl::exp(-Load(gv, gv.offset(i))))));
    });
  });
}
namespace {
int64_t Index(View idx, int64_t row) {
  return idx.dtype == DType::kI64 ? static_cast<const int64_t*>(idx.data)[idx.offset(row)]
                                : static_cast<const int32_t*>(idx.data)[idx.offset(row)];
}
void CheckIndices(Queue& q, const Tensor& idx, int64_t limit) {
  if (!idx.Numel()) return;
  const View ids(idx); const auto rows = idx.Numel();
  CheckDeviceMetadata(q, [=] {
    for (int64_t r = 0; r < rows; ++r) if (Index(ids, r) < 0 || Index(ids, r) >= limit) return false;
    return true;
  }, "XPU index out of range", {&idx});
}
void Rows(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx, bool scatter, bool embedding) {
  TraceXpuOp(embedding ? OpId::kEmbedding : scatter ? OpId::kIndexCopy : OpId::kIndexSelect, q, {&out, &in, &idx});
  CheckIndices(q, idx, scatter ? out.shape[0] : in.shape[0]);
  if (idx.Numel() == 0) return;
  if (embedding) { FloatTensor(out); FloatTensor(in); }
  const int64_t rows = idx.Numel();
  const int64_t width = (scatter ? in.Numel() : out.Numel()) / rows;
  const size_t bytes = SizeOf(out.dtype);
  WithOutput(q, out, {&in, &idx}, [&](Tensor& target) {
    const View dst(target), src(in), ids(idx);
    NativeQueue(q).parallel_for(sycl::range<1>(rows * width), [=](sycl::id<1> item) {
      const auto row = item[0] / width, col = item[0] % width;
      const int64_t index = Index(ids, row);
      if (scatter) {
        // CPU index_copy scans in order: duplicate indices keep the last source.
        for (int64_t r = row + 1; r < rows; ++r) if (Index(ids, r) == index) return;
      }
      const int64_t di = (scatter ? index : row) * dst.stride[0] + col;
      const int64_t si = (scatter ? row : index) * src.stride[0] + col;
      if (embedding && dst.dtype != src.dtype) Store(dst, di, Load(src, si));
      else for (size_t b = 0; b < bytes; ++b)
        static_cast<char*>(dst.data)[di * bytes + b] = static_cast<const char*>(src.data)[si * bytes + b];
    });
  }, scatter);
}
void Matmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b, bool transpose) {
  TraceXpuOp(transpose ? OpId::kMatmulBT : OpId::kMatmul, q, {&out, &a, &b});
  FloatTensor(out); FloatTensor(a); FloatTensor(b);
  WithOutput(q, out, {&a, &b}, [&](Tensor& target) {
    const View dst(target), av(a), bv(b);
    const auto n = out.shape[1], k = a.shape[1];
    NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
      const auto row = item[0] / n, col = item[0] % n;
      float sum = 0;
      for (int64_t inner = 0; inner < k; ++inner) {
        const auto bi = transpose ? col * bv.stride[0] + inner * bv.stride[1]
                                  : inner * bv.stride[0] + col * bv.stride[1];
        sum += Load(av, row * av.stride[0] + inner * av.stride[1]) * Load(bv, bi);
      }
      Store(dst, row * dst.stride[0] + col * dst.stride[1], sum);
    });
  });
}
}
void IndexSelectKernel(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx) { Rows(q, out, in, idx, false, false); }
void IndexCopyKernel(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx) { Rows(q, out, in, idx, true, false); }
void EmbeddingKernel(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx) { Rows(q, out, in, idx, false, true); }
void MatmulKernel(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) { Matmul(q, out, a, b, false); }
void MatmulBTKernel(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) { Matmul(q, out, a, b, true); }
}  // namespace vt::xpu
