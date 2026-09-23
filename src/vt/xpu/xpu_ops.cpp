#include "xpu_kernels.h"
namespace vt::xpu {
namespace {
struct Registrar {
  Registrar() {
#define XPU_OP(id, type, fn) RegisterOp(OpId::id, DeviceType::kXPU, reinterpret_cast<void*>(static_cast<type>(&fn)))
    XPU_OP(kCopy, CopyFn, CopyKernel);
    XPU_OP(kCastBf16, CastBf16Fn, CopyKernel);
    XPU_OP(kCastF16, CastF16Fn, CopyKernel);
    XPU_OP(kCastF32, CastF32Fn, CopyKernel);
    XPU_OP(kAdd, AddFn, AddKernel);
    XPU_OP(kSiluAndMul, SiluAndMulFn, SiluAndMulKernel);
    XPU_OP(kMoeSiluMul, MoeSiluMulFn, MoeSiluMulKernel);
    XPU_OP(kSigmoidGateBf16, SigmoidGateBf16Fn, SigmoidGateKernel);
    XPU_OP(kIndexSelect, IndexSelectFn, IndexSelectKernel);
    XPU_OP(kIndexCopy, IndexCopyFn, IndexCopyKernel);
    XPU_OP(kEmbedding, EmbeddingFn, EmbeddingKernel);
    XPU_OP(kMatmul, MatmulFn, MatmulKernel);
    XPU_OP(kMatmulBT, MatmulFn, MatmulBTKernel);
    XPU_OP(kRmsNorm, RmsNormFn, RmsNormKernel);
    XPU_OP(kGreedyArgmax, GreedyArgmaxFn, GreedyArgmaxKernel);
#undef XPU_OP
  }
};
[[maybe_unused]] Registrar registrar;
}
}  // namespace vt::xpu
