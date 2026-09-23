#pragma once
#include "vt/ops.h"
namespace vt::xpu {
void CopyKernel(Queue&, Tensor&, const Tensor&);
void AddKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void SiluAndMulKernel(Queue&, Tensor&, const Tensor&);
void MoeSiluMulKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void SigmoidGateKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void IndexSelectKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void IndexCopyKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void EmbeddingKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void MatmulKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void MatmulBTKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void RmsNormKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const RmsNormArgs&, Tensor*);
void GreedyArgmaxKernel(Queue&, Tensor&, const Tensor&);
}  // namespace vt::xpu
