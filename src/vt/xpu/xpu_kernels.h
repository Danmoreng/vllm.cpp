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
void Exl3HadR128Kernel(Queue&, Tensor&, const Tensor&, const Exl3HadArgs&);
void Exl3GemmKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                    const Tensor&, Tensor&, const Exl3GemmArgs&);
bool Exl3PrefillKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, int, bool matrix = true);
void Exl3OutputHadPanel(Queue&, Tensor&, const Tensor&, const Tensor&, int64_t);
void CausalConv1dFwdKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*,
                           Tensor&, const Tensor&, const Tensor&, const CausalConv1dArgs&);
void CausalConv1dUpdateKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*,
                              Tensor&, const Tensor*, const CausalConv1dArgs&);
void GdnPostConvKernel(Queue&, Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, const Tensor&,
                        const Tensor&, const Tensor&, const Tensor&, const Tensor&, const L2NormArgs&);
void GdnPrefillKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                       const Tensor&, Tensor&, const Tensor&, const GdnArgs&);
void GdnDecodeKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                      const Tensor&, Tensor&, const Tensor*, const GdnArgs&);
void RmsNormGatedKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const RmsNormGatedArgs&);
void GdnStateGatherKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor*);
void GdnStateScatterKernel(Queue&, Tensor&, const Tensor&, const Tensor&);
void AttnGateSplitKernel(Queue&, Tensor&, Tensor&, const Tensor&);
void RopeNeoxKernel(Queue&, Tensor&, Tensor&, const Tensor&, const RopeArgs&);
void RopeCosSinCacheKernel(Queue&, Tensor&, const Tensor&, const RopeArgs&);
void RopeFromCacheKernel(Queue&, Tensor&, Tensor*, const Tensor&, const Tensor&, const RopeArgs&);
void ReshapeAndCacheKernel(Queue&, const Tensor&, const Tensor&, Tensor&, Tensor&, const Tensor&);
void PagedAttentionKernel(Queue&, Tensor&, const Tensor&, const Tensor&, const Tensor&,
                            const Tensor&, const Tensor&, const Tensor&, const PagedAttentionArgs&);
}  // namespace vt::xpu
