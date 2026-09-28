# W5: Python M04 Q5 attention replay on native C++ Split-K (2026-09-28)

The layer bisect in `w5_same_prefix_layer_bisect.md` showed growing target
differences at full-attention layers. This test isolates the attention operator
from upstream projections, RoPE and already different cache contents. A
temporary probe in the installed production `b70_attention.py` captured the
first real five-row M04 verification call after a 4096-token retrieval prompt.
The probe was excluded from the committed runtime and the temporary Python
server was stopped after one eight-token request.

The Python image was
`local/b70-qwen38-vllm:vllm-0.30.0-xpu-kernels-0.1.15.4`, with the pinned
GPTQ-G128 model, eager MTP4, T1/p.95/k20, FP8 E4M3 KV, no prefix reuse,
and M04 shared-KV operator SHA-256
`784916abd42b614794becac634a6a154f854c73d6dfcb1a50cd939b563020c6a`.
The request reported exactly 4096 prompt tokens. M04 dispatched with Q5,
KV length 4101, 1664-token physical pages, unit K/V scales, softmax scale
0.0625 and 16 splits. Its Q was F16 `[5,24,256]`; the three active physical
K/V pages were copied in logical block-table order as E4M3 bytes. The raw
capture remains outside Git in `/tmp/b70_m04_real_replay` (~10 MiB). SHA-256:

| File | SHA-256 |
| --- | --- |
| `metadata.json` | `055d3114d3812e54e700ef32c052b2387b37ecc61c8173921dd43bf9aff09e53` |
| `q.bin` | `2360959279560c8b8796cd30e8f0d0c47e5509f26d5c2c2468cb3db50a2b58e1` |
| `k.bin` | `e25846c9cb395a6e949c1990183ec47749ba478f38bbe686212777e30f3d606c` |
| `v.bin` | `c4df2f5c697816210a7ed863a88218bea77994acee3a0faea5c536afb3f9a2d8` |
| `output.bin` | `51685e65deab1be2d75b2fb9fc67e936288d375c6489cc48392377f22323b0c1` |

The focused native test in `test_xpu_attention_fast.cpp` repacks only the K/V
memory layout for the public VT cache contract, retaining each FP8 value and
logical token order. It feeds the captured Python Q and cache bytes to C++
`vt::PagedAttention` with the same causal positions and scale, and checks that
the `attention_split_partial` route was selected. The pinned oneAPI builder
compiled the target, and the B70 test passed **29/29 assertions** with no
reference-tier hits.

| Verification row | Relative RMS difference | Max absolute difference |
| ---: | ---: | ---: |
| 0 | 0.02965% | 0.000976562 |
| 1 | 0.03259% | 0.000976562 |
| 2 | 0.03169% | 0.000976562 |
| 3 | 0.03148% | 0.000976562 |
| 4 | 0.03014% | 0.000976562 |
| All rows | **0.03114%** | **0.000976562** |

This is a same-input operator comparison, not an end-to-end target-quality
pass or a performance claim. It rules out a large intrinsic arithmetic gap
between these two attention kernels at this Q5 input. It does not rule out
different Q/K/V generation, cache quantization, positions, or small errors
amplified by downstream layers. The next discriminating capture is the native
C++ Q and active FP8 KV from the same prompt and first verification layer,
compared directly with the Python operands before revisiting shared-KV tiling.
