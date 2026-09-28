# Qwen XPU hybrid attention page parity

Python vLLM's XPU FlashAttention backend starts with 64-token alignment. The
hybrid KV unification in `vllm/platforms/interface.py` enlarges the attention
page until it can hold one Mamba state page. The matched Python FP8/MTP4 log
reports 1664 attention tokens per page. For the pinned 27B GPTQ model, the
Mamba page is 3,289,088 bytes and attention storage is 2,048 bytes per token:
`64 * ceil(3289088 / (64 * 2048)) = 1664`. Without MTP the derived page is
1600. C++ now derives this XPU Qwen page before sizing the pool and accepts
1664 in the Xe2 prefill and split-K kernels.

Focused checks: `test_hybrid_kv_budget` page derivation 6/6 assertions;
`test_xpu_attention_fast` selected 1600/1664 kernel comparisons 52/52;
real checkpoint, FP8 KV, MTP4 sampling, 4096-token retrieval prompt and 8
output tokens 43/43, including automatic 1600 -> 1664 load resolution and
zero reference-tier hits.

Matched C++ run: eager, one request, FP8 KV, MTP4, 4096 retrieval prompt,
1024 sampled output tokens, temperature 1, top-p .95, top-k 20, seed 42,
20 KV blocks, no prefix cache. TTFT uses a separate one-token request and
decode rate is `(1024-1)/(full-request time - TTFT)`.

| Engine/page | TTFT | Prefill token/s | Full 1024 output | Derived decode token/s |
|---|---:|---:|---:|---:|
| C++ / 1600 | 4.18559 s | 978.60 | 28.7966 s | 41.5667 |
| C++ / 1664 | 4.07580 s | 1004.96 | 28.6826 s | 41.5738 |
| Python / 1664 | 2.58152 s | 1586.66 | 16.82313 s | 71.8318 |

The 1664 run proposed 1012 draft tokens, accepted 770, and used 253 draft
steps. C++/Python parity with the same page is 63.3% for client prefill and
57.9% for derived decode. Sampling uses different RNG streams, so the token
sequences and acceptance histories are not expected to match exactly.
