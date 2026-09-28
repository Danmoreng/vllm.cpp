# B70 Xe2 shared-KV verification donor

This optional Torch-free Q2–Q5 FP8 verification policy derives from
`vllm-project/vllm-xpu-kernels` commit
`6d92b1bfbf32767ecda8e819613eb151e70030ad` plus the production
`shared-kv-verification.patch` in the pinned local Python deployment.
The root Apache-2.0 license is in `LICENSE`; copied collective and kernel
headers retain their BSD-3-Clause notices.

The source is built only with `VLLM_CPP_XPU_XE2_PREFILL=ON`, the pinned
SYCL-TLA checkout `87f6850680a580654b9ea2c80dbc01aeb36ad231` and
oneAPI 2026.1.1. It is off by default at runtime. Set
`VT_XPU_XE2_VERIFY=1` for the eligible automatic route or
`VT_XPU_ATTENTION=verify` for an explicit diagnostic; unsupported shapes
fall back to the existing attention implementation.

`paged_decode.hpp` is narrowed to the one required policy and no longer
contains the Torch dispatcher. `src/vt/xpu/xpu_attention_verify_xe2.cpp`
adapts VT's head-contiguous FP8 K/V pages and Q layout to the donor's
packed Q layout. No Python or Torch code is loaded for C++ inference.
