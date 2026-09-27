# Observed 4096-token call paths

Evidence is in `route_evidence.json` and `raw/`. This map identifies the
selected paths; it does not assert that their kernels or timing boundaries are
identical. The exact source of the installed native XPU wheel 0.1.15.4 remains
unavailable; `baseline_manifest.json` records its binary hashes.

## Python reference

`LLM.generate` -> V1 scheduler -> `XPUWorker` -> **`XPUModelRunnerV2`** ->
`vllm/v1/worker/gpu/model_runner.py::execute_model` -> Qwen3.5 text model.
The worker log explicitly says `Using V2 Model Runner`. The selected V2 runner
source is preserved in `reference/python_v2_model_runner.py`; the older
`gpu_model_runner.py` is not the active runner for this measurement.

The V2 runner prepares inputs and attention/state metadata before `model()`.
Its `sample()` selects hidden-state rows, calls `model.compute_logits()`, then
invokes the sampler. This split is why the historical `llm_engine.step()` timer
is not a model-forward timer.

For each of 48 linear-attention layers,
`qwen_gdn_linear_attn.py::forward_xpu` ->
`torch.ops.vllm.gdn_attention_core_xpu` -> `vllm/_xpu_ops.py` registration ->
`torch.ops._xpu_C.gdn_attention` in the installed wheel. The 4096-token
profiler records 48 calls to each op and 48 launches each of
`ChunkPrepareKernel`, `ChunkComputeAKernel`, `ChunkInverseOptKernel`,
`ChunkComputeWUKernel`, `ChunkFwdOKernel`, and
`chunk_update_states_kernel`. This proves the active native GDN path; the
constructor's `Triton/FLA` warmup log does not prove that prefill uses FLA.

GPTQ uses `auto_gptq.py` -> XPU mixed-precision linear kernel ->
`torch.ops._xpu_C.int4_gemm_w4a16` -> profiled `gemm_kernel` launches.
Full attention routes through vLLM's `flash_attn.py` and the installed
`vllm_xpu_kernels/flash_attn_interface.py` wrapper to
`torch.ops._vllm_fa2_C.varlen_fwd`. Its profiled Xe2 kernels include
`XeFMHAFwdKernel` and `XeFMHAFwdSplitKVKernel` (16 of each in this prefill
trace). The same trace contains 16 FP8 `reshape_and_cache_flash_strided_kernel`
launches. Exact native implementations and launch geometries for wheel
0.1.15.4 remain open, as do per-layer FP8 scale values.

## C++ branch

`ModelRegistry::Forward` -> Qwen3.5 dense text forward -> 64 layers: 48 GDN
and 16 full attention. GPTQ projections use `MatmulGptq4Kernel` with oneDNN
W4A16; the FP16 LM head uses the existing oneDNN dense primitive and FP32
output cast. For GDN prefill, `vt::GdnPrefill` -> `GdnPrefillKernel` ->
`GdnChunkedPrefillKernel`; the two-chunk variant is default. For attention,
`vt::PagedAttention` selects the Xe2 prefill kernel at 4096 tokens and the
split kernel for the tested decode step.

The diagnostic route log has one warmup and one measured forward. It records
96 `gdn_prefill=chunked`, 32 `paged_attention=xe2_prefill`, and 32
`paged_attention=split` selections in total: 48, 16, and 16 per run. It is a
route check, not a scored latency measurement.

## Shape and precision facts / open mapping

The checkpoint text config has hidden width 5120, 64 layers, 48 linear-
attention layers, 16 full-attention layers, GDN key heads 16, value heads 48,
key/value dimension 128, convolution kernel length 4, and full-attention head
dimension 256. The initial comparison uses FP16 activations, FP32 persistent
SSM state, and FP8 E4M3 KV with 1600-token pages. Per-op strides, FP8 scales,
queue identities, launch geometries, and temporary bytes still need a
structured trace for both engines; these fields are **not** inferred from
the model config.
