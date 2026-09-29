# Qwen3.8 TensorFold gap blocker evidence

Status: `BLOCKED_MISSING_ARTIFACTS`.

Two successful repository `rc` jobs inspected `dgx:gpu0`; only SHA-256 fingerprints of their job IDs are retained. The leased `/workspace` had neither the staged UD-IQ1_S directories documented by earlier records nor a TensorFold MLX/MTP checkpoint or pinned TensorFold source. Local Hugging Face state contained a repository ref but no snapshot or weight bytes. No artifact was downloaded.

Because prerequisites were absent, no server was started and no correctness, timing, clock, memory, ladder, or profiler result was produced. Every workload and both same-tool profiles are explicitly `NOT_RUN_PREREQUISITE`; there is no cross-engine ratio. Production vLLM remains a named blocked denominator.

The MTP verdict is independently `BLOCKED_NO_MTP_WEIGHTS`, derived from the committed real-header manifest `tests/vllm/models/qwen4_exp_gguf_manifest.inc`: 1,224 tensor entries, trunk blocks 0 through 47, and zero tensor names matching `mtp`, `nextn`, `draft`, `eh_proj`, `enorm`, or `hnorm`. This says nothing about the absent TensorFold MLX checkpoint.
