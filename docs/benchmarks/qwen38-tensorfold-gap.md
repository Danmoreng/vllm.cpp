# Qwen3.8 Flash Next: TensorFold gap

## Result

The 2026-09-29 DGX reproduction attempt is
`BLOCKED_MISSING_ARTIFACTS`. It publishes no performance number and no
cross-engine ratio.

Two repository `rc` jobs successfully inspected `dgx:gpu0` (NVIDIA GB10,
driver 580.173.02). Their raw job IDs are not retained; the evidence stores only
SHA-256 fingerprints. The leased workspace contained neither required model
arm:

- the staged `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S bytes needed by
  vllm.cpp were absent from both previously documented workspace locations;
- the `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` checkpoint required by
  TensorFold was absent;
- no pinned TensorFold or MiaAI recipe source checkout was present.

Local Hugging Face state had a repository ref for the GGUF repository but no
snapshot or weight bytes. Operator checkpoint configuration did not resolve to
an available path. No large artifact was downloaded.

## Consequence

Artifact hashes and launch environment files could not be created from measured
bytes. Therefore no server was started and the executable correctness gate did
not run. Serial decode, drafted decode, prefill and serving ladders, busy clock
windows, memory capture, and same-tool profiles are all
`NOT_RUN_PREREQUISITE`. Production vLLM remains the required denominator and is
recorded as `BLOCKED`: no policy-authorized fitting safetensors artifact was
present, and production vLLM cannot consume an absent GGUF.

TensorFold's publisher rates remain **unverified publisher claims**. They are
not local observations, a target derived from a local profile, or evidence for
an optimization. No ratio, speedup, winner, residual gap, or ceiling is stated.

## MTP artifact verdict

The vllm.cpp GGUF verdict is independently
`BLOCKED_NO_MTP_WEIGHTS`. The committed real-header manifest
`tests/vllm/models/qwen4_exp_gguf_manifest.inc` pins revision
`8bdc666649440e9bdc97e16f3f75782c98478ff5` and contains 1,224 tensor entries.
It has trunk blocks 0 through 47 and zero tensor names matching `mtp`, `nextn`,
`draft`, `eh_proj`, `enorm`, or `hnorm`. This verdict applies to that GGUF; it
does not claim anything about the absent TensorFold MLX checkpoint.

Consequently, native MTP cannot be implemented against the selected GGUF bytes.
Conversion is not authorized or justified here, and no speculative loader is
scoped for weights that are absent.

## Follow-on disposition

W2 through W6 are not justified by a Task 4 profile:

- incremental QSA and long-context selector work have no local profile premise;
- native MTP is additionally blocked by missing GGUF MTP weights;
- PLE, prefill geometry, and copy-draft work have no measured bottleneck or
  correctness baseline.

The campaign can resume only after both exact artifacts and pinned TensorFold
sources are staged through an authorized path. It must then rerun correctness
before timing rather than reusing this blocker as benchmark evidence.

## Evidence and validation

Versioned evidence:
`.agents/evidence/bench-qwen38-tensorfold-gap/20260929T180547Z/`.

Validate it with:

```sh
python3 tools/bench/validate_qwen38_tensorfold_evidence.py \
  .agents/evidence/bench-qwen38-tensorfold-gap/latest
```
