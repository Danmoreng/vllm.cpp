# Qwen3.8 Flash Next: TensorFold gap

## Result

The 2026-09-29 DGX reproduction attempt is
`BLOCKED_MISSING_ARTIFACTS`. It publishes no performance number and no
cross-engine ratio.

Two successful repository `rc` jobs established access; a fresh successful audit
lease then executed the committed bounded scan script on `dgx:gpu0` (NVIDIA
GB10, driver 580.173.02). Raw job IDs are not retained; the evidence stores only
a SHA-256 lease fingerprint. The raw receipt establishes absence only from the
two documented staging paths and from `/workspace` to depth 3 for the committed
TensorFold, MiaAI, Vontra, MLX-MTP, and Flash-Next-MTP patterns. It does not
claim exhaustive host-wide absence. Those bounded locations contained neither
required model arm:

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

The execution-plan outcomes are explicit:

- **Task 5:** `SKIPPED_NO_PROFILE` / `NO_PORT_DECISION`. Incremental QSA is not
  implemented because Task 4 produced no profile proving repeated compressor
  work material.
- **Task 6:** `BLOCKED_NO_MTP_WEIGHTS`. The selected GGUF has no MTP head.
- **Task 7:** `SYNTHESIS_NO_PRODUCT_OPTIMIZATION`. Synthesis still executes to
  publish these outcomes, but there is no profile-justified product change to
  retain. Long-context selection, PLE, prefill geometry, and copy drafting have
  no measured bottleneck or correctness baseline.

The campaign can resume measurement only after both exact artifacts and pinned TensorFold
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
