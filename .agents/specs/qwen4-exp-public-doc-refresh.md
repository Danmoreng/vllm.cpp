# Qwen3.8-Flash-Next public documentation refresh

## Now

Documentation audit at base `cef9f8216`, 14 September 2026. No model lifecycle,
runtime behavior, or benchmark acceptance changes. Implementation and independent
review are pending. One pull request carries the spec and documentation commits.

## Scope

Replace the contradictory `Qwen4ExpForConditionalGeneration` entry in
`docs/FEATURES.md` with a concise account of current behavior. Add one README
news item for real-checkpoint text generation on CPU and ROCm. Preserve the
superseded feature cell verbatim in an era-stamped file under `.agents/completed/`.
Do not edit the model matrix, kernel code, benchmark numbers, or unrelated docs.
The existing EXL3 and HTTP multimodal documentation pull requests own those topics.

## Source and evidence inventory

| Surface | Authority | Required disposition |
|---|---|---|
| Production decode | `src/vllm/model_executor/models/qwen4_exp.cpp`, `qwen4_exp_forward.cpp`, and the model registry | Verify actual registration, decode routing, and one-sequence limit |
| Backend dispatch | `src/vt/rocm/rocm_qwen4_exp.hip`, quantized embedding registration and loader guards | Distinguish reachable code from measured real-checkpoint output |
| CPU checkpoint | `docs/bench-evidence/qwen4exp-released-checkpoint-tokens-20260831.md` | Prompt-dependent generation, no oracle token gate |
| ROCm checkpoint | `docs/bench-evidence/qwen4exp-rocm-hcnorm-gfx1151-20260913.md` | Generation on gfx1151, correctness and competitive speed still ungated |
| CUDA and artifacts | `docs/USAGE.md`, retained CUDA token evidence, owning model spec | No obsolete claim that CUDA emits no tokens; preserve actual unresolved token disagreement |
| Oracle | `.agents/oracles/vllm.md`, `llama-cpp-qwen4exp.md`, owning model spec | vLLM registration exists; no claim of a completed primary-oracle run |

Read exact paths and line anchors before writing. The public feature entry must
name the tested UD-IQ1_S GGUF, single-sequence serving, text-only artifact, and
unresolved correctness. Link evidence instead of repeating attempt history.
Do not publish an engine ratio or promote a liveness measurement to parity.

## Design and risks

Keep the existing feature-table columns. Use short sentences and descriptive
links. README news links to the feature entry or its evidence. Do not introduce
new commands: existing usage recipes retain authority. Archived relative links
must resolve from the archive, without changing the quoted source cell itself.

The main risk is mistaking a stale sentence for current behavior. Reconcile
conflicts against source and later committed evidence. Runtime verification is
unavailable on this CPU-only host; no new runtime claim is permitted.

## Tests and gates

This is a documentation correction, with no upstream test port or runtime
benchmark applicable. Capture the contradictory before text as the failing
baseline. Run existing `check-readme-structure.py`, `check-supported-models.py`,
`check-site.py`, `check-agent-record.py`, and their relevant mutation suites.
Run full preflight and distinguish unchanged baseline failures from regressions.
An independent reviewer checks every new claim against source and evidence,
checks preserved archive bytes, and mutates links in a scratch copy to verify
the link gate. The operator reruns the focused checks at the reviewed SHA.

## Completion and stop conditions

Done when scoped documentation is source-checked, focused checks pass, a fresh
review passes, and the fork pull request is open. No upstream merge is authorized.
Stop for a claim requiring a new GPU run; describe the gap without filling it.

## Owed

`ISSUE-LOCAL-01M2EXYPHDZ03EX01VRTB5FECQ` owns this documentation correction.
The model's existing runtime issues remain with its owning spec.
