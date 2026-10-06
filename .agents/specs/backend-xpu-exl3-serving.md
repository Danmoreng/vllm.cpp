# Native EXL3 text serving on Intel XPU

## Now

Row: `BACKEND-XPU-EXL3-SERVING`. State: `SPIKE`.
Canonical issue: `ISSUE-GH-3331`.
Base: `0fbd7c994fd40f4e62f9a75ad135146b00f0952c`.
The first commit records this integration contract before importing product code.
The next step composes the complete serving chain, then builds the public server.
Composition, server execution, independent review and reference qualification
remain separate gates. No gate is inferred from this specification.

## Scope

Deliver one native text-serving contribution for Qwen3.8-27B EXL3 on Intel
Arc Pro B70. Include the XPU operations and complete target/MTP dependency chain.
Preserve target-only inference, the full target head, compact draft vocabulary,
existing graphs, recurrent state, sampling and prefix integration.
Keep shared oneDNN, reachable experimental INT4 operators and safety fallbacks.

The public endpoint is the normal server through `include/vllm.h`.
An internal model test alone cannot discharge that endpoint requirement.
CPU builds must remain independent of oneAPI, model files and reference Python.

Exclude vision, multiple devices, new formats and performance redesign.
Exclude private development records, local instruction changes and old ancestry.
Do not improve chunk alignment, GDN kernels, precision or MTP settings here.
Do not redistribute model weights or large captured tensors.

One pull request carries the specification and implementation.
The development source predates this spec and used a separate local workflow.
This spec governs its current integration, not that earlier implementation.

## Upstream chain

The ordinary behavior pin remains the revision in `../upstream-sync.md`.
Do not change that global pin or label a custom EXL3 plugin as stock vLLM.
The existing local source checkout is at `e126687a9a`, behind the current pin.
Reference gates needing the current pin remain pending until it is available.

Read the matching vLLM platform, Qwen, scheduler, sampler and MTP paths.
Use the model-author EXL3 format and explicitly pinned XPU donors where stock
vLLM has no implementation. The existing secondary registry is not replaced.
Its exllamav3 pin and fleet gateability do not qualify this B70 plugin recipe.
Record this backend-specific reference scope alongside actual binary provenance.

Existing donor source identities:

| Component | Source revision | License |
|---|---|---|
| EXL3 ESIMD | `c59d9442aba8610188837e37724600f1517d7335` | MIT |
| XPU attention | `6d92b1bfbf32767ecda8e819613eb151e70030ad` | Apache-2.0 and individual BSD notices |
| Xe2 GDN | `ddf336d86e3c8602888572a3502f951abd51df12` | Apache-2.0 and individual BSD notices |
| SYCL-TLA | `87f6850680a580654b9ea2c80dbc01aeb36ad231` | BSD-3-Clause |
| oneDNN 3.13.0 | `0e2a5bfeef1bfbffc3137464606540233086ce9b` | Apache-2.0 |

The original request is upstream issue 3331. Its historical GPTQ-first wording
does not override the present EXL3 serving scope.

## Our baseline

The pinned base has no native XPU platform implementation or SYCL backend.
The backend matrix records the older XPU hardware-blocked exploration.
The normal server already routes through the public C ABI.
`examples/CMakeLists.txt` builds target `server`, output name `vllm-server`.
`VLLM_CPP_SERVER` enables the library's HTTP and server-entry translation units.
These source observations do not prove an XPU server build or model execution.

The reviewed development source is `4a5485c1297f06482b1d5b41946ed1997b0f4d4f`.
Its separate composition patch starts at `e67a071a649814e102a91145a0b1b424884f16da`.
Patch SHA256: `48393ffe8ba34a55eea9eb707a5fe58a0c972a4b5e65f276269cca538e7acee7`.
A check on this integration base accepts that patch without editing the tree.
The inventory selects 259 paths; it is not a size or completeness requirement.
The newer benchmark driver and development report are not imported wholesale.

Historical in-process serving evidence completed C1–C4 with four admitted slots.
The attempted native 16-slot recipe hung or failed device allocation.
The longest measured input was 199673 tokens plus 1024 output tokens.
Those results neither qualify HTTP serving nor the configured maximum boundary.

## Port map

| Area | Contribution purpose |
|---|---|
| XPU backend and platform | Device operations, allocation, queues and backend selection |
| Model loading and Qwen dense forward | EXL3 checkpoint/config resolution and full target execution |
| MTP speculator and compact vocabulary | Native draft/verify with the exact token mapping |
| Scheduler, worker and prefix owner | Admission, streaming lifecycle, recurrent state and cache ownership |
| Sampling and structured output | Preserve shared device sampling and public output contracts |
| Graph support | Preserve graph/eager routing, retirement and slot reuse |
| Shared oneDNN/INT4 support | Keep dependencies used by EXL3 and bounded reachable GPTQ paths |
| Public C ABI and server | Expose the complete chain through the normal executable |
| Tests, dependency notices and portable tools | Reproduce guards without private machine inputs |

Apply changes by file-level patch. Preserve unrelated current-main edits.
Review the CMake, test-CMake and documentation export derivatives explicitly.
Review the newer full-model test delta before selecting any public test changes.
Preserve donor modes, notices, source-specific floating-point flags and pins.
Record selected-path dispositions outside public source when they carry private data.

## Tests to port

Retain the selected CPU loader/config, shared-cache and model-free tests.
Retain XPU synthetic operator, dtype, guard and ownership tests.
Retain portable required/optional external-artifact admission and SmallM coverage.
Retain poisoned-slot, target/MTP and recurrent-prefix regression coverage.
Do not turn default reference failures into xfails or widen tolerances.
New server tests must enter through the actual public HTTP/API surface.

External integration jobs must enforce model and draft-map presence.
A legacy exit77 means skipped, never passed. Present corrupt data must fail.
Reference tools are optional test dependencies, never native inference dependencies.

## Gates

The pristine-base `bash scripts/agent-preflight.sh --quiet` exited 1.
It ran for 623 seconds on the pinned base before any tracked edit.
Failed gates: `test_tower_skip_rss_arm`, `test_rocprof_attach_preflight`,
and `tools suites`. Twelve gates were skipped: seven Python capture, render,
conversion and adherence suites lacked numpy; five checker entry points needed
arguments that automatic preflight does not supply. These remain open findings.
The records-only compile scope was empty; it does not count as a CPU build.

Preserve the pristine-base preflight's actual exit and findings.
Classify subsequent findings as introduced, baseline, unavailable or out of scope.
A baseline failure is not a waiver. Do not repair unrelated checks in this PR.
Run the applicable staged, focused and full gates required by upstream policy.
Record exact commands and actual results instead of copying historical verdicts.

1. Clean CPU configuration builds the normal library with XPU disabled.
2. Fresh XPU configuration builds the normal server with server support enabled.
3. Inspect the compiled server's help; document only actual public options.
4. Start that source-built server with explicit model and compact draft inputs.
5. Exercise model identification, template handling, non-streaming and streaming.
6. Count actual tokens, not SSE chunks; prove target-only and bounded MTP3 use.
7. Exercise four short concurrent requests, cancellation, EOS and freed-slot reuse.
8. Check request isolation, output limits and graph/state ownership.
9. Compare a C1 4K/O256 sentinel and short lifecycle against unchanged native control.
10. Run one aligned cold/warm prefix check and record the batch4096 limitation.
11. Record memory, allocation failures, host OOM, swap and contention observations.
12. Inspect the final tree and every introduced commit for private material and ancestry.

Only a diagnosed sentinel regression can justify an extra 4K/O1024 run.
No complete benchmark matrix is required or authorized by this integration scope.
Independent static/mutation review remains pending until a distinct reviewer runs it.
A self-review cannot discharge that gate. Preserve the developer's single-agent setting.

## Dependencies

Checkpoint: `turboderp/Qwen3.8-27B-exl3` at
`113cf7ab958054860e43fb7f3063b1af19171095`.
EXL3 body: 4.00 bpw. Full target head: 6 bpw, 248320 rows.
Compact draft: exact 65536-token mapping. MTP depth: 3.
Activations: FP16. Persistent GDN state: FP32. KV: FP8, physical page 1600.

Use the existing measured toolchain and dependencies; do not upgrade them.
Use explicit external input paths. Do not download models during build or tests.
The available local accelerator is one Arc Pro B70 at the existing 180 W limit.
Use the upstream non-fleet mutex; oracle and native GPU runs are sequential.
Production vLLM remains stopped under the developer's existing authorization.

## Work breakdown

1. Commit the current issue/spec and scoped backend inventory row.
2. Compose the reviewed full feature without old development ancestry.
3. Resolve public-server build/entry-point gaps with focused coverage.
4. Run bounded serving and unchanged-native continuity checks.
5. Finish portable build/run docs and the accurate capability table.
6. Inspect all new commits and prepare one self-contained PR body.
7. Freeze the exact final contribution commit as PR_BASE.

Do not open or merge an upstream PR without submission authority.
Do not begin a descendant performance branch inside this task.

## Risks and decisions

The verifier opt-in remains experimental; preserve guards and numerical failures.
Existing default P128/D64 and integrated MTP state gates are not cleared by throughput.
Four slots cover measured C1–C4, not sixteen admitted requests or four maximum contexts.
The 262144 configured context is not a tested maximum-boundary qualification.
Prefix reuse works with aligned chunks; batch4096 resend failure remains open.
Documentation must keep those configurations and results separate.

A server repair is in scope; numerical redesign and performance tuning are excluded.
A maintainer prerequisite must name its exact gate rather than silently expand scope.
Stop on destructive conflicts, unavailable required resources or unresolved behavior.
Keep the branch reviewable when merge approval or independent review remains pending.
No exception can be recorded as a passing correctness or review gate.

## Owed

Independent review, current-pin reference availability and known numerical/capacity
qualification remain explicit gates. Existing owning rows retain their obligations.
This integration does not silently close those reference or performance gaps.
