# B70 EXL3: Upstream Integration, Portable Qualification, and the Remaining Performance Gap

**New implementation plan — 5 October 2026**

**Reviewed snapshot:** `3803ad4a6b9974f4afd1b6c9ad3e333d5e9b6ff8`  
**Last product commit:** `0ea34de4e005688ebc80f28137c93c83b9c461c3`  
**Development repository:** `Danmoreng/vllm.cpp`  
**Intended upstream:** `mudler/vllm.cpp`, not `vllm-project/vllm`  
**Upstream main observed during this review:** `e67a071a649814e102a91145a0b1b424884f16da`

## 1. Decision

Preserve the current native implementation. End the old P0–P7 campaign, as requested in the handoff. Do not restart its completed investigations.

Use this order:

1. Preserve the measured checkpoint and original evidence; make one small portable-test improvement.
2. Integrate the pinned upstream main in a separately identifiable integration commit, without performance changes or dependency upgrades.
3. Establish a reviewable, portable product/test/documentation boundary and an explicit supported-route contract.
4. Implement at most two primary new performance candidates, plus one conditional shape-specific candidate, using the current code rather than historical bottleneck lists.
5. Qualify the intended ordinary launch path and present small, dependency-ordered upstream changes.

**Do not make full Python speed parity a prerequisite for discussing or submitting a correctly scoped foundational PR.** Equally, do not claim that the complete backend has passed upstream parity/performance requirements while its declared gates remain open. Maintainers decide whether a limited or experimental contribution is mergeable.

This is one native implementation, not two deployed quality/performance profiles. An immutable development baseline and an offline oracle are test instruments. Safe operator selection for unsupported shapes is not a second model profile.

## 2. Review basis and limits

This review used the complete uploaded source snapshot, selected original-runtime sources, and the archived raw serving JSONs. It also read current upstream metadata, contribution guidance, changed-file information, and relevant open PR descriptions through GitHub, without changing any remote resource.

Independent checks performed here:

- All **9,520** manifest payloads matched their recorded sizes and SHA-256 values.
- All **8,880** recorded tracked-source Git blob identities matched.
- **93 host tests in 13 suites passed.** A separate provider-trace Python script was initially invoked without its required compiled probe argument. That invocation failed because of reviewer test preparation; the probe-backed test remains unexecuted here. Its CMake registration supplies the missing executable correctly.
- C1/C4 primary serving metrics were recomputed from request observations rather than copied from summary tables. Native output IDs and non-timing cycles repeat exactly within each three-run series.
- Selected source paths, build configuration, test exit behavior, ownership boundaries, and remaining optimization opportunities were inspected.

Not performed here: a native SYCL build, GPU inference, new profiling, a real upstream merge rehearsal, a complete review of every source file, a full-history secret scan, or an independent comparison of omitted large tensor arrays. Container network access was unavailable; current GitHub reads used the connected GitHub API. The archive has no complete working Git history from which to execute a merge locally.

The attached records are evidence, not an instruction to rerun every historical experiment. See Appendix E01–E15 and the companion audit files for exact checks and source excerpts.

## 3. What the results establish

### 3.1 Primary C1 serving comparison

P4096/O1024, MTP3, cold prefix after the existing O32 warmup:

| Metric | Final native median | Pinned original median | Interpretation |
|---|---:|---:|---|
| Emitted decode tokens/s | 51.781375 | 63.863406 | Native is 18.9186% below original throughput |
| TTFT | 2.330729 s | 1.951991 s | Native is 19.4026% slower |
| End-to-end | 22.086893 s | 17.970134 s | Native is 22.9089% slower |

The native trials use the same final product source; the first trial reuses the approved same-source typed-GDN default run and the other two were newly executed. The three original trials are immutable earlier measurements reused by the handoff, not newly executed after the final native optimization. The slower third original continuation remains in the median.

Native repetitions produce the same 1,024 IDs and 371 non-timing cycles, with 656 accepted / 1,104 proposed draft tokens. Cross-engine continuations differ. This is autonomous-serving evidence, not an identical-trajectory kernel experiment.

To reach the original rate from the native rate requires **23.3328% higher native throughput**, not simply a 20% increase. With 368 native decode cycles for 1,023 post-first output tokens, the observed native yield is 2.77989 output tokens/cycle. Matching 63.8634 tokens/s at that yield would require approximately **43.5287 ms/cycle**. This is an illustrative budget, not a measured original GPU-cycle latency.

TTFT must not be called pure prefill kernel time. It includes additional work up to the first output.

### 3.2 Concurrency is not yet a like-for-like GPU comparison

Native C4 P4096/O1024 medians: 44.467989 s batch elapsed, 129.524802 emitted tokens/s during the common four-request decode interval. The original batch median is 76.083842 s, but its observed decode overlap is **one**, versus **four** in the native run. The independent request-timestamp calculation confirms this for all repetitions.

The native batch completes sooner under the measured scheduling behavior. It does **not** establish that native four-way target kernels outperform original four-way target kernels. Do not change the historical numbers to hide this distinction.

For a new matched C4 GPU comparison, first establish genuine simultaneous original execution and record the actual active batch. Keep the historical serving-policy comparison separately. If the original continues to serialize under its unchanged policy, report matched GPU C4 parity as unavailable rather than manufacturing a rate by adding per-request throughput.

### 3.3 The implementation has advanced substantially

Already delivered and retained:

- Physical-page-1600 packed verification, including the qualified C1 Q2–Q5 and uniform C2/C3/C4 Q4 cases.
- Group-correct W8A8 panels, default 1,024 columns, instead of the old 128-column-only organization.
- Exact typed and lookup-based SiLU/norm changes, with explicitly bounded NaN-payload qualifications.
- Immutable map certificates and completion-owned validation/workspace reuse.
- Coalesced speculative GDN SLM and its final typed-memory path, preserving full FP32 snapshots and scalar ascending-K arithmetic.
- Native MTP3, compact draft, graph/prefix/lifecycle functionality within the documented tested scopes.

The old 34.7313 -> 51.7814 tokens/s comparison is descriptively about +49.09%. Do not present it as one cumulative identical-output causal experiment: attention arithmetic and proposal yield changed during the campaign.

### 3.4 Qualification is still scoped

Preserve the following ledger rather than relabeling it:

- Practical screen: semantic 8/8, including four small code functions, two **60,872-token** retrievals and two structured/tool tasks. Sequential requests with C4 capacity are not concurrent-C4 semantic coverage.
- Isolated compact draft/head: complete documented tensors pass. Not all evolving MTP iterations are covered.
- Frozen default P128/D64 target gate still exits 1 with four failures: D27 TV/KL and D29 TV/KL. D29 TV is about 0.212752, not a negligible numerical deviation.
- All 65 complete rows match the separately identified controlled deterministic-B/A oracle. That does not silently replace the frozen default oracle.
- Two held-out eager/no-MTP P128/D1 prefixes have complete hybrid-state equality after the signed-zero repair. This is not an integrated original MTP/C4/long-context state proof.
- Full 262,144-token generation, broad stochastic/RNG equivalence and integrated original-state qualification remain open.
- The 51.8 tokens/s benchmark explicitly selects `VT_XPU_XE2_VERIFY=1`. Arbitrary ordinary defaults are not yet proven to reproduce that rate.

A source-cleanup commit must not change this ledger. A performance commit must not relax its thresholds. A reference-contract correction, if justified, is a separately reviewed change backed by repeatability and held-out evidence, not by selecting a favorable captured run.

## 4. Upstream status and merge decision

The connected GitHub metadata identifies `Danmoreng/vllm.cpp` as a fork of `mudler/vllm.cpp`. This is the native project relevant to this work. Python `vllm-project/vllm` is the oracle project, not the repository into which this C++ tree should be merged.

At the reviewed main pin, the compare API reports:

- Common ancestor: `9d96b162c8c3dfa8f0143932962e445b68fe3a8b`.
- **61 upstream-only commits**.
- **168 branch-only commits**.

These are dated observations. Resolve and record the actual upstream ref once at implementation start; do not repeatedly chase a moving main during a performance A/B.

Changed paths overlap in `CMakeLists.txt`, `tests/CMakeLists.txt`, the Qwen dense model and weight headers, and `qwen3_5.cpp`, `qwen3_5_dense.cpp`, and `qwen3_5_dense_weights.cpp`. Upstream also changes shared model/ownership infrastructure and test/CI rules. Overlapping paths are a conflict-risk map, not proof that a particular hunk will conflict.

The comparison of branch-only changes reaches GitHub's 300-file response limit. It is **not a complete PR-size inventory**. Produce the full path/line/dependency inventory with local Git after fetching; do not infer a 300-file total from the API response.

### Merge policy

- Preserve the current baseline ref and external evidence before any integration.
- Use an integration branch or equivalent reversible local checkpoint. Do not force-push/rewrite the development history as part of routine cleanup.
- Merge/replay upstream changes in their own commit(s); include only actual conflict resolution and mandatory compatibility repairs.
- No new kernel, compiler option, device policy, driver, dependency version, weight, or benchmark change in the same integration commit.
- Never resolve shared Qwen code by accepting the entire local or upstream file blindly. Preserve the EXL3 FP16 boundaries and ownership fixes as well as upstream's independent features.
- Inspect CPU/CUDA/ROCm/shared build consequences where shared interfaces changed. A successful B70 binary does not establish other backends still compile.
- Compare the integrated native tree against the preserved native baseline, not against a newly changed Python image.

A successful textual merge is not a passed model/state test. Conversely, unrelated pre-existing upstream check failures should be recorded by comparing the clean upstream base, not repaired opportunistically in this PR series.

## 5. Publication hygiene: more than replacing a home directory

The scoped scan found **2,271 occurrences of the operator's personal home-root literal in 14 tracked text files**: 13 documentation/evidence files and one tools README. No occurrence of that particular root was found in `src/`, `include/`, or `tests/`.

This is not a full credential scan and does not establish that all other machine-specific settings are portable. It does establish that the primary problem is packaging and reproducibility, not a runtime dependency on that exact home directory.

### Separate three kinds of material

| Material | Treatment |
|---|---|
| Product code, compact tests, fixture generators, build/usage documentation | Keep in the upstream changes, with portable paths and correct dependency boundaries |
| Original measured receipts, raw logs, old plans, large runtime-source snapshots | Preserve unchanged in a named development/evidence archive; include only the necessary public digest/index and reproducible methods in the PR |
| Personal environment, container names, local services, local filesystem mappings | Move into untracked `.env`/explicit CLI configuration; never make them silently mandatory for a public test |

Do **not** globally rewrite strings in immutable measured JSONs and continue citing the old hashes. A sanitized derivative gets a new hash and a provenance relation to the original. Keep raw private records intact. Artifact identities should use content digests and relative names; local absolute paths are optional location metadata, not the identity.

Removing a path in the latest commit does not remove it from already published branch history. For a clean upstream contribution, use a new contribution branch based on upstream and apply only the selected product/test/documentation changes. Keep the development branch and evidence archive for provenance. If a separate secret scan finds credentials, handle rotation/history remediation explicitly; an ordinary home directory path is not itself a credential.

### Specific portability and reviewability repairs

1. **Fixture resolution:** use explicit `--artifact-root`, `--model`, `--oracle-source`/`--oracle-image`, and an optional untracked environment file. Do not auto-download a 27B model or stop a named personal service during routine tests.
2. **Test outcomes:** the SmallM test currently returns success after a missing-fixture message in one case, while another case in the same executable calls `exit(77)`. Separate hermetic unit tests from external-model/oracle executables. Missing optional artifacts become an explicit CTest skip only for that external test; missing artifacts in a required qualification job make that job incomplete/failing, never green.
3. **Build names:** `VLLM_CPP_XPU_GPTQ4` currently gates oneDNN code also needed by EXL3. Introduce an accurately named internal/optional oneDNN capability, preserve a documented compatibility alias while needed, and test XPU-disabled builds. This is build cleanup, not an upgrade of oneDNN.
4. **Agent instructions:** the development branch replaces a substantial part of root `AGENTS.md` with local workflow instructions. Do not send that local operator policy upstream. Restore the current upstream procedure in the contribution branch; keep local preferences outside the public patch. Do not modify upstream checkers merely to waive this contribution.
5. **Diagnostics:** classify the many `VT_B70_*` capture/probe controls. Keep stable useful diagnostics, isolate test-only hooks, and exclude local receipt-generation orchestration from ordinary inference. Remove dead experimental variants only after checking reachability and preserving their rejection records externally.
6. **Capabilities:** exact device/compiler/driver checks in the verifier are conservative numerical/admission guards, not merely personal paths. Centralize and document them; do not delete them to claim all Intel hardware works. Unsupported hardware needs an explicit supported fallback or a clear unsupported diagnostic.
7. **Licensing/provenance:** retained EXL3 donor sources include MIT licensing; the adapted attention/GDN headers contain Apache/BSD provenance. Inventory all redistributed files, preserve their notices, and update the top-level notice/provenance table where missing. A clean source tree must not erase original attribution.
8. **Source links:** replace fragile line-only citations in published instructions with symbols plus pinned revisions. Historical evidence may retain historical line references.
9. **Fresh-build reproducibility:** a public contributor must be able to build from declared sources/toolchain, not only use a privately named prebuilt image. Record actual binary/source hashes for claims, without bundling weights or opaque reference answers into inference.

## 6. Small work packages

Names below are new tasks, not a restart of old P0–P7. Proposed new filenames are explicitly labeled; adapt them to current upstream naming conventions rather than duplicating an existing helper.

### U0 — Freeze the baseline and make the first portable test entry point

**First concrete assignment.** Add one small fixture-resolution helper (proposed: `tools/exl3_reference/artifacts.py`) and adapt one EXL3 projection test/tool to use it. Use a generated tiny manifest/artifact for the host tests. Do not move the entire historical evidence tree in this first patch.

**Inputs:** preserved product/source ref, `tools/exl3_reference/*`, the existing projection capture/compare tests, `tests/CMakeLists.txt`.

**Behavior:** command-line root overrides environment root; resolve relative artifact names under that root, check declared digest/shape/type, reject traversal/incompatible manifest, and produce a clear missing-artifact result. Local root changes do not change content identity. The helper must not run a model or alter a service.

**Smallest test:** the same tiny artifact in two temporary roots, one containing spaces; identical comparison result/digest; corrupt payload fails; missing payload skips only the optional external check; malformed metadata fails. Run the already available host projection tests.

**Additional deliverable:** a short upstream-delta/dependency map using the recorded ref. Do not author a new giant execution journal. Archive the current source/binary/command identifiers and all open gates.

**Stop rule:** finish one portable slice and commit it. A broader cleanup inventory must not delay U1 or turn into a multi-day receipt exercise. No GPU benchmark is needed for an unchanged arithmetic/test-path-only patch.

### U1 — Integrate current upstream without tuning

**Areas:** the overlapping Qwen dense source/weight/header files, CMake and test registration, shared ownership and model-loader interfaces.

**Implementation:** use an isolated integration commit or a clearly identified replay series. Keep old and new refs. Review each semantic conflict. Preserve source-specific floating-point flags; do not apply global fast-math or formatting to unrelated backend code.

**Checks:**

- `git diff --check` and current upstream's applicable source/registration checks, with a clean-upstream baseline for pre-existing failures.
- Build the changed native targets and a CPU/XPU-disabled configuration. Compile shared affected backend surfaces where the existing toolchain supports them; unavailable hardware/toolchains remain explicitly unverified.
- Existing full focused GDN snapshot/operator and route tests affected by the merge.
- One native C1 P4096/O256 MTP3 and the existing lifecycle test against the preserved native baseline: identical IDs/non-time metadata and no new state/ownership error. A changed sequence triggers a targeted comparison; do not call a textual merge harmless.
- One short cold-prefill timing sanity check; if no regression signal exists, defer the three-repeat primary score to U6.

**Stop rule:** separate incompatibilities from independent optimization work. Revert/bisect the integration if a shared numerical or lifetime regression appears. Do not rebenchmark the original Python image to conceal a merge regression.

### U2 — Prepare a publishable surface and a runnable capability slice

**Areas:** public docs/build options, fixture tests, diagnostics, notices, contribution branch composition.

**Implementation:** apply the hygiene work in section 5. Build the contribution branch from the integrated upstream baseline with only intended changes. Retain the full native functionality on the development branch. Use one implementation and a declared experimental/qualified capability matrix, not parallel product profiles.

**Checks:** a clean checkout with no personal `.env`, no external model, and no oneAPI must configure/build the intended CPU path; portable tests pass, external B70 qualification tests report unavailable explicitly. A declared B70 environment builds and runs the small native smoke using supplied model/artifact paths. Check no accidental paths, credentials, binary dumps, private policy changes or unrelated GPTQ experiments enter the selected diff.

**Policy:** preserve uncertainty rather than documenting full262K, sampled RNG equivalence or integrated reference-state parity as passed. Defaults should not silently promise 51.8 tokens/s until the measured verifier route is admitted normally.

**Stop rule:** reach a reviewable minimal contribution and ask maintainers about the dependency split. Do not complete every optional historical test migration before implementing a new measured performance candidate.

### Q1 — Close the ordinary-route/reference decision separately

This task is qualification, not a way to regenerate passing goldens.

**Areas:** `xpu_attention.cpp`, `xpu_attention_verify_xe2.cpp`, Qwen graph-route policy, the external target/MTP capture tools.

**Minimum useful evidence:** reuse existing exact eager held-out state checks. Add a bounded integrated original/native MTP sequence with identical input tokens, accepted lengths, initial Conv/SSM/KV state, and active rows. Start with C1 Q4 and then a small C4 -> C2 -> C1 transition. Check full per-token FP32 snapshots and only initialized FP8 KV entries; do not compare uninitialized capacity.

Maintain both reference labels. If correcting the default-reference contract, independently establish why the controlled arithmetic is the intended reproducible oracle, obtain held-out evidence beyond the two existing short checks, and get explicit review approval. Do not delete the old D27/D29 failures or widen TV/KL thresholds in a performance patch.

When the required scope passes, use a capability-based automatic selection for the tested B70/page1600 contract. Verify that an ordinary launch without the development opt-in executes the same path and produces the same bounded results as the prior explicitly enabled run. Keep a diagnostic override for A/B if appropriate, not an undocumented mandatory flag.

**Stop rule:** if integrated state parity remains open, preserve the fast route as explicitly experimental and the whole-backend qualification as incomplete. This need not block an independently safe infrastructure PR or same-input performance implementation; it does block claiming the complete backend is production/reference-qualified.

### F1 — Target decode: one producer/consumer fusion candidate

**Hypothesis, not a proven gain:** after the successful SLM rewrite, the next useful difference may be the materialization/launch boundary between speculative post-convolution preparation and the recurrence, rather than another private-register recurrence variant.

**Source entry points:**

- `src/vllm/model_executor/models/qwen3_5.cpp`: `GdnPostConv` outputs and the subsequent `GdnSpecDecode` call.
- `src/vt/xpu/xpu_gdn.cpp`: `GdnPostConvKernel` and the current typed-SLM `GdnSpecDecodeKernel`.
- Operator declaration/registration and focused synthetic/original-operand tests as necessary.

**Bounded precondition:** use one short separate profile or an operator-chain replay to measure the entire post-conv -> speculative recurrence chain on real C1 Q4 and C4 Q4 operands. The latest graph receipt exposes queue spans, not internal kernel attribution. A nearby generic FLA function in the Python tree is not proof that it executes on XPU.

**Implementation:** add a narrow internal fused chain for the existing contiguous F16/F32 Hk16/Hv48/D128 contract. Start C1 Q4. Keep existing WG64/SG32 typed-SLM state update, full snapshot layout, alias behavior and accepted-index selection. Normalize/prepare inputs with precisely the existing reduction ordering and explicit FP16 materialization; do not algebraically bypass those boundaries by keeping more precision. Avoid redundant per-value-row exponentials or repeated normalization that erase the gain. If local-memory pressure would exceed the current resource budget, do not force fusion; a typed producer-only variant is a smaller alternative.

**Required checks:** identical actual Q/K/V halves and g/beta values at the conceptual boundary, full FP32 snapshots for every candidate token, initialized Conv/KV state, accepted counts 1..4, aliased state pools, poisoned spare rows, C1 Q2/Q3/Q4 and uniform C4 Q4. Follow with one native same-output P4096/O256 pair and lifecycle test before any scored full run.

**Memory/ownership:** no new persistent per-layer state copy, no reduced state precision, no early release of an input after async enqueue, no graph-policy mismatch. Debug intermediate materialization may be test-only; product computation may not substitute capture arrays.

**Acceptance:** a reproducible complete-chain gain and a visible full-request/target-cycle benefit beyond measurement noise, with unchanged declared numerical/state behavior. A sub-kernel gain that loses complete-operator or request time is rejected.

**Stop rule:** one fusion and at most one smaller fallback variant. If the measured boundary is negligible or exactness/occupancy costs remove the benefit, stop F1. Do not retry the already rejected private128-state package.

### F2 — Prefill: completion-owned leases, then direct checked-preparation consumption

**Concrete current difference:** checked W8A8 preparation writes quantized activations/scales into a completion-owned candidate buffer; after successful checks, two D2D copies publish them into the public workspace. The matmuls then read those published copies. This is visible in `xpu_exl3_w8a8.cpp`, not the old eliminated repeated validation.

`WithEagerW8A8Workspace` also executes `wait_and_throw()` after each preparation/workspace callback. This is an intentional lifetime fence, not an accidental no-op. Its wall duration includes queued GPU work and must not be presented as removable transfer time.

**First narrow candidate:** replace per-call end fences with a final-consumer event for the private eager pool. A same-queue next user must be ordered after that event; a different queue must depend on it explicitly. Growth, destruction and exception cleanup must wait for the actual previous readers before freeing or reusing storage. Do not allocate an unbounded workspace per invocation or request. Confirm the oneDNN stream participates in the event chain rather than assuming a SYCL barrier covers an unrelated stream. Public error delivery and validation-before-mutation remain intact. Benchmark the full call before adding further changes.

**Second, separately measured candidate:** the native model-owned path consumes the checked preparation buffer directly, removing publication copies/duplicate storage while preserving the explicit-public-workspace contract. The benefit of either candidate is unknown until the complete call is measured. This is not a retry of the old GPTQ GDN-workspace experiment: the concrete consumer chain here is the current EXL3 W8A8 preparation/panel/oneDNN/Hadamard path.

**Source entry points:** `src/vt/xpu/xpu_exl3_w8a8.cpp`, `WithExl3W8A8Preparation` and workspace owner in `xpu_backend.cpp`, private EXL3 grouped call metadata, `exl3_w8a8_panel_plan`.

**Implementation:** introduce an internal prepared-activation view owned until every consuming GEMM and final required operation has completed. Either enqueue all consumers inside an extended ownership scope or attach the final consumer event to the lease. Do not let the current callback end after preparation while later readers still use the storage. Preserve the explicit-public-workspace operator's existing observable success/failure behavior; do not silently stop populating caller-owned scratch if that is its contract.

Keep oneDNN at the same version, panel width 1,024, group boundaries and all W8A8/FP16 rounding unchanged. Preserve error precedence and 'no public output/panel/activation mutation on failed validation'. Do not delete `check_status` or permit invalid INT8 conversion to proceed on an unchecked path.

**Smallest checks:** real M129/M1600/M896 projections including grouped QKVZ and gate/up; exact xq/scales/integer intermediates and final F16 outputs; invalid input/svh/map leaves public buffers unchanged; two queued calls, cancellation/destruction after enqueue, budget exhaustion, and previous generic path. Test reuse after both success and error.

**Performance:** measure the complete projection and the three-chunk 4K prefill, including event/lease overhead and peak device allocation. Profile copies separately only for diagnosis; their time is not assumed to be the whole TTFT gap.

**Stop rule:** at most the two named changes, measured separately. If event-owned reuse or private direct consumption is not materially faster or complicates public semantics without a memory win, retain the relevant current implementation. Do not respond by repeating the 1,024/2,048 panel sweep: the larger setting was already evaluated for a marginal gain and higher scratch use.

### F3 — Conditional ragged verification / remaining compiled-route mismatch

Only start this after a post-integration trace demonstrates material time in an unoptimized form.

The current verifier supports C1 Q2–Q5 and uniform C2/C3/C4 Q4. Nonuniform lengths still use a generic path. First count actual *executed* shape/time, including graph replays, in the C4-long/mixed case. Python-level dispatch logs deduplicated by shape are not execution counts.

If ragged verification is material, test one representative shape such as query lengths [1,4,2,3], using per-request offsets and correct last-visible KV bounds. No cross-request KV sharing, wrong six-head mapping, or reading a rejected future token. Extend packing/metadata and the native packed kernel only if the donor or a separate exact reference supports the same arithmetic. Verify its partial-output/scratch footprint and graph-key retirement.

Otherwise, investigate the one specifically observed FA-output M4 native/donor anomaly or graph-vs-eager compiled-route difference. The existing 13-family/27-shape SmallM result already establishes broad core equivalence within its 10% band; no global DPAS tile search is authorized.

**Stop rule:** at most two variants for one demonstrated expensive shape. Do not rebuild a general speculative attention framework or retune every matrix.

### U6 — Bounded final comparison and contribution handoff

Run the small matrix in section 7 only after accepted code changes. Freeze exact source/binary/route/tool identities. Use unprofiled scores and separate profile runs.

Prepare a PR-ready result table, open-gates table, reproducible test commands, dependency/provenance list and a short explanation of each performance change. Do not paste thousands of lines of execution history into a PR body.

A candidate is retained only if it preserves its declared correctness/safety scope and improves the intended complete operation/request. Aim to move toward original latency, not merely to improve an intermediate TFLOPS number. If the attempt budget ends without parity, report the remaining quantified gap and submit the correctly scoped functionality/performance improvements; do not declare parity achieved.

## 7. Fixed validation and performance matrix

These are proposed budgets for the next work period, not measurements already run.

| Stage | Cases | Repetitions / purpose |
|---|---|---|
| Portable tooling | tiny generated manifests in two unrelated temporary roots; missing/corrupt inputs | host-only, per patch |
| Merge smoke | C1 P4096/O256 MTP3; existing lifecycle; directly affected focused operator/state tests | one before/after native pair, repeat only on a signal |
| Local kernel attempt | real affected projections or GDN/attention chains | short warm operator comparison, separate from serving score |
| Final C1 score | P4096/O1024 MTP3 | three unprofiled native trials; paired current original control for publication when authorized |
| Final target-only sentinel | C1 P4096/O256 MTP0 | one per arm; separates ordinary target cost from speculation |
| Final C4 | P4096/O1024 each, MTP3 | initially one, then three only after genuine original C4 overlap is demonstrated |
| Long/mixed | C1 P32768/O256, C4 P32768/O256, staggered4K/32K | one per arm or retain a valid unchanged control with explicit date; no broad sweep |
| Lifecycle/prefix | existing1/4/2/1, EOS/cancel/reuse, poisoned spares, graph/eager/graph; prefix32K/O64 | same-state/output checks; timings only where instrumented consistently |
| Quality/reference | affected frozen same-input/state gates plus bounded integrated MTP tests | preserve default-vs-controlled labels and expected open failures |
| 262K | only if the release claims actual maximum-context generation | separate explicit qualification, not inferred from configuration or60K retrieval |

Use fixed 180 W, the same checkpoint/subset/model and software stack, one GPU worker at a time, the same 24-GiB/no-swap/two-CPU envelope where it fits the existing contract, and identical warmup/cache-reset conditions. Native tracked memory and original Torch allocation/reservation are different accounting scopes.

Measure actual output tokens, target and draft queue spans, full MTP cycle time, accepted/proposed/emitted counts, TTFT, request elapsed time, chunk-observed pauses, common-overlap throughput, compiled route and active shapes. Do not add nested queue spans or treat MTP chunk timestamps as individual-token latency samples.

For arithmetic-preserving native A/B changes, require identical tokens and non-time cycle fields in the representative autonomous run plus appropriate full intermediate/state tests. For an intentionally different qualified numerical route, document that difference and use matched-operand/state tests; a more favorable continuation is not a kernel speedup.

Suggested engineering triage: seek a repeatable >=2–3% complete target-cycle/request improvement or an independently useful memory reduction. Reject regressions outside noise in a protected primary case unless explicitly accepted for a measured, documented tradeoff. These are iteration criteria, not universal upstream acceptance limits or a reason to weaken correctness.

## 8. Upstream PR structure

Discuss the dependency map with upstream before preparing one enormous feature diff. A reasonable starting split is:

1. **Native XPU substrate and portable tests:** queue/allocator/provider integration, build configuration, independently callable operators and synthetic tests. The change must build and expose a real reachable/tested capability; do not submit dead code whose only caller exists in a future unreviewable patch.
2. **EXL3/Qwen target integration:** grouped SmallM/W8A8, model loading and FP16 boundaries, target-only end-to-end behavior, the needed compact external fixture manifests/generators and accurate support documentation.
3. **MTP3, graph ownership, recurrent prefix and verified fast routes:** complete dependent behavior and lifecycle tests. Preserve an explicit experimental scope if integrated original-state or stochastic equivalence is not qualified.

If the dependency graph requires a different cut, adjust the boundaries. Safe standalone shared fixes can precede the backend changes. Do not automatically include unrelated GPTQ experimentation, all old reports, or the locally rewritten agent protocol.

At the reviewed upstream pin, `scripts/check-pr-size.py` explicitly says the old line budgets are retired. **There is no current 900-line hard limit to cite.** It enforces path classification and checker-evidence/PR rules; splitting is a reviewability/dependency judgment, not a fabricated numerical policy.

Read the current upstream `CONTRIBUTING.md`, `AGENTS.md` and applicable workflow at the pinned integration ref. Their actual contribution requirements apply. The reviewed guide describes squash landing and protocol/AI attribution trailers. Do not fabricate historical commit signatures or rewrite the public development branch to satisfy them; prepare valid contribution commits and the final composed squash message.

The local merge of upstream into an integration branch is a different operation from landing a PR upstream. No remote PR, issue, merge, force push, service change or GPU action was performed by this review. Publishing/landing remains an explicit implementation-stage action.

One relevant open tokenizer PR was found during the current search (upstream #3363). It concerns EOS/BOS resolution for some EXL3 Qwen checkpoints. Recheck its actual status and applicability during integration; do not cherry-pick an unmerged change blindly or diagnose this exact checkpoint from its PR description alone.

## 9. Completion and stop rules

A useful end state has three independent labels:

- **Portable/reviewable:** clean contribution diff, build-from-source recipe, notices, fixtures, explicit skips, no personal inference/test root dependencies, preserved upstream procedure.
- **Qualified capability:** exact declared model/route/state/lifecycle scope passes; unqualified sampling/262K/default-reference issues are not hidden.
- **Performance:** measured current difference on the same workload and execution scope, with no claim of C4 parity from serialized original runs.

These labels must not be collapsed into 'done'. A foundational PR may be useful before full-engine parity, subject to maintainer approval. A model PR may not claim fully qualified performance by renaming an old failed gate.

Execution discipline:

- One assistant, one active implementation attempt and one GPU worker; normal commits between distinct changes.
- No new resource-lease system, broad benchmark campaign, power/driver/library/model/draft search, or repeated rejected kernel family.
- Every work package ends with a small code/test result, an honest measured outcome, or a precise blocker. Documentation is an output of work, not its substitute.
- Do not redo the 13-family SmallM study, the private-state GDN failures, the panel sweep, or solved norm/SiLU/map work without new causal evidence.
- Preserve known failures and immutable raw records. Missing binaries/arrays are missing evidence, not passes.
- When two candidate variants fail to deliver useful complete-path progress, stop that candidate and continue the next independent task. Do not let the whole project become an unbounded 'match Python' goal.

## 10. First prompt for a new Codex session

> Work from the current native EXL3 checkpoint and this new plan; do not resume the closed P0–P7 optimization campaign. First preserve the measured source/binary/evidence identities and verify the actual upstream parent/ref. Implement U0 as one small portable fixture-resolution/test slice without changing GPU arithmetic. Then integrate the pinned upstream main in a separate commit, resolving shared Qwen/ownership/build changes explicitly and running the bounded native regression checks. Keep the full current functionality, the exact model/FP16/FP32/FP8 contracts, and all open default-reference/state gates. Prepare a clean upstream contribution surface without personal paths or local agent-policy replacement. Proceed to the limited F1/F2 performance candidates, one at a time, only with their stated same-input/state and lifetime tests. Do not rerun old rejected experiments, invent passing evidence, silently promote the verifier default, or replace implementation work with another large receipt campaign. Report separate status for integration, qualification, and measured speed.


---

# Evidence appendix

Source ranges below refer to the uploaded snapshot, not to a post-merge tree. Code excerpts are unchanged unless a section explicitly says it is a reviewer summary. Paths are repository-relative. Source hashes identify the complete file, not just the excerpt.

## E01. Independent audit and test scope


Payload verification: 9,520; Git blobs: 8,880; failures: 0. No GPU or native build was executed here.

| Host suite | Tests | Result |
|---|---:|---|
| `tests/scripts/test_exl3_attention_capture.py` | 6 | PASS |
| `tests/scripts/test_exl3_attention_rope.py` | 3 | PASS |
| `tests/scripts/test_exl3_grouped_capture.py` | 6 | PASS |
| `tests/scripts/test_exl3_projection_capture.py` | 5 | PASS |
| `tests/scripts/test_exl3_projection_compare.py` | 4 | PASS |
| `tests/scripts/test_exl3_projection_extract.py` | 6 | PASS |
| `tests/scripts/test_exl3_runtime_layout.py` | 10 | PASS |
| `tests/scripts/test_exl3_target_capture.py` | 22 | PASS |
| `tests/tools/test_exl3_verify_batch.py` | 2 | PASS |
| `tests/tools/test_exl3_verify_capture.py` | 2 | PASS |
| `tests/tools/test_exl3_verify_family.py` | 3 | PASS |
| `tests/scripts/test_b70_inventory.py` | 22 | PASS |
| `tests/scripts/test_b70_provider_trace.py` | — | Not qualified here: required probe executable was not supplied |
| `tests/tools/test_b70_resource_receipt.py` | 2 | PASS |

The unsuccessful provider-probe invocation is retained as a review-preparation error. It is not reported as a product regression or a passed test. Source CMake passes the required executable; see E05.

## E02. Raw serving recomputation


The companion `recompute_performance.py` checks request counts, observation increments, finish reasons, native cycle sums and repeated IDs/non-timing cycles. It does not execute inference.

```json
{
  "medians": {
    "c1_native": {
      "e2e_s": 22.086893151,
      "ttft_ms": 2330.7287359999996,
      "common_decode_tps": 51.78137474248805,
      "peak_decode_overlap": [
        1,
        1,
        1
      ]
    },
    "c1_original": {
      "e2e_s": 17.970133999013342,
      "ttft_ms": 1951.9909050140996,
      "common_decode_tps": 63.86340557303269,
      "peak_decode_overlap": [
        1,
        1,
        1
      ]
    },
    "c4_native": {
      "e2e_s": 44.467988566,
      "ttft_ms": 7835.841082999999,
      "common_decode_tps": 129.5248016944058,
      "peak_decode_overlap": [
        4,
        4,
        4
      ]
    },
    "c4_original": {
      "e2e_s": 76.08384238899453,
      "ttft_ms": 29524.101639995934,
      "common_decode_tps": null,
      "peak_decode_overlap": [
        1,
        1,
        1
      ]
    }
  },
  "derived": {
    "native_rate_gap_pct": -18.918550807203538,
    "native_speedup_needed_pct": 23.332773397055064,
    "native_post_first_yield_per_cycle": 2.779891304347826,
    "hypothetical_ms_per_cycle_at_original_tps_same_native_yield": 43.528704418507836,
    "warning": "Hypothetical budget is not a measured original GPU-cycle latency; different continuations and reused original trials."
  }
}
```

Raw source identities:

| Group | File in archive | SHA-256 |
|---|---|---|
| c1_native | `evidence/performance-p7-gdn-slm-typed-v1/c1-default-v1.json` | `19904bc6b0e9eda705534ecd7cae9368ea1a7608df456d359f36c76d7effae27` |
| c1_native | `evidence/performance-p7-candidate-matrix-v3/c1-native-r2-v1.json` | `d075ffce9b8bf549dc690b346c9646908161bbd67fee9fc381f58bbd68622c87` |
| c1_native | `evidence/performance-p7-candidate-matrix-v3/c1-native-r3-v1.json` | `d8076fe972508d8bad9d725cf7313f8288dfac7e75072b4ecfc57d2d44435137` |
| c1_original | `evidence/performance-p7-candidate-matrix-v2/c1-original-r1-v1.json` | `a22b26daa128dc8f83d0f095636ff31a9a7eb7ed8dfe00c8a7d23a268d23cba1` |
| c1_original | `evidence/performance-p7-candidate-matrix-v2/c1-original-r2-v1.json` | `5fd09c4dc6bf243599bfec98c5dcb47e02bbf842ed2882303a777504e22d57b7` |
| c1_original | `evidence/performance-p7-candidate-matrix-v2/c1-original-r3-v1.json` | `bb3076c90f963a81bfb9bfbbeb2e9714b9d2242eb5e138cdff95940b780f34fa` |
| c4_native | `evidence/performance-p7-candidate-matrix-v3/c4-native-r1-v1.json` | `fd6f5d350803694883f4017f2123b0d10a406874ec6c4b004420d65231d11448` |
| c4_native | `evidence/performance-p7-candidate-matrix-v3/c4-native-r2-v1.json` | `6465cd40e38a135410ed59710c9027ec6138439425fb2d6033b267103148def2` |
| c4_native | `evidence/performance-p7-candidate-matrix-v3/c4-native-r3-v1.json` | `482da46d8f1ef4f6971794f641723caba17ff6d66a23e783613b2621235c87af` |
| c4_original | `evidence/performance-p7-candidate-matrix-v2/c4-original-r1-v1.json` | `15a22185d15aac975eaf242581cef7a1c4254cd3d4d9684ac4987b5bca3fa4c3` |
| c4_original | `evidence/performance-p7-candidate-matrix-v2/c4-original-r2-v1.json` | `ab1da7071570e5b80a52dc349fb1b95abb8fd25aecca64234f01ac0e4245f2bc` |
| c4_original | `evidence/performance-p7-candidate-matrix-v2/c4-original-r3-v1.json` | `fee43441f41c37612143e690e537a696078c80c85addbd7e34374ad3455fc3f1` |

## E03. Current campaign closure and capability ledger



**Source:** `PRO_REVIEW_REPORT.txt`, lines 1–32. Full-file SHA-256: `95110eb3c2176b16401f1f35f73c686d08a32e83f2626a628256828a0a3b6d5b`.

```text
1: B70 EXL3 NATIVE PERFORMANCE: FINAL PRO REVIEW HANDOFF
2: Date: 2026-10-05
3: Product code: 0ea34de4e005688ebc80f28137c93c83b9c461c3
4: Branch: b70-gptq-int4. The archive MANIFEST identifies the later pushed documentation/closure commit.
5: Previous performance-plan baseline: 7536aededc049b2157f12eadb0f4cc6bbf29b7b7.
6: 
7: DECISION AND REVIEW REQUEST
8: 
9: The developer explicitly ended this P0-P7 optimization period after the final typed-memory speculative GDN SLM optimization. No additional optimization is authorized under the old plan. Final checks and this handoff are complete; the next work should follow a new Pro-reviewed implementation plan. Closure means the agreed work period is finished, not that every original qualification gate or the Python-performance objective was achieved. Preserve the unresolved gates below. Do not restart the old plan from P0.
10: 
11: Please produce a bounded implementation-first plan to close the remaining real prefill and decode gap. Inspect the included code and original sources. Distinguish GPU arithmetic/memory/compiled-route costs from host submission, state snapshots, proposal yield and scheduling. Deliver a practical first code assignment, a ranked short sequence, exact correctness/state/lifetime invariants and a stop rule for each attempt. Do not propose more receipt-only loops or repeat rejected variants without new causal evidence.
12: 
13: FIXED CONTRACT AND ENVIRONMENT
14: 
15: Intel Arc Pro B70, one32GB card and32GB host, existing180W; no driver/power/library changes. Exclusive native/original GPU execution, production b70-qwen38-vllm.service stopped. Only the tested instance uses the card. No resource leases or GPU lock files. One assistant, current branch, focused checks, regular commit/push.
16: 
17: Checkpoint turboderp/Qwen3.8-27B-exl3 revision113cf7ab958054860e43fb7f3063b1af19171095. Body4.00bpw; full248320-token6bpw EXL3 target head; exact65536-token block-aligned compact draft/global map; MTP3. Large-M is rotated EXL3 W8A8, not GPTQ W4A8. FP16 materialization, FP32 recurrent state, FP8 KV, page1600,262144 configured context objective. Native target/draft/MTP, graphs, prefix caching and lifecycle are implemented. Full262K generation is not tested; the final matrix uses4K/32K and practical retrievals actually60872 tokens long. No whole-model FP16 expansion or reference tensors in product inference.
18: 
19: Builder image sha256:ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca.
20: Original image sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918.
21: Compiler icpx2026.1.1, IGC2.41.5; driver1.17.39758+10. GPU workers24GiB cgroup/no-swap/CPU2. Startup/loading excluded consistently, O32request warmup then cold-prefix reset. Keep native tracking and original Torch allocation/reservation accounting separate.
22: 
23: PLAN EXECUTION
24: 
25: P0: bounded measurement/readiness identities and resource/lifecycle evidence delivered.
26: P1: page1600 packed native verifier, C1Q4 then boundaries/Q2/Q3/Q5/C4/ragged and owner tests delivered with measured serving gain. P1 automatic/default promotion and integrated original full-state qualification remain open. Benchmarks explicitly set attention=auto and VT_XPU_XE2_VERIFY=1; do not claim arbitrary ordinary defaults have the same performance. PartialC2/C3 original-FP16 verifier handling was subsequently delivered and measured.
27: P2: wider group-correct W8A8 panels, bounded workspace/planning and real grouped/operator/model proof delivered. Preserved int8/S32/scales and FP16 boundaries; previous128 route retained as control.
28: P3: typed FP16 SiLU and later mathematical lookup paths delivered with exact finite/Inf/signed-zero outputs. Two-NaN payload differences remain explicitly scoped, not universal NaN-bit equality.
29: P4: checked preparation, immutable model map certificates and completion-owned reuse delivered. Dynamic/public guards preserved. Cached SmallM model maps remove repeated eager map readbacks; no unsafe pointer-only certificate.
30: P5: two runnable private-register GDN variants rejected (C1 slower; second also uniformC4 slower). This register package is closed, not a missing optimization. Later profile-driven coalesced SLM recurrence and the final typed-memory extension succeeded; keep the rejected package identities.
31: P6: same-input donor comparison on13real weight families/27shapes, native/donor packed/intermediate bytes and largely equivalent DPAS ISA. Weighted core costs were within the declared10% band. One isolated FA-outM4 gap remained unexplained; absolute SmallM cost is not proof of a missing donor port. No new global tile search.
32: P7: additional exact norms/gated SiLU, map/guard reuse, no-MTP aligned capacity, mixed token/output views, partial verifier and coalesced speculative GDN retained only after scoped proof/gain checks. Final current practical/draft/target, native primary three-repeat matrix and one-shot long/mixed/no-MTP/prefix evidence refreshed. Developer now closes the optimization period, with incomplete numerical/product qualification visible.
```

**Source:** `PRO_REVIEW_REPORT.txt`, lines 60–74. Full-file SHA-256: `95110eb3c2176b16401f1f35f73c686d08a32e83f2626a628256828a0a3b6d5b`.

```text
60: LAST APPROVED OPTIMIZATION: TYPED MEMORY IN EXISTING SPECULATIVE GDN SLM
61: 
62: src/vt/xpu/xpu_gdn.cpp::GdnSpecDecodeKernel: direct typed F16/F32 loads/stores under existing contiguous F16 output/qkv, F32g/beta/state, C<=4/Q<=4/Hk16/Hv48/D128 contract. Same WG64/SG32,32768B SLM xor layout, ascending-K scalar arithmetic, full FP32 per-token snapshots and ownership/barriers. No new workspace, state precision change, private128-state retry or host polling. Default1; VT_XPU_GDN_SPEC_SLM_TYPED=0 returns to the prior dynamic SLM branch. Unsupported contracts fall back; existing WG override wins.
63: 
64: Focused exact snapshots/alias/metadata/F16graph case495/495; complete operator321/321; final default route probe38/38; default C1model1900/1900 and lifecycle483/483. Full outputs/state arrays exact, not only selected FP16 final rows. Complete-operator medians improved29.46%C1Q4/14.89%C4Q4/23.09%ragged. One paired native unprofiled pilot gave C1decode+3.36% and C4shortP4096/O256 common decode+8.66%; E2E-2.88%/-3.86%. All IDs/non-timecycles exact and device memory unchanged. These are pilot native A/B gains, not refreshed Python parity percentages.
65: 
66: Separate identical native C1P4096/O32 profile: target decode queue48.954836->46.685796ms, draft5.223043->5.238103ms, graph-validation readback0.028711->0.028199ms. Graph-internal kernels are not exposed by this receipt. Different/nested clocks must not be summed. This points to remaining target GPU work; it does not justify removing a required wait or claiming the residual gap is host-only. Profile throughput is not used as the serving score.
67: 
68: NUMERICAL/STATE/QUALIFICATION LEDGER
69: 
70: Final practical screen392/392, semantic8/8: four code functions20executed cases, two actual60872-token retrievals, two structured/tool tasks. Prompts/IDs/text/finishes/MTP counters exact previous candidate. Sequential tasks with C4 capacity; not concurrent C4 semantic coverage.
71: Final isolated compact draft131644/131644: all655360hidden and65536compact logits plus independent identical-input head outputs exact after widening originalF16 toF32. Not every evolving MTP iteration.
72: Final P128+D64 full248320 head: actual exit1,808assertions/804pass/4fail against UNCHANGED frozen default reference. D27TV0.0305754/KL0.00212581 andD29TV0.212752/KL0.163144, top10minimum9. All65complete rows (16140800F32values) byte-exact prior and separately controlled deterministic-BA original. No default-reference promotion or gate relaxation.
73: Two held-out eager no-MTP P128/D1 prefixes: complete48GDN Conv/SSM and16initialized FP8attention-KV layers byte-exact after the prior signed-zero repair. Initially a single layer43negative-zero key byte differed; actual norm/RoPE preimages identified the cause, then generic signed-zero-preserving F32 products repaired it. Each prefix5711/5711. Nonzero arithmetic unchanged; bounded complete-preamble cost about+0.6%. This is scoped eager hybrid proof, not integrated original MTP/C4/long-context state qualification. The typed GDN same-input full snapshot proof and native model continuity are separate evidence.
74: P1 automatic/default admission, independently justified default-reference contract, integrated original MTP/C4/long states, full262K generation and stochastic-RNG equivalence remain unqualified. Do not convert test assertion fractions into model accuracy. Do not claim the C++ engine is more accurate than Python from these observations.
```

## E04. Personal-root footprint and history handling

This is a reviewer summary of a scoped literal-root scan, not an exhaustive secret audit. The original personal root is intentionally not reproduced in this public-plan appendix. Original raw manifests must remain unchanged in the developer evidence archive.
| Repository path | Matching occurrences | Full-file SHA-256 |
|---|---:|---|
| `docs/B70-EXL3-Migration-and-Parity-Plan.md` | 5 | `f82186533265b69e204019ca3b2d2693090d635e7fd0e602c41f51b03db5eb87` |
| `docs/b70-exl3/EXECUTION_MAP.md` | 149 | `ab0362dc8db47dbf6405b4e18832d18a92370758246f50f87a6e7348caee2d9f` |
| `docs/b70-exl3/PRO_PERFORMANCE_REVIEW_2026-10-05.txt` | 3 | `95110eb3c2176b16401f1f35f73c686d08a32e83f2626a628256828a0a3b6d5b` |
| `docs/b70-exl3/RECOVERY_STATUS.json` | 193 | `b109b010cdeff5ab583afdf5b0580153fbbdaf2cf71561bc1051eb01da9f72b1` |
| `docs/b70-exl3/REFERENCE_MANIFEST.json` | 1907 | `1d5342776f167e6afe1378303c2ef37313497d4cbdc872089019969972891a19` |
| `docs/bench-evidence/b70-gptq-fp8-parity-20260927/p4_native_long.md` | 1 | `7a315d204905d9d6935e04eed8724dc903ab8bf147774d85916afa5ef278bb11` |
| `docs/bench-evidence/b70-gptq-fp8-parity-20260927/p4_real_layer0.md` | 1 | `f4165d992d6cea7a850bf7ce9294d4c12b4e741eb9984655b14c9408e260346f` |
| `docs/bench-evidence/b70-gptq-fp8-parity-20260927/profile_p2_diagnostics.json` | 1 | `1029beac0e16cd6d9ac842007f85e5f88004524ca518f0bd5b94d73df912e416` |
| `docs/bench-evidence/b70-gptq-fp8-parity-20260927/profile_p2_diagnostics.md` | 1 | `af9c57b66e28cc8c2b65bf586c68bf4712dbd0653520c5872f99cbc47a43dd4c` |
| `docs/bench-evidence/b70-gptq-fp8-parity-20260927/python_harness_checks.json` | 1 | `4ba91b7739cc1edc1ce6701f46d616876823f47b419d1ba57949d1e6df1c5d7c` |
| `docs/bench-evidence/b70-gptq-fp8-parity-20260927/quality_baseline.md` | 1 | `463ed7dd77cdffb23f171c7e26f91b745ce3e9c4c5431f0bff10a12393c32431` |
| `docs/bench-evidence/b70-gptq-fp8-parity-20260927/w1_graph_case5_isolation.md` | 1 | `057dd75eb43f7a8ea021d3f467d7730b78b2894128940339264ec7b0f04e6165` |
| `docs/bench-evidence/b70-gptq-fp8-parity-20260927/w2_topk20_ab.md` | 1 | `fd699fa5f648592faf39c58aefce54a80e6ea74e36f290cbc2fddee21d53f913` |
| `tools/exl3_reference/README.md` | 6 | `09200407281c54ccf28d22e3a414a00e737adbc10ae453ef9a66872bf94c78e0` |

The upstream branch comparison also shows the local root `AGENTS.md` replacement (+60/-678 lines in the API response). Exclude the local policy change from the contribution branch; do not erase its development history. Public artifact indexes need logical names/digests rather than personal storage locations.

## E05. Missing fixtures are not successful tests



**Source:** `tests/vt/test_xpu_exl3_smallm.cpp`, lines 79–87. Full-file SHA-256: `abf46d197bd02ee824bcb0948167f152e95c99f8d839383641112862486a7eda`.

```cpp
79: 
80: TEST_CASE("XPU P6 SmallM: same original packed weights and complete operator") {
81:   const char* fixture_path = std::getenv("VT_B70_SMALLM_FIXTURE");
82:   const char* report_path = std::getenv("VT_B70_SMALLM_REPORT");
83:   if (!fixture_path || !report_path) { MESSAGE("Set P6 original fixture/report paths"); return; }
84:   std::ifstream stream(report_path); REQUIRE(stream.good());
85:   nlohmann::json original; stream >> original;
86:   REQUIRE(original.at("schema") == "b70-exl3-p6-smallm-original-v1");
87:   REQUIRE_FALSE(original.at("profiled").get<bool>());
```

**Source:** `tests/vt/test_xpu_exl3_smallm.cpp`, lines 544–551. Full-file SHA-256: `abf46d197bd02ee824bcb0948167f152e95c99f8d839383641112862486a7eda`.

```cpp
544: 
545: TEST_CASE("XPU EXL3 producer SmallM: real M1/M4 captures and first/last head blocks") {
546:   const char* env = std::getenv("VT_B70_EXL3_S0B_FIXTURES");
547:   if (!env) {
548:     std::cerr << "SKIP: set VT_B70_EXL3_S0B_FIXTURES to pinned local captures.\n";
549:     std::exit(77);
550:   }
551:   Queue gpu(vt::DeviceType::kXPU);
```

**Source:** `tests/CMakeLists.txt`, lines 22–33. Full-file SHA-256: `ed0efbc53821d484b06f31f83e90ab49687209e2fe1c6b3b6bed59a05d6ff90a`.

```cmake
22: add_executable(b70_provider_trace_probe
23:   vt/b70_provider_trace_probe.cpp
24:   ${CMAKE_SOURCE_DIR}/src/vt/op_provider.cpp
25:   ${CMAKE_SOURCE_DIR}/src/vt/backend.cpp
26:   ${CMAKE_SOURCE_DIR}/src/vt/dtype.cpp
27:   ${CMAKE_SOURCE_DIR}/src/vt/cpu/cpu_backend.cpp)
28: target_include_directories(b70_provider_trace_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
29: target_include_directories(b70_provider_trace_probe SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
30: vllm_cpp_set_warnings(b70_provider_trace_probe)
31: add_test(NAME test_b70_provider_trace COMMAND ${Python3_EXECUTABLE}
32:   "${CMAKE_SOURCE_DIR}/tests/scripts/test_b70_provider_trace.py" $<TARGET_FILE:b70_provider_trace_probe>)
33: if(VLLM_CPP_XPU)
```

**Source:** `tests/CMakeLists.txt`, lines 69–85. Full-file SHA-256: `ed0efbc53821d484b06f31f83e90ab49687209e2fe1c6b3b6bed59a05d6ff90a`.

```cmake
69:   else()
70:     target_link_libraries(${name} PRIVATE vllm::vllm vllm_test_main)
71:   endif()
72:   vllm_cpp_set_warnings(${name})
73:   add_test(NAME ${name} COMMAND ${name})
74:   # A gate that cannot run must not report success. doctest exits 0 after a
75:   # TEST_CASE returns early, printing "assertions: 0 | 0 passed | 0 failed" and
76:   # "Status: SUCCESS!" — indistinguishable, in a log or a `&&` chain, from a gate
77:   # that loaded a model and matched an oracle (issue #463; multimodal-speed.md
78:   # §17.7 hit exactly this). A test whose preconditions are absent exits 77
79:   # instead and CTest reports it **Skipped**. Inert for any test that never
80:   # returns 77, which today is all but test_voxtral_e2e.
81:   set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
82: endfunction()
83: 
84: vllm_cpp_add_test(test_exl3_w8a8_panel_plan vt/test_exl3_w8a8_panel_plan.cpp)
85: if(VLLM_CPP_XPU)
```

## E06. Build/dependency boundaries and retained math flags



**Source:** `CMakeLists.txt`, lines 1825–1853. Full-file SHA-256: `dd61c1735685e55d57180a5c98654201e9231e902d1b3fac835839641fc3f00c`.

```cmake
1825:   unset(CMAKE_REQUIRED_FLAGS)
1826:   if(NOT VLLM_CPP_SYCL_COMPILER_WORKS)
1827:     message(FATAL_ERROR "VLLM_CPP_XPU=ON requires a SYCL compiler/runtime. Configure with -DCMAKE_CXX_COMPILER=icpx.")
1828:   endif()
1829:   add_library(vllm_xpu_backend OBJECT src/vt/xpu/xpu_backend.cpp)
1830:   add_library(vllm_xpu OBJECT
1831:     src/vt/xpu/xpu_elementwise.cpp src/vt/xpu/xpu_norm.cpp
1832:     src/vt/xpu/xpu_sampling.cpp src/vt/xpu/xpu_ops.cpp src/vt/xpu/xpu_exl3.cpp
1833:     src/vt/xpu/xpu_gdn.cpp src/vt/xpu/xpu_attention.cpp src/vt/xpu/xpu_exl3_prefill.cpp
1834:     src/vt/xpu/xpu_exl3_smallm.cpp
1835:     src/vt/xpu/xpu_exl3_w8a8.cpp
1836:     src/vt/xpu/xpu_gdn_chunked.cpp src/vt/xpu/xpu_attention_split.cpp
1837:     src/vt/xpu/xpu_attention_prefill.cpp)
1838:   # The pinned producer builds these ESIMD kernels with -ffast-math. Scope it
1839:   # to this port; unrelated native model/operators retain their own numerics.
1840:   set_source_files_properties(src/vt/xpu/xpu_exl3_smallm.cpp
1841:     src/vt/xpu/xpu_exl3_w8a8.cpp PROPERTIES
1842:     COMPILE_OPTIONS "-ffast-math")
1843:   if(VLLM_CPP_XPU_XE2_PREFILL OR VLLM_CPP_XPU_XE2_GDN)
1844:     if(NOT EXISTS "${VLLM_CPP_SYCL_TLA_DIR}/include/cutlass/cutlass.h")
1845:       message(FATAL_ERROR "VLLM_CPP_SYCL_TLA_DIR must name the pinned SYCL-TLA checkout")
1846:     endif()
1847:     execute_process(COMMAND git -c "safe.directory=${VLLM_CPP_SYCL_TLA_DIR}"
1848:       -C "${VLLM_CPP_SYCL_TLA_DIR}" rev-parse HEAD
1849:       OUTPUT_VARIABLE _vt_sycl_tla_commit OUTPUT_STRIP_TRAILING_WHITESPACE
1850:       ERROR_QUIET)
1851:     if(NOT _vt_sycl_tla_commit STREQUAL "87f6850680a580654b9ea2c80dbc01aeb36ad231")
1852:       message(FATAL_ERROR "Xe2 kernels require SYCL-TLA 87f6850680a580654b9ea2c80dbc01aeb36ad231")
1853:     endif()
```

**Source:** `CMakeLists.txt`, lines 1920–1939. Full-file SHA-256: `dd61c1735685e55d57180a5c98654201e9231e902d1b3fac835839641fc3f00c`.

```cmake
1920:     target_compile_definitions(vllm_xpu PRIVATE VLLM_CPP_XPU_XE2_GDN)
1921:     target_sources(vllm PRIVATE $<TARGET_OBJECTS:vllm_xpu_xe2_gdn>)
1922:   endif()
1923:   if(VLLM_CPP_XPU_GPTQ4)
1924:     find_package(dnnl 3.13.0 EXACT CONFIG REQUIRED)
1925:     target_sources(vllm_xpu PRIVATE src/vt/xpu/xpu_gptq4.cpp
1926:       src/vt/xpu/xpu_exl3_attention.cpp)
1927:     target_compile_definitions(vllm_xpu PRIVATE VLLM_CPP_XPU_GPTQ4)
1928:     target_compile_definitions(vllm_xpu_backend PRIVATE VLLM_CPP_XPU_GPTQ4)
1929:     target_link_libraries(vllm_xpu PRIVATE DNNL::dnnl)
1930:     target_link_libraries(vllm PUBLIC DNNL::dnnl)
1931:   endif()
1932:   foreach(xpu_target vllm_xpu vllm_xpu_backend)
1933:     set_target_properties(${xpu_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
1934:     target_include_directories(${xpu_target} PRIVATE include src)
1935:     target_include_directories(${xpu_target} SYSTEM PRIVATE third_party)
1936:     target_compile_options(${xpu_target} PRIVATE -fsycl)
1937:     if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
1938:       # Device division/sqrt approximations are a separate icpx default from
1939:       # host fast math and can flip BF16 rounding at layer boundaries.
```

**Source:** `third_party/exl3xpu/README.md`, lines 1–7. Full-file SHA-256: `ce802483b7648fcd7e73b488870d72a4d3fa7940d5c22793604bb0aa3fb4d23c`.

```text
1: EXL3 ESIMD kernels from 0xSero/exl3xpu, commit
2: `c59d9442aba8610188837e37724600f1517d7335`, MIT license (see LICENSE).
3: 
4: `exl3_esimd.h` is an unchanged copy of `csrc/exl3_esimd.h`, SHA-256
5: `aabdb13eddbcf7387dac2716b26e1005259a0a658d4b7d8b0e6b338499d0fdc6`.
6: Native VT launch/allocation wrappers live in `src/vt/xpu/xpu_exl3_smallm.cpp`;
7: they do not link Torch or use its tensor/stream wrappers.
```

Check the retained MIT, BSD and Apache license/attribution files and update the top-level notice/provenance inventory. This review does not declare a license violation or make a legal clearance determination.

## E07. Verifier admission is implemented but ordinary-default promotion remains open



**Source:** `src/vt/xpu/xpu_attention_verify_xe2.cpp`, lines 20–31. Full-file SHA-256: `622e11847c0e328b40997e7e0bd32f312371dc120e2bed35245d5ef7cfee7961`.

```cpp
20: int64_t PagedAttentionXe2VerifyQueryLength(int64_t tokens, int64_t requests,
21:                                          const int32_t* host_offsets) {
22:   if (requests == 1) return tokens >= 2 && tokens <= 5 ? tokens : 0;
23:   // Original C2/C3/C4 Q4 fixtures qualify uniform partial batches as requests
24:   // finish. Do not infer uniformity from shape or the speculative routing hint.
25:   // Missing/ragged host metadata
26:   // leaves the generic route available without an extra D2H readback.
27:   if (requests < 2 || requests > 4 || tokens != requests * 4 || !host_offsets) return 0;
28:   for (int64_t r = 0; r <= requests; ++r)
29:     if (host_offsets[r] != r * 4) return 0;
30:   return 4;
31: }
```

**Source:** `src/vt/xpu/xpu_attention_verify_xe2.cpp`, lines 52–87. Full-file SHA-256: `622e11847c0e328b40997e7e0bd32f312371dc120e2bed35245d5ef7cfee7961`.

```cpp
52:       args.kv_cache_dtype != Fp8KVCacheDataType::kFp8E4M3 ||
53:       key_cache.shape[0] != value_cache.shape[0] ||
54:       (page != 1600 && page != 1664) || value_cache.shape[1] != page ||
55:       key_cache.shape[2] != 4 || value_cache.shape[2] != 4 ||
56:       key_cache.shape[3] != 256 || value_cache.shape[3] != 256 ||
57:       (key_cache.stride[1] != 4 * 256 && key_cache.stride[1] != 4 * 512) ||
58:       value_cache.stride[1] != key_cache.stride[1] ||
59:       key_cache.stride[2] != key_cache.stride[1] / 4 ||
60:       value_cache.stride[2] != key_cache.stride[2] ||
61:       key_cache.stride[3] != 1 || value_cache.stride[3] != 1 ||
62:       key_cache.stride[0] != value_cache.stride[0] ||
63:       key_cache.stride[0] % key_cache.stride[1] != 0 ||
64:       key_cache.stride[0] / key_cache.stride[1] < page ||
65:       block_table.rank != 2 || block_table.dtype != DType::kI32 ||
66:       block_table.shape[0] != requests || block_table.stride[1] != 1 ||
67:       (requests > 1 && block_table.stride[0] != block_table.shape[1]) ||
68:       seq_lens.rank != 1 || seq_lens.dtype != DType::kI32 ||
69:       query_start_loc.rank != 1 ||
70:       query_start_loc.dtype != DType::kI32 || query_start_loc.Numel() != requests + 1 ||
71:       args.max_seq_len < rows ||
72:       block_table.shape[1] < (args.max_seq_len + page - 1) / page ||
73:       !args.causal || args.window_size || args.logits_soft_cap != 0 ||
74:       args.k_scale != 1.0f || args.v_scale != 1.0f ||
75:       args.scale != 1.0f / 16 ||
76:       (reinterpret_cast<uintptr_t>(query.data) & 15) ||
77:       (reinterpret_cast<uintptr_t>(out.data) & 15) ||
78:       (reinterpret_cast<uintptr_t>(key_cache.data) & 15) ||
79:       (reinterpret_cast<uintptr_t>(value_cache.data) & 15) ||
80:       !device.has(sycl::aspect::ext_intel_device_id) ||
81:       device.get_info<sycl::ext::intel::info::device::device_id>() != 57891 ||
82:       std::string_view(__VERSION__) !=
83:           "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)" ||
84:       device.get_info<sycl::info::device::driver_version>() != "1.17.39758+10" ||
85:       key_cache.shape[0] > std::numeric_limits<int>::max() /
86:           (key_cache.stride[0] / key_cache.stride[1]))
87:     return false;
```

**Source:** `src/vt/xpu/xpu_attention.cpp`, lines 439–460. Full-file SHA-256: `c11e4c3e1a9c60d42e4e5ea5783ce2803abee81b0198bcc3481bf79a82eaf0f5`.

```cpp
439:   const auto* offsets = static_cast<const int32_t*>(query_start_loc.data);
440:   const auto* table = static_cast<const int32_t*>(block_table.data);
441:   const char* setting = std::getenv("VT_XPU_ATTENTION");
442:   const std::string_view mode = setting ? setting : "auto";
443:   int64_t packed_verify_rows = 0;
444: #ifdef VLLM_CPP_XPU_XE2_VERIFY
445:   const char* verify_setting = std::getenv("VT_XPU_XE2_VERIFY");
446:   const bool packed_requested = mode == "verify" ||
447:       (mode == "auto" && verify_setting && std::string_view(verify_setting) == "1");
448:   if (requests > 1 && packed_requested)
449:     packed_verify_rows = PagedAttentionXe2VerifyQueryLength(
450:         tokens, requests, args.query_start_loc_host);
451: #endif
452:   const int64_t max_length = args.max_seq_len;
453:   CheckDeviceMetadata(q, [=] {
454:     if (offsets[0] != 0 || offsets[requests] != tokens) return false;
455:     for (int64_t r = 0; r < requests; ++r) {
456:       const int64_t first = offsets[r], end = offsets[r + 1], length = lengths[r];
457:       if (first < 0 || end < first || end > tokens) return false;
458:       // A uniform host hint selects the packed batched layout. Prove it against
459:       // fresh device offsets inside the existing eager/graph metadata check;
460:       // never trust total-token division or add a per-layer host readback.
```

## E08. Speculative GDN producer/consumer boundary



**Source:** `src/vllm/model_executor/models/qwen3_5.cpp`, lines 5946–5993. Full-file SHA-256: `2c09c07dd603a45d6ae51ef84bccb48cc045173d2be829b5d84785f8e010588b`.

```cpp
5946:   // intermediates are allocated. This mirrors vLLM v0.25.0
5947:   // _forward_core_decode_non_spec:1644-1695.
5948:   DBuf dcore(d, outdt, {T, Hv, Dv});
5949:   // GDN-MOE-BF16-OUT (#1168): read off the tensors, not off GdnOutDType, so a
5950:   // gate entering through ModelRegistry::Forward observes what this layer RAN.
5951:   // This is the ONLY recording site. The mixed spec+non-spec batch returns into
5952:   // GdnBlockPagedMixedSpec above and never reaches it, so a mixed step leaves
5953:   // the record untouched rather than stale-free — see the header's limits.
5954:   RecordGdnOutActivationDTypes(dcore.t().dtype, z.dtype);
5955:   const float scale = 1.0F / std::sqrt(SizeF(Dk));
5956:   if (packed_decode) {
5957:     Tensor gidx = SubView(sdi.gdn_state_idx.t(), 0, nd);
5958:     Tensor ssm_cache = state.ssm_state;
5959:     vt::GdnPackedDecode(d.q, dcore.t(), dconv.t(), araw, braw,
5960:                         a_log_dev, dt_bias_dev, ssm_cache, gidx,
5961:                         vt::GdnArgs{scale});
5962:   } else {
5963:     // Coupled bf16 (VT_GDN_BF16, default ON): matmul-input activations q/k/v
5964:     // are bf16 (native WMMA fragments + halved traffic); g/beta and recurrence
5965:     // arithmetic stay f32.
5966:     const DType actdt = GdnActivationDType(d);
5967:     DBuf vf(d, actdt, {T, Hv, Dv});
5968:     DBuf dg(d, DType::kF32, {T, Hv});
5969:     DBuf dbeta(d, DType::kF32, {T, Hv});
5970:     DBuf dql2(d, actdt, {T, Hk, Dk});
5971:     DBuf dkl2(d, actdt, {T, Hk, Dk});
5972:     if (fp16_producer_prefill || GlueFuseEnabled()) {
5973:       vt::GdnPostConv(d.q, dql2.t(), dkl2.t(), vf.t(), dg.t(), dbeta.t(),
5974:                       dconv.t(), araw, braw, a_log_dev, dt_bias_dev,
5975:                       vt::GdnPostConvArgs{1e-6F, fp16_producer_prefill});
5976:       DumpGdnStage(d, "mixed", mixed);
5977:       DumpGdnStage(d, "postconv_q", dql2.t());
5978:       DumpGdnStage(d, "postconv_k", dkl2.t());
5979:       DumpGdnStage(d, "postconv_v", vf.t());
5980:       DumpGdnStage(d, "postconv_beta", dbeta.t());
5981:       if (const char* td = std::getenv("VT_DUMP_TRUST")) {
5982:         vt::tenstorrent::TrustDump(d.q, td, "pc_q", dql2.t());
5983:         vt::tenstorrent::TrustDump(d.q, td, "pc_v", vf.t());
5984:       }
5985:     } else {
5986:       DBuf qf(d, actdt, {T, Hk, Dk});
5987:       DBuf kf(d, actdt, {T, Hk, Dk});
5988:       Tensor q2 = Reshape(qf.t(), {T, key_dim});
5989:       Tensor k2 = Reshape(kf.t(), {T, key_dim});
5990:       Tensor v2 = Reshape(vf.t(), {T, value_dim});
5991:       vt::GdnConvSplit(d.q, q2, k2, v2, dconv.t());
5992:       vt::GdnGBeta(d.q, dg.t(), dbeta.t(), araw, braw, a_log_dev,
5993:                    dt_bias_dev);
```

**Source:** `src/vllm/model_executor/models/qwen3_5.cpp`, lines 6003–6021. Full-file SHA-256: `2c09c07dd603a45d6ae51ef84bccb48cc045173d2be829b5d84785f8e010588b`.

```cpp
6003:       // ── PURE spec recurrence (SPEC-MTP I5a; qwen_gdn_linear_attn.py:
6004:       // 1455-1475). Every token is a spec token, in batch order (identity gather
6005:       // in a pure batch), so the whole [T,...] post-conv output feeds one
6006:       // GdnSpecDecode over the per-request k+1 state slots. The kernel selects
6007:       // each request's initial state from column num_accepted-1 and snapshots
6008:       // the post-token state of timestep t into column t — the SSM rollback.
6009:       // Persistent per-step device tensors (uploaded once, shared by all GDN
6010:       // layers). state_indices is the flat [num_spec_decodes*num_cols] upload
6011:       // viewed row-major as [num_spec_decodes, num_cols]. ──
6012:       Tensor ssm_cache = state.ssm_state;  // full [slots, Hv, Dv, Dk] cache
6013:       Tensor spec_idx_2d =
6014:           Reshape(sdi.gdn_spec_state_idx.t(), {ns, sdi.gdn_spec_num_cols});
6015:       vt::GdnSpecDecode(d.q, dcore.t(), dql2.t(), dkl2.t(), vf.t(), dg.t(),
6016:                         dbeta.t(), ssm_cache, sdi.gdn_spec_qsl.t(), spec_idx_2d,
6017:                         sdi.gdn_num_accepted.t(), vt::GdnArgs{scale});
6018:     } else {
6019:     // Recurrence — decode segment first (leading nd_tok tokens), then prefill.
6020:     if (nd > 0) {
6021:       // Decode recurrence IN PLACE on the persistent ssm_state at each sequence's
```

**Source:** `src/vt/xpu/xpu_gdn.cpp`, lines 223–266. Full-file SHA-256: `a53a6501eff9084aec297b674fb37ed2b92e467b566f2930910d67a514a7b6ba`.

```cpp
223: void GdnPostConvKernel(Queue& q, Tensor& qo, Tensor& ko, Tensor& vo, Tensor& go, Tensor& bo,
224:                         const Tensor& conv, const Tensor& araw, const Tensor& braw,
225:                         const Tensor& alog, const Tensor& bias, const GdnPostConvArgs& args) {
226:   TraceXpuOp(OpId::kGdnPostConv, q, {&qo, &ko, &vo, &go, &bo, &conv, &araw, &braw, &alog, &bias});
227:   const int64_t tokens = conv.shape[0], hk = qo.shape[1], dk = qo.shape[2];
228:   const int64_t hv = vo.shape[1], dv = vo.shape[2], keys = hk * dk, values = hv * dv;
229:   const View src(conv), qs(qo), ks(ko), vs(vo), gs(go), bs(bo), a(araw), b(braw), al(alog), dt(bias);
230:   const float eps = args.eps;
231:   if (args.xpu_fp16_prefill) {
232:     // The pinned _xpu_C Conv binary uses SIMD32 (its launch lambda does not
233:     // inherit the source functor's SIMD16 annotation). Each Q/K head occupies
234:     // one subgroup with four contiguous FP32 features per lane. The separate
235:     // libgdn_attn recurrence kernels do use SIMD16; do not conflate the two.
236:     const auto event = NativeQueue(q).submit([&](sycl::handler& handler) {
237:       handler.parallel_for(sycl::nd_range<1>(tokens * (hk + hv) * 64, 64),
238:           [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
239:         const int64_t group = item.get_group(0);
240:         const int64_t token = group / (hk + hv), head = group % (hk + hv);
241:         const int lane = item.get_local_id(0);
242:         const int64_t base = token * (2 * keys + values);
243:         if (head < hk) {
244:           const bool is_q = lane < 32;
245:           const int feature = (lane % 32) * 4;
246:           float value[4];
247:           for (int i = 0; i < 4; ++i) {
248:             value[i] = Load(src, base + (is_q ? 0 : keys) + head * dk + feature + i);
249:           }
250:           // Both pinned Q/K quadratsums square feature 1 before contracting
251:           // feature 0, then 2 and 3. The independent FP32 producer replay
252:           // distinguishes this from starting with feature 0.
253:           float square = value[1] * value[1];
254:           square = sycl::fma(value[0], value[0], square);
255:           square = sycl::fma(value[2], value[2], square);
256:           square = sycl::fma(value[3], value[3], square);
257:           const auto sg = item.get_sub_group();
258:           const float total = sycl::reduce_over_group(sg, square, sycl::plus<float>());
259:           // Match the pinned Conv binary, not just its source expression:
260:           // Q uses SQRT(sum+eps), SQRT(D), MUL, then INV; K uses RSQRT.
261:           // The producer executes these as subgroup-uniform scalar math.
262:           // On B70, varying RSQRT differs even for identical input bits.
263:           // Compute on the leader and broadcast to preserve its rounding.
264:           float inv = 0;
265:           if (sg.get_local_linear_id() == 0)
266:             inv = is_q
```

F1 is a new, conditional fusion/typed-producer hypothesis around this boundary, not a claim that a matching donor fusion has been shown to execute on XPU. Preserve the original FP16 materialization and reduction semantics.

## E09. Checked W8A8 preparation and remaining publication copies



**Source:** `src/vt/xpu/xpu_exl3_w8a8.cpp`, lines 221–278. Full-file SHA-256: `e1e16eff8173892fdb97904aec577a9fd997d1d4f14f1e72c25574f2c87ef4fa`.

```cpp
221:   const auto& panels = args.model_map ? args.model_map->Panels(args.w8a8_panel_columns)
222:                                     : checked_panels;
223:   auto& backend = GetBackend(q.device);
224:   const auto* sv_bits = static_cast<const uint16_t*>(svh.data);
225: 
226:   auto* bytes = static_cast<uint8_t*>(workspace.data);
227:   auto* xq = reinterpret_cast<int8_t*>(bytes + plan.activation_offset);
228:   auto* sx = reinterpret_cast<float*>(bytes + plan.row_scale_offset);
229:   auto* y = reinterpret_cast<sycl::half*>(bytes + plan.intermediate_offset);
230:   auto* sw = reinterpret_cast<float*>(bytes + plan.weight_scale_offset);
231:   // Both GPU checks use spare words in the existing 64-byte scale region.
232:   // Read their results together before output/panel/activation writes. This
233:   // avoids a separate metadata allocation, readback and retirement drain.
234:   auto* valid = reinterpret_cast<uint32_t*>(sw + 1);
235:   NativeQueue(q).single_task([=] { valid[0] = 1; valid[1] = 1; });
236:   Launch(q, n / 128, 8, ValidateOutputScale{sv_bits, n, valid + 1},
237:          "exl3_w8a8_validate_svh");
238:   ::exl3::HadInQ8Kernel<sycl::half> input{
239:       static_cast<const sycl::half*>(in.data), static_cast<const sycl::half*>(suh.data),
240:       xq, sx, m, k, groups, k, plan.padded_rows};
241:   const auto check_status = [&] {
242:     std::array<uint32_t, 2> finite{};
243:     backend.Copy(q, finite.data(), valid, sizeof(finite));
244:     backend.Synchronize(q);
245:     VT_CHECK(finite[1], "EXL3 W8A8 requires finite svh");
246:     VT_CHECK(finite[0], "EXL3 W8A8 requires finite inputs and finite FP16 transformed rows");
247:   };
248:   const char* prepare_setting = std::getenv("VT_XPU_W8A8_PREPARE");
249:   bool prepared = false;
250:   // Exact checked preparation is the default;0 restores the original route.
251:   // Unsupported K or insufficient private budget keeps the existing checks.
252:   if ((!prepare_setting || std::string_view(prepare_setting) == "1") && k / 128 <= 144) {
253:     const size_t qbytes = size_t(groups) * plan.padded_rows * k;
254:     const size_t sbytes = size_t(groups) * plan.padded_rows * sizeof(float);
255:     prepared = WithExl3W8A8Preparation(q, qbytes + sbytes, [&](void* storage) {
256:       auto candidate = input;
257:       candidate.xq = static_cast<int8_t*>(storage);
258:       candidate.sx = reinterpret_cast<float*>(static_cast<uint8_t*>(storage) + qbytes);
259:       NativeQueue(q).memset(candidate.xq, 0, qbytes);
260:       NativeQueue(q).parallel_for(sycl::range<1>(size_t(groups) * plan.padded_rows),
261:           [=](sycl::id<1> i) { candidate.sx[i[0]] = 1.0f; });
262:       if (k / 128 <= 48)
263:         Launch(q, int64_t(groups) * m * 8, 8, PrepareRows<8, 6>{candidate, valid},
264:                "exl3_w8a8_prepare_checked");
265:       else
266:         Launch(q, int64_t(groups) * m * 16, 16, PrepareRows<16, 9>{candidate, valid},
267:                "exl3_w8a8_prepare_checked");
268:       check_status();
269:       // Public preparation stays untouched on either validation failure.
270:       RecordGraphWrite(q, workspace.data, Span(workspace));
271:       const auto qe = NativeQueue(q).memcpy(xq, candidate.xq, qbytes);
272:       const auto se = NativeQueue(q).memcpy(sx, candidate.sx, sbytes);
273:       RecordProfileEvent(q, "exl3_w8a8_prepared_commit", qe);
274:       RecordProfileEvent(q, "exl3_w8a8_prepared_commit", se);
275:     });
276:   }
277:   if (!prepared) {
278:     Launch(q, int64_t(groups) * m * (k / 128), 8, ValidateRows{input, valid},
```

**Source:** `src/vt/xpu/xpu_exl3_w8a8.cpp`, lines 297–324. Full-file SHA-256: `e1e16eff8173892fdb97904aec577a9fd997d1d4f14f1e72c25574f2c87ef4fa`.

```cpp
297:     } else if (k / 128 <= 144) {
298:       ::exl3::HadInQ8WgKernel<sycl::half, 16, 9> had{
299:           input.x, input.suh, xq, sx, m, k, groups, k, plan.padded_rows};
300:       Launch(q, int64_t(groups) * m * 16, 16, had, "exl3_w8a8_input_quantize");
301:     } else {
302:       Launch(q, int64_t(groups) * m, 8, input, "exl3_w8a8_input_quantize");
303:     }
304:   }
305: 
306:   const auto weight_scale = Tensor::Contiguous(sw, DType::kF32, q.device, {1});
307:   for (const auto& part : panels) {
308:     if (args.bits == 4) Reconstruct<4>(q, tr, panel, k, n, part.first_column, part.columns);
309:     else Reconstruct<6>(q, tr, panel, k, n, part.first_column, part.columns);
310:     const int group = part.source_group;
311:     const auto a = Tensor::Contiguous(xq + size_t(group) * plan.padded_rows * k,
312:         DType::kI8, q.device, {plan.padded_rows, k});
313:     const auto scales = Tensor::Contiguous(sx + size_t(group) * plan.padded_rows,
314:         DType::kF32, q.device, {plan.padded_rows});
315:     auto dst = Tensor::Contiguous(y + part.first_column, DType::kF16, q.device,
316:                                   {plan.padded_rows, part.columns});
317:     dst.stride[0] = n;
318:     // Reconstruction writes a compact K*width prefix, including short tails.
319:     const auto weight_view = Tensor::Contiguous(panel.data, DType::kI8, q.device,
320:                                                 {k, part.columns});
321:     Exl3W8A8Matmul(q, dst, a, weight_view, scales, weight_scale);
322:   }
323:   // oneDNN rounds to F16 before the output Hadamard, as the production donor
324:   // does. The fallback I32 HadOutQ8 recipe is a different arithmetic route.
```

## E10. Current successful typed-SLM recurrence: preserve rather than repeat



**Source:** `src/vt/xpu/xpu_gdn.cpp`, lines 582–631. Full-file SHA-256: `a53a6501eff9084aec297b674fb37ed2b92e467b566f2930910d67a514a7b6ba`.

```cpp
582:     bool slm_selected = slm_mode == "1" && requests <= 4 && cols <= 4 && hk == 16 && hv == 48 &&
583:         dk == 128 && dv == 128 && qi.dtype == DType::kF16 && ki.dtype == DType::kF16 &&
584:         vi.dtype == DType::kF16 && g.dtype == DType::kF32 && beta.dtype == DType::kF32 &&
585:         std::getenv("VT_XPU_GDN_SPEC_WG") == nullptr;
586:     if (slm_selected) {
587:       const auto device = NativeQueue(q).get_device();
588:       const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
589:       slm_selected = device.get_info<sycl::info::device::local_mem_size>() >= 32768 &&
590:           device.get_info<sycl::info::device::max_work_group_size>() >= 64 &&
591:           std::find(sizes.begin(), sizes.end(), 32) != sizes.end();
592:     }
593:     if (slm_selected) {
594:       const char* typed_setting = std::getenv("VT_XPU_GDN_SPEC_SLM_TYPED");
595:       const std::string_view typed_mode = typed_setting ? typed_setting : "1";
596:       VT_CHECK(typed_mode == "0" || typed_mode == "1", "Invalid VT_XPU_GDN_SPEC_SLM_TYPED");
597:       const bool typed_selected = typed_mode == "1" && target.dtype == DType::kF16 &&
598:           target.IsContiguous() && qi.IsContiguous() && ki.IsContiguous() &&
599:           vi.IsContiguous() && g.IsContiguous() && beta.IsContiguous();
600:       // Each WG owns 64 independent value rows within one head/request.
601:       // Contiguous transfers use [K,value xor low K bits] SLM so both transfer
602:       // and value-row accesses spread over the local addresses. Each
603:       // value owner retains the original ascending-K scalar accumulation.
604:       // No private128-element state, parallel-K reduction or precision change.
605:       constexpr int values = 64, keys = 128;
606:       // Typed memory removes dynamic View dtype branches from the inner-K
607:       // loops. Preserve the same SLM layout and ascending-K arithmetic.
608:       const auto* query_half = static_cast<const sycl::half*>(qi.data);
609:       const auto* key_half = static_cast<const sycl::half*>(ki.data);
610:       const auto* value_half = static_cast<const sycl::half*>(vi.data);
611:       const auto* gates_float = static_cast<const float*>(g.data);
612:       const auto* beta_float = static_cast<const float*>(beta.data);
613:       auto* output_half = static_cast<sycl::half*>(target.data);
614:       const auto launch = [&]<bool Typed>() {
615:         return NativeQueue(q).submit([&](sycl::handler& h) {
616:         sycl::local_accessor<float, 1> s(sycl::range<1>(values * keys), h);
617:         h.parallel_for(sycl::nd_range<1>(items, values),
618:             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
619: #pragma clang fp contract(off)
620:           const int64_t index = item.get_group_linear_id() * values;
621:           const int lane = item.get_local_linear_id();
622:           const int64_t request = index / (hv * dv), head = (index / dv) % hv;
623:           const int64_t first_value = index % dv, value = first_value + lane;
624:           const int64_t first = offsets[request], last = offsets[request + 1];
625:           const int32_t initial = ids[request * cols + nat[request] - 1];
626:           const int64_t out_channel = head * dv + value;
627:           const auto write = [&](int64_t pos, float x) {
628:             if constexpr (Typed) output_half[pos] = sycl::half(x);
629:             else Store(dst, pos, x);
630:           };
631:           if (initial < 0) {
```

The full state update, snapshot order, barriers and alias validation are part of the contract even where not reproduced in this excerpt. Do not infer that removing a condition is safe from this shortened listing.

## E11. Draft-head row selection is already fixed; residual synchronization needs ownership



**Source:** `src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp`, lines 68–90. Full-file SHA-256: `85608acea736cf73d09d9fafd00ebef99ad5d46b4534db47d253b3b09589e8ab`.

```cpp
68:   vllm::Qwen3_5MTPHiddenStates hidden = draft.ForwardPaged(
69:       spi.input_ids, positions32, target_hidden, target_attn_meta, draft_kv, queue);
70:   PrefillOutcome out;
71:   // ── Greedy draft pick over each request's last (sampled) row
72:   // (spec_decode/speculator.py:276-280). ──────────────────────────────────────
73:   out.sampled_rows.assign(
74:       spi.last_token_indices.begin(),
75:       spi.last_token_indices.begin() + static_cast<size_t>(num_reqs));
76:   // Keep all forward rows for draft KV/feedback, but apply the head only to the
77:   // sampled rows, as the pinned producer does before sample_draft.
78:   const auto selected = draft.GatherHiddenRows(hidden.tensor, out.sampled_rows, queue);
79:   const auto logits = draft.ComputeLogits(selected.tensor, queue);
80:   VT_CHECK(logits.on_device() && logits.rows == num_reqs,
81:            "MtpProposePrefill: unexpected selected draft logits shape");
82:   out.draft_tokens = draft.SelectDraftTokens(logits, queue);
83: 
84:   // :346 — the positions of those same rows. The decode half advances from them.
85:   // The k=1 caller drops them, and dropping a host vector costs nothing.
86:   out.positions.resize(static_cast<size_t>(num_reqs));
87:   for (int64_t r = 0; r < num_reqs; ++r) {
88:     out.positions[static_cast<size_t>(r)] =
89:         positions32[static_cast<size_t>(out.sampled_rows[
90:             static_cast<size_t>(r)])];
```

**Source:** `src/vllm/model_executor/models/qwen3_5.cpp`, lines 10472–10518. Full-file SHA-256: `2c09c07dd603a45d6ae51ef84bccb48cc045173d2be829b5d84785f8e010588b`.

```cpp
10472: std::vector<int32_t> Qwen3_5MTPModel::SelectDraftTokens(
10473:     const ForwardLogits& logits, vt::Queue& queue) const {
10474:   VT_CHECK(logits.on_device() && logits.rows > 0 &&
10475:                logits.device_tensor.device == queue.device,
10476:            "MTP selection: expected owned device logits on the queue");
10477:   Dev device{vt::GetBackend(queue.device.type), queue};
10478:   if (weights_->IsExl3() && queue.device.type == vt::DeviceType::kXPU) {
10479:     const auto& compact = weights_->draft_head_exl3;
10480:     VT_CHECK(!compact.Empty() && logits.vocab == 65536,
10481:              "EXL3 MTP selection: expected compact logits and validated token map");
10482:     Tensor map = ResidentWeight(device, compact.token_ids);
10483:     DBuf chosen(device, DType::kI32, {logits.rows});
10484:     vt::MappedGreedyArgmax(queue, chosen.t(), logits.device_tensor, map, compact.target_vocab);
10485:     std::vector<int32_t> ids(static_cast<size_t>(logits.rows));
10486:     chosen.Download(device, ids.data());  // synchronizes before releasing consumers
10487:     for (const auto id : ids)
10488:       VT_CHECK(id >= 0 && id < compact.target_vocab,
10489:                "EXL3 MTP selection: invalid map or nonfinite draft logits");
10490:     return ids;
10491:   }
10492:   DBuf chosen(device, DType::kI64, {logits.rows});
10493:   vt::GreedyArgmax(queue, chosen.t(), logits.device_tensor);
10494:   std::vector<int64_t> wide(static_cast<size_t>(logits.rows));
10495:   chosen.Download(device, wide.data());
10496:   return std::vector<int32_t>(wide.begin(), wide.end());
10497: }
10498: 
10499: std::vector<float> Qwen3_5MTPModel::ForwardLogitsHost(
10500:     const std::vector<int32_t>& input_ids,
10501:     const std::vector<int32_t>& positions,
10502:     const vt::Tensor& target_hidden_states, vt::Queue& queue,
10503:     int64_t spec_step_idx) const {
10504:   Qwen3_5MTPHiddenStates hidden =
10505:       Forward(input_ids, positions, target_hidden_states, queue, spec_step_idx);
10506:   ForwardLogits logits = ComputeLogits(hidden.tensor, queue);
10507:   std::vector<float> host(static_cast<size_t>(logits.rows) * logits.vocab);
10508:   Backend& backend = vt::GetBackend(queue.device.type);
10509:   backend.Copy(queue, host.data(), logits.device_tensor.data,
10510:                host.size() * sizeof(float));
10511:   backend.Synchronize(queue);
10512:   return host;
10513: }
10514: 
10515: // Shared shape/count validation for the dense paged forward entry points (the
10516: // 27B analogue of CheckPagedForward). Same contract, over the dense weights.
10517: static void CheckDensePagedForward(const std::vector<int32_t>& token_ids,
10518:                                    const std::vector<int32_t>& positions,
```

Do not repropose the already implemented row selection or GPU argmax. A future all-device multistep draft must preserve metadata/RNG and buffer lifetimes; its potential is bounded by the measured draft share.

## E12. Do not recycle closed kernel experiments


`EXECUTION_MAP.md` occurrences of `0.32`: 2450.
`EXECUTION_MAP.md` occurrences of `within10`: .
`EXECUTION_MAP.md` occurrences of `P6`: 2612, 2629, 2643, 2645, 2655, 2697.

**Source:** `PRO_REVIEW_REPORT.txt`, lines 25–32. Full-file SHA-256: `95110eb3c2176b16401f1f35f73c686d08a32e83f2626a628256828a0a3b6d5b`.

```text
25: P0: bounded measurement/readiness identities and resource/lifecycle evidence delivered.
26: P1: page1600 packed native verifier, C1Q4 then boundaries/Q2/Q3/Q5/C4/ragged and owner tests delivered with measured serving gain. P1 automatic/default promotion and integrated original full-state qualification remain open. Benchmarks explicitly set attention=auto and VT_XPU_XE2_VERIFY=1; do not claim arbitrary ordinary defaults have the same performance. PartialC2/C3 original-FP16 verifier handling was subsequently delivered and measured.
27: P2: wider group-correct W8A8 panels, bounded workspace/planning and real grouped/operator/model proof delivered. Preserved int8/S32/scales and FP16 boundaries; previous128 route retained as control.
28: P3: typed FP16 SiLU and later mathematical lookup paths delivered with exact finite/Inf/signed-zero outputs. Two-NaN payload differences remain explicitly scoped, not universal NaN-bit equality.
29: P4: checked preparation, immutable model map certificates and completion-owned reuse delivered. Dynamic/public guards preserved. Cached SmallM model maps remove repeated eager map readbacks; no unsafe pointer-only certificate.
30: P5: two runnable private-register GDN variants rejected (C1 slower; second also uniformC4 slower). This register package is closed, not a missing optimization. Later profile-driven coalesced SLM recurrence and the final typed-memory extension succeeded; keep the rejected package identities.
31: P6: same-input donor comparison on13real weight families/27shapes, native/donor packed/intermediate bytes and largely equivalent DPAS ISA. Weighted core costs were within the declared10% band. One isolated FA-outM4 gap remained unexplained; absolute SmallM cost is not proof of a missing donor port. No new global tile search.
32: P7: additional exact norms/gated SiLU, map/guard reuse, no-MTP aligned capacity, mixed token/output views, partial verifier and coalesced speculative GDN retained only after scoped proof/gain checks. Final current practical/draft/target, native primary three-repeat matrix and one-shot long/mixed/no-MTP/prefix evidence refreshed. Developer now closes the optimization period, with incomplete numerical/product qualification visible.
```

A 1,024-column panel remains the declared default; the existing 2,048 comparison must be consulted before suggesting that variant again. The two rejected private-state GDN candidates and the 13-family/27-shape SmallM investigation are closed unless a genuinely new causal observation justifies reopening their exact scope.

## E13. Current W8A8 pool completion fences



**Source:** `src/vt/xpu/xpu_backend.cpp`, lines 930–986. Full-file SHA-256: `d6e4c9130e80aaacc4a07e940262d7e52a5704fd97a9910d7cf705a3979430fd`.

```cpp
930: }
931: namespace {
932: bool WithEagerW8A8Workspace(Queue& q, Workspace& workspace, size_t bytes,
933:                            const std::function<void(void*)>& launch, const char* wait_stage) {
934:   VT_CHECK(bytes > 0, "XPU W8A8 workspace must be nonempty");
935:   auto& native = NativeQueue(q);
936:   auto& c = GetContext(q.device.index);
937:   std::lock_guard execution(workspace.mutex);
938:   {
939:     std::lock_guard lock(c.mutex);
940:     // No W8A8 pool address enters a graph. Existing SmallM graph ownership is
941:     // independent, so eager growth cannot invalidate a captured panel.
942:     VT_CHECK(!c.recordings.count(&native), "XPU shared W8A8 workspace is eager-only");
943:     if (bytes > workspace.bytes) {
944:       if (bytes - workspace.bytes > c.budget - c.allocated - c.graph_bytes) return false;
945:       // Every previous lease completed before returning, including exceptions.
946:       // Free first to avoid retaining both the old and new high-water capacities.
947:       if (workspace.data) {
948:         sycl::free(workspace.data, c.context);
949:         c.allocations.erase(workspace.data);
950:         c.allocated -= workspace.bytes;
951:         workspace.data = nullptr; workspace.bytes = 0;
952:       }
953:       void* storage = sycl::aligned_alloc_device(64, bytes, c.device, c.context);
954:       VT_CHECK(storage != nullptr, "XPU shared W8A8 workspace allocation failed");
955:       try { c.allocations.emplace(storage, bytes); }
956:       catch (...) { sycl::free(storage, c.context); throw; }
957:       c.allocated += bytes;
958:       c.peak_allocated = std::max(c.peak_allocated, c.allocated);
959:       workspace.data = storage; workspace.bytes = bytes;
960:     }
961:   }
962:   try {
963:     launch(workspace.data);
964:     // Retain this completion fence until an asynchronous panel-lease protocol
965:     // can safely retire all validation, oneDNN and output-Hadamard consumers.
966:     const auto start = HostProfileEnabled() ? SteadyNs() : 0;
967:     native.wait_and_throw();
968:     if (start) RecordHostProfileSpan(q, wait_stage, start, SteadyNs());
969:   } catch (...) {
970:     try { native.wait_and_throw(); } catch (...) {}
971:     throw;
972:   }
973:   return true;
974: }
975: } // namespace
976: bool WithExl3W8A8Workspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch) {
977:   return WithEagerW8A8Workspace(q, GetContext(q.device.index).w8a8, bytes, launch,
978:                                 "workspace_wait_w8a8");
979: }
980: bool WithExl3W8A8Preparation(Queue& q, size_t bytes, const std::function<void(void*)>& launch) {
981:   if (bytes > 64 * 1024 * 1024) return false;
982:   return WithEagerW8A8Workspace(q, GetContext(q.device.index).w8a8_preparation, bytes, launch,
983:                                 "workspace_wait_w8a8_preparation");
984: }
985: bool WithGdnWorkspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch) {
986:   VT_CHECK(bytes > 0 && bytes <= 32 * 1024 * 1024, "XPU GDN workspace exceeds 32 MiB budget");
```

F2 proposes a new final-consumer event contract, not unconditional fence deletion. Cross-queue reuse, growth, free-on-error and oneDNN participation are mandatory checks. A wait span includes preceding work and is not an additive free speedup.

## E14. Live upstream observations (read-only, pinned)

These are reviewer summaries of GitHub API reads, not source-tree changes. No upstream branch was merged or rebased here.

- Parent repository: `mudler/vllm.cpp`.
- Observed main: `e67a071a649814e102a91145a0b1b424884f16da`.
- Base for comparison: `3803ad4a6b9974f4afd1b6c9ad3e333d5e9b6ff8`.
- Compare status: diverged; upstream-only 61, local-only 168; merge base `9d96b162c8c3dfa8f0143932962e445b68fe3a8b`.
- Key overlapping product paths are listed in section 4. The local Git diff is still needed for a complete dependency/size inventory and actual conflict resolution.

Pinned source links:

1. [Fork parent metadata](https://api.github.com/repos/Danmoreng/vllm.cpp)
2. [Exact comparison](https://api.github.com/repos/mudler/vllm.cpp/compare/3803ad4a6b9974f4afd1b6c9ad3e333d5e9b6ff8...e67a071a649814e102a91145a0b1b424884f16da)
3. [Contribution guide at the reviewed main](https://github.com/mudler/vllm.cpp/blob/e67a071a649814e102a91145a0b1b424884f16da/CONTRIBUTING.md)
4. [Current path-classification checker; old line budgets retired](https://github.com/mudler/vllm.cpp/blob/e67a071a649814e102a91145a0b1b424884f16da/scripts/check-pr-size.py)
5. [Relevant open tokenizer PR to recheck, not assumed merged](https://github.com/mudler/vllm.cpp/pull/3363)

The contribution guide treats vllm.cpp as a native C++ implementation with reference/performance goals and no Python inference dependency. It requires applicable tests, explicit unavailable gates and its current commit/landing procedure. Read the pinned guide rather than deriving obligations from an old copy or from the filename of a checker.

## E15. Release claims that still require evidence



**Source:** `PRO_REVIEW_REPORT.txt`, lines 68–74. Full-file SHA-256: `95110eb3c2176b16401f1f35f73c686d08a32e83f2626a628256828a0a3b6d5b`.

```text
68: NUMERICAL/STATE/QUALIFICATION LEDGER
69: 
70: Final practical screen392/392, semantic8/8: four code functions20executed cases, two actual60872-token retrievals, two structured/tool tasks. Prompts/IDs/text/finishes/MTP counters exact previous candidate. Sequential tasks with C4 capacity; not concurrent C4 semantic coverage.
71: Final isolated compact draft131644/131644: all655360hidden and65536compact logits plus independent identical-input head outputs exact after widening originalF16 toF32. Not every evolving MTP iteration.
72: Final P128+D64 full248320 head: actual exit1,808assertions/804pass/4fail against UNCHANGED frozen default reference. D27TV0.0305754/KL0.00212581 andD29TV0.212752/KL0.163144, top10minimum9. All65complete rows (16140800F32values) byte-exact prior and separately controlled deterministic-BA original. No default-reference promotion or gate relaxation.
73: Two held-out eager no-MTP P128/D1 prefixes: complete48GDN Conv/SSM and16initialized FP8attention-KV layers byte-exact after the prior signed-zero repair. Initially a single layer43negative-zero key byte differed; actual norm/RoPE preimages identified the cause, then generic signed-zero-preserving F32 products repaired it. Each prefix5711/5711. Nonzero arithmetic unchanged; bounded complete-preamble cost about+0.6%. This is scoped eager hybrid proof, not integrated original MTP/C4/long-context state qualification. The typed GDN same-input full snapshot proof and native model continuity are separate evidence.
74: P1 automatic/default admission, independently justified default-reference contract, integrated original MTP/C4/long states, full262K generation and stochastic-RNG equivalence remain unqualified. Do not convert test assertion fractions into model accuracy. Do not claim the C++ engine is more accurate than Python from these observations.
```

A maintained pending gate is not a reason to discard the implementation. It is a reason to scope the contribution honestly. The next iteration must preserve separate labels for local operator equality, controlled-reference equality, immutable default-reference results, autonomous generation, lifecycle and fully qualified serving.

**End of plan.**
