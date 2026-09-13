# ROCm host residency after upload — stop prefaulting, and release the source

Row: `MODEL-MM-QWEN4-EXP`
Issues: `ISSUE-LOCAL-01M2BZ5DZ2710201WMH9TNSVH3` (the defect),
`ISSUE-LOCAL-01M2BZ5QK4XRETK48CXKSHKRDW` (owed: chunked H2D)

## 1. The defect

On `strix:gpu0` (gfx1151) the 67.56 GiB `Qwen3.8-Flash-Next UD-IQ1_S` loads with
zero op refusals and then wedges forever inside `svm_range_set_attr`. The thread
is uninterruptible and `gpu_busy` reads 0. The host has about 31 GiB of RAM. We
call no SVM API ourselves: the staging path is `hipMalloc` plus
`hipMemcpyAsync` (`src/vt/rocm/rocm_backend.hip`).

The stall is a host-residency problem, and two sites compound to make it.

**One. Every borrowed GGUF span is synchronously prefaulted.**
`PrefaultBorrowedSpan` (`src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp`)
issues `madvise(MADV_WILLNEED)` and then reads one byte per page across the whole
span. The two keep-quant call sites pass `prefault = true` unconditionally. On
this checkpoint that faults in 65.488 GiB of file pages that each weight reads
exactly once, on a box with half that much RAM.

The prefault has a good reason on the CPU tier, and only there: a weight left
BORROWED in the mapping is not resident until first touch, and without the
prefault those faults land in the timed prefill. On a device that stages, the
weight is copied to the device once at load and the host pages are never read
again. There is nothing to keep warm.

**Two. Nothing releases the source pages after the device upload.**

THERE ARE TWO `ResidentWeight`s AND THIS ROW'S MODEL USES THE OTHER ONE. The
first draft of this spec named only the `ResidentWeight` in the unnamed namespace
of `src/vllm/model_executor/models/qwen3_5.cpp`, which is true of Qwen3.5 and
false of `qwen4_exp`. That function shadows the header one inside its own
translation unit, so it serves the Qwen3.5 dense weights and — through
`KqResidentSlice` and `KqGrouped` — the shared MoE seam's keep-quant expert
towers, which `qwen4_exp` does reach via `RunQwen4ExpMoeBlock` ->
`RunMoeBlock` -> `MoeBlock`. Everything else in `qwen4_exp` stages through
`dense_attn::ResidentWeight`
(`include/vllm/model_executor/models/dense_attn_block.h`): 12 call sites in
`qwen4_exp_forward.cpp`, 10 in `qwen4_exp_qsa_block.cpp`, 7 in
`qwen4_exp_ple_block.cpp`, 1 in `qwen4_exp_registry.cpp`, and not one reference
to the `qwen3_5.cpp` function anywhere in that model. Fixing one arm and not the
other was measured, not argued: with the release in `qwen3_5.cpp` alone, the load
peak on `strix:gpu0` fell from 27.67 GB to 8.18 GB and host `RssFile` still
climbed to 21.08 GB during the forward and stayed there.

BOTH arms have the same body and the same defect. Each stages the weight
with `Alloc` + `Copy` and then calls `AdoptDeviceBytesAsHost`, which returns
immediately for a GGUF borrow. `OwnedTensor::ReleaseHost()` likewise refuses to
`madvise` a BORROWED buffer, arguing that the pages are clean and file-backed and
the kernel can reclaim them unaided.

That argument holds against the page reclaimer and fails against the KFD. The
resident pages are still mapped into this process, and that is what the
accounting in `svm_range_set_attr` walks. Existing release machinery does not
reach this path either: `GgufFile::DropSpanResidency` is this tree's port of
llama.cpp's `unmap_fragment` and is called only for tensors the loader COPIED or
EXPANDED; `ReleaseDirectUploadSource` is inert because the GGUF loader never sets
`mmap_src`; and the bf16 MoE release loop in `qwen3_5.cpp` has no counterpart on
the keep-quant tower path, which is the entire 67.56 GiB.

## 2. #2511 falsified the premise the ROCm policy still states

`RocmPlatform::residency_policy()` (`src/vllm/platforms/rocm.cpp`) leaves
`release_host_weights_after_upload` false and its comment gives the reason:

> on a unified part (780M, Strix Halo) freeing the host copy after "upload"
> would free the ONLY copy

That is no longer true, and a stale comment is what kept 65 GiB pinned. Since
**#2511**, gfx1151 reports `pageableMemoryAccess = 0`. `HostMemoryIsDeviceAddressable`
therefore answers false (`src/vt/rocm/rocm_backend.hip`, `src/vllm/platforms/rocm.cpp`),
`ResidentWeight`'s host-alias arm is not taken (`qwen3_5.cpp`), and the staging
branch below it always makes a real second copy on the device. The host copy has
not been the only copy on this part since that change landed.

This spec records the reversal and repairs the comment. It does NOT flip
`release_host_weights_after_upload`: that flag is read only through
`ShouldReleaseHostWeights` / `ShouldInterleaveLoadStream`, both of which also
require `marlin_committed`, which is false on ROCm, so flipping it would change
no behaviour and would assert a pool/release measurement nobody has taken. The
release this spec adds is gated on the property that is actually load-bearing and
is checkable at the call site — the device cannot dereference host memory, and
the borrow is a re-faultable read-only file mapping — not on a policy bit whose
other consumers are dead here.

## 3. Oracle

`llama-cpp` pin `10bf611e5` (b10451), the recorded pin in
`.agents/oracles/llama-cpp.md`.

- `src/llama-model-loader.cpp:1567-1577` grows `mmap_used` ONLY for a tensor
  that lands in a HOST buffer. A tensor going to a device buffer takes
  `ggml_backend_tensor_set` and grows nothing.
- `src/llama-model-loader.cpp:1683-1694` then calls
  `unmap_fragment(0, mmap_used.first)`. For a fully offloaded model
  `mmap_used.first` is the whole mapping, so the whole mapping goes.
- `src/llama-mmap.cpp:492-507` shows `unmap_fragment` is a real `munmap`, not an
  advisory hint.
- `src/llama-mmap.cpp:449-461` is the prefetch: `MAP_POPULATE` plus an advisory
  `posix_madvise(POSIX_MADV_WILLNEED)`. There is **no synchronous touch loop**.

Peak host residency for a fully offloaded model is therefore O(one tensor)
during load and zero after it. Ours is O(whole model) for the process lifetime.

## 4. Scope — two fixes

### Fix 1 — release the source pages after a staging upload

In BOTH `ResidentWeight` staging arms — the one in `qwen3_5.cpp`'s unnamed
namespace and `dense_attn::ResidentWeight` in `dense_attn_block.h`, which is the
one `qwen4_exp` takes — once the device copy exists and has completed,
drop the resident interior pages of a BORROWED, file-backed source span when the
platform is NOT host-addressable. The borrow itself is untouched and stays a
valid, re-faultable `PROT_READ MAP_PRIVATE` view, so a later read re-faults from
the file and nothing observes a byte difference. This is `unmap_fragment`'s
intent expressed as `MADV_DONTNEED`, which is what this tree already uses for the
same job in `DropSpanResidency`, `ReleaseHost` and `AdoptDeviceBytesAsHost`.

Two things are load-bearing.

**It must be memoized, and #1299 is why.** `ResidentWeight` is called about 1,361
times per forward step on this checkpoint. A release that re-tests its condition
on every call would `MADV_DONTNEED` the very pages the GPU is about to read, on
every step, and the kernel would fault them straight back in. Correctness
survives that; throughput does not. The release therefore goes inside
`if (!w.d_dev)`, immediately before the `AdoptDeviceBytesAsHost` call and NOT
inside it — the same memo that made the aligned-borrow branch of
`MakeHostBytesDeviceAliasable` a repeat hazard when it had none. It is a separate
helper because `AdoptDeviceBytesAsHost` returns early for a GGUF borrow by
design, which is exactly the case this release exists for. The red test covers
the REPEAT call, not only the first, on both arms.

**The copy must have completed.** `RocmBackend::Copy` is `hipMemcpyAsync` on a
stream. Dropping the source pages while the copy may still be reading them is a
correctness bug, so the staging arm synchronizes the queue before the release.
It is behind the same `d_dev` memo, so it costs one synchronize per weight on
the first forward and nothing thereafter.

### Fix 2 — make the prefault device-aware

`prefault` is decided at the two keep-quant call sites in
`qwen3_5_gguf_weights.cpp` and passed as a literal `true`. Give it the same
device term `quant_repack` got in #2406: default OFF when the resolved device
cannot dereference host memory, because on that device the prefault reads the
whole tower off disk into pages exactly one `memcpy` then reads. An explicit
`VT_GGUF_PREFAULT=1` (or `vllm_cpp.mmap.prefault: true`) still wins, so the A/B
stays available in the same binary; `ResolveGgufPrefault` remains the sole reader
of that variable and the new helper only decides the DEFAULT.

### Out of scope

Chunked H2D through a pinned bounce buffer — llama.cpp's 4 x 64 MiB shape — is
NOT built here. See `## Deferred, with an issue`.

### Also in scope

The stale comment at `src/vllm/platforms/rocm.cpp` (§2).

## 5. Tests — red first

`tests/vllm/model_executor/test_resident_weight_host_addressable.cpp` already
carries the fake platform and fake backend this needs, and it enters through
`Qwen3_5EmbeddingTable`, the named production bridge over `ResidentWeight`. New
cases there:

1. **The pages actually go.** Build a real temporary file, `mmap` it
   `PROT_READ MAP_PRIVATE`, touch every page so `RssFile` in
   `/proc/self/status` grows by the span, borrow it into an `OwnedTensor`, and
   stage it through `Qwen3_5EmbeddingTable` on a platform whose
   `host_memory_is_device_addressable()` is false. Assert the resident set drops
   back. This is the assertion a counter cannot make: the release either unmaps
   the pages or it does not.
2. **Once, not 1,361 times.** Call the same bridge repeatedly on the same weight
   and assert the release instrument counts exactly one release. Mutating the
   memo away must fail this case.
3. **A host-addressable platform is unchanged.** The alias arm never reaches the
   release, and the existing cases in this file must stay green.
4. **The SECOND arm, which is the one this row's model takes.** Three more cases
   repeat 1-3 against `dense_attn::ResidentWeight`. They call that seam directly,
   as `test_resident_weight_f32_copy_retires.cpp:263` already does for
   `dense_attn::ResidentWeightF32` and for the same reason: the fake backend
   implements memory operations only and registers no `Embedding`, `MatmulBT` or
   `RmsNorm` for `kXPU`, so every `qwen4_exp` entry point above that seam refuses
   on a missing op before residency is asked about. The seam is production code
   in a production header, not a test hook; reachability is carried by the 31
   production call sites named in §1, and the mutation that convicts the wiring
   is deleting the `MaybeReleaseStagedBorrowSource` call from
   `dense_attn_block.h`.

For fix 2 the instrument already exists: `NoteGgufPrefaultedSpan` /
`GgufPrefaultSnapshot` in `include/vllm/config/weight_residency.h` count spans
actually prefaulted, and were added precisely because a prefault changes no byte
and a byte-transparency case cannot see it. A truth-table case over the new
device helper pins the default per device and pins that an explicit knob wins.

Every added assertion is mutation-proven for the release, the memo and the
platform term, on both arms. The synchronize and the BACKEND half of the
host-addressability pair are NOT convicted by this harness and are recorded
as an ungated guarantee rather than chased with a contorted test.

## 6. Gates

- Focused: `-tc=*release*`, `-tc=*prefault*`, `-tc=*DSA*`.
  `-tc=*resident*` was the first draft of this line and it SELECTED NOTHING:
  no case name in `test_resident_weight_host_addressable.cpp` contains the
  substring `resident` (`residency` does not, the `t` is missing), so doctest
  reported 0 cases, 0 assertions and `Status: SUCCESS!` — a third of the declared
  focused gate passing without measuring anything. Every declared selector's case
  and assertion counts are printed with the evidence, because a selector that
  matches nothing is indistinguishable from one that passes.
- Full ROCm cross-device suite: 60 cases / 84833 assertions, unchanged.
- `scripts/check-agent-record.py`, `scripts/check-commit-style.py`,
  `scripts/check-commit-trailers.py`, `scripts/check-pr-size.py`.
- The model gate: does the 67.56 GiB UD-IQ1_S at
  `/workspace/ckpt/qwen4exp-flash-next-iq1s` forward on `strix:gpu0` and produce
  a token? If it does, that is this row's G3, and load time, peak host and device
  memory, prefill and decode tok/s, recipe, revisions, model sha256, environment
  and contention are recorded with it. gfx1151 greedy decode fails about two runs
  in five with an illegal GPU memory access
  (`ISSUE-LOCAL-01M2BY2M2ATNVR3XQKV2DB1BJD`), so every measurement is repeated at
  least three times and reported as a spread. If it does not forward, NO number
  is recorded and the thread's `/proc/<tid>/stat` state and `wchan` are reported
  instead.

## 6a. The model gate: MEASURED, AND THE STALL SURVIVES BOTH FIXES

Measured 2026-09-13 on `strix:gpu0` (gfx1151, Radeon 8060S, ROCm 7.2.4 / HIP
7.2.53211, 30 GiB host, `mem_total_bytes` 33,270,497,280), under `rc` job
`cd38c438-e5a7-487f-9aab-27324f04f2d7`, box idle and exclusively leased. Built on
the worker from a clean clone of `969dd6f` with
`cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_HIP=ON`
`-DVLLM_CPP_HIP_ARCHITECTURES=gfx1151 -DROCM_PATH=/opt/rocm`, `ninja -j 6
vllm-cli`, 210 s. The artifact was ASSERTED HIP-linked before it was timed:
`ldd` shows `libamdhip64.so.7`, `libhsa-runtime64.so.1`, `libhipblaslt.so.1` and
`librocblas.so.5`, all from `/opt/rocm-7.2.4/lib`. Run through
`examples/vllm-cli --device auto --max-tokens 8 --temperature 0
--max-num-seqs 1` over
`/workspace/ckpt/qwen4exp-flash-next-iq1s/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf`
(shard 1 of 3; 10,946,624 + 49,990,818,368 + 22,544,696,352 bytes).
`VT_ROCM_MANAGED_ALLOC` unset.

**NO TOKEN, TWICE.** Each run was killed at a 1200 s deadline (`SIGKILL`, exit
137) having produced no output past the auto-fit line. Two runs, not three: the
result is the same failure in both and the axis is not a number.

| | run 1 | run 2 |
|---|---|---|
| peak `VmHWM` | 27,250,548 kB (25.99 GiB) | 27,320,420 kB (26.05 GiB) |
| peak `RssFile` | 21,207,224 kB (20.22 GiB) | 21,338,160 kB (20.35 GiB) |
| peak `RssAnon` | 5,821,836 kB (5.55 GiB) | 5,821,172 kB (5.55 GiB) |
| peak device memory | 31,878,860,800 B (29.69 GiB) | 31,880,183,808 B (29.69 GiB) |
| token | none, killed at 1200 s | none, killed at 1200 s |

**Device memory is no longer UNVERIFIED.** `rocm-smi` is not on `PATH` in the
leased container, so this is read from
`/sys/class/drm/card0/device/mem_info_vram_used`: 154,816,512 B at rest, climbing
to 31.88 GB of the board's 33.27 GB total. The earlier "717 MB of 103 GB" figure
does not describe this board and is withdrawn rather than carried forward.

**Where it blocks.** Sampling `/proc/<tid>/{stat,wchan}` every 6 s over both
runs, the uninterruptible thread is in `svm_range_set_attr` for 153 of 196
samples in run 1 and 148 of 198 in run 2, in `folio_wait_bit_common` for 41 and
47, and in `lock_mm_and_find_vma` for 3. So the KFD SVM path is still the
dominant blocker, and a second, smaller share is plain page-cache read wait —
the checkpoint is on a CIFS mount (`//192.168.68.102/Data`), which is its own
confound and is NOT separated here.

**THIS IS §7's FOURTH RISK, REALISED.** The release is wired into both staging
arms and gated, and the checkpoint still does not forward. Host `RssFile` during
the forward is 20.2-20.4 GiB, essentially unchanged from the 21.08 GB the
one-arm build showed, so releasing the spent source pages is not by itself
sufficient on this board. The pageable, file-backed source is therefore the next
suspect and `ISSUE-LOCAL-01M2BZ5QK4XRETK48CXKSHKRDW` (chunked H2D through a
pinned bounce buffer) is required, not optional. No throughput, latency or
prefill number is recorded, because no token was produced.

## 7. Risks

- **A released page that something still reads.** The borrow stays valid, so a
  read re-faults from the file and the bytes are identical. The cost of being
  wrong is a page fault, not a wrong token. This is the same contract
  `DropSpanResidency` has carried since it landed.
- **A repeat release on the hot path.** #1299's shape exactly. Held off by the
  `d_dev` memo and pinned by case 2 above.
- **A dropped page mid-DMA.** Held off by the synchronize, which is itself
  behind the memo.
- **The stall survives both fixes.** Then the pageable file-backed source is
  itself the trigger and the owed chunked-H2D change is required. That is a
  legitimate result and is reported with `wchan` evidence rather than papered
  over.

## Ungated guarantees

Two preconditions of `MaybeReleaseStagedBorrowSource` that this tree's harness
cannot convict. This section is deliberately NOT spelled `## Owed`: that heading
is the ownership surface for a ROWLESS issue under `.agents/issues/_owed`, and
`scripts/issue_records.py` refuses a row-owned issue that appears under it.
Nothing here is unreached code; both call sites are production and both are
gated. What is owed is an INSTRUMENT. The code is correct and the gap is in the instrument, so neither
test is contorted to manufacture a conviction. `ISSUE-LOCAL-01M2CCNA0S74WT5WBV50B3VD0W`
owns both.

- **The `backend.DeviceMemoryIsHostAddressable()` term is UNGATED.** Deleting it
  leaves 25/25 green with the binary proven changed. The fake backend in
  `test_resident_weight_host_addressable.cpp` answers `false` unconditionally, so
  only the platform half of the pair is ever exercised. The helper does refuse on
  either — that claim is about the code and is true — but only one arm is
  measured. Convicting the other needs a fake backend that answers `true` while
  the platform answers `false`, which is a combination no device in this fleet
  presents and which the file's registrar cannot hold alongside the existing one.
- **The `backend.Synchronize(queue)` call is UNGATED.** Deleting it also leaves
  25/25 green. `HostBackend::Copy` is a synchronous `memcpy` and the class does
  not override `Synchronize`, so no case in this tree can express a DMA still
  reading the source pages. Convicting it needs a backend whose `Copy` defers.

## Deferred, with an issue

- `ISSUE-LOCAL-01M2BZ5QK4XRETK48CXKSHKRDW` (row-owned, not started) covers chunked H2D through a pinned bounce
  buffer, llama.cpp's 4 x 64 MiB shape. Needed only if the stall survives fixes 1
  and 2. Medium-size and touches every staged weight on every backend, so it gets
  its own row, spec and measurement.

## 8. Stop conditions

- The release cannot be placed behind an existing memo without a new one: STOP
  and return `NEEDS_DECISION` rather than adding an unmemoized hot-path
  `madvise`.
- `strix:gpu0` is unreachable or the controller is down: the model gate is
  reported UNVERIFIED. It is never replaced by an `ssh` plus a file mutex the
  fleet cannot see.
