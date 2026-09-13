# ROCm chunked H2D through a bounded pinned bounce ring

Row: `MODEL-MM-QWEN4-EXP`
Issue: `ISSUE-LOCAL-01M2BZ5QK4XRETK48CXKSHKRDW` (row-owned, deferred by
`.agents/specs/rocm-host-residency-after-upload.md` § "Deferred, with an issue",
and made REQUIRED by that spec's §6a measurement)

## 1. The defect

`RocmBackend::Copy` is one call:

```cpp
void Copy(Queue& q, void* dst, const void* src, size_t bytes) override {
  Check(hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, AsStream(q)), "hipMemcpyAsync");
}
```

(`src/vt/rocm/rocm_backend.hip:278-280` at `98e2cd7da`.)

On `strix:gpu0` (gfx1151) that call is handed a **multi-GiB, pageable,
file-backed, CIFS-backed** source: a `PROT_READ MAP_PRIVATE` view of a GGUF
shard on `//192.168.68.102/Data`. To DMA out of a pageable range the ROCr
runtime has to make that range resident and describe it to the KFD, and
`svm_range_set_attr` is where that work lands.

`.agents/specs/rocm-host-residency-after-upload.md` §6a measured the
consequence. The 67.56 GiB `Qwen3.8-Flash-Next UD-IQ1_S` loads with zero op
refusals, stages 29.69 GiB onto the board, and then never produces a token: two
1200 s runs, killed at the deadline, with the compute thread uninterruptible in
`svm_range_set_attr` for 153 of 196 and 148 of 198 `wchan` samples. Releasing
the spent source pages after upload (that spec's fix 1) and stopping the
prefault (fix 2) both landed and neither moved it: host `RssFile` during the
forward is still 20.2-20.4 GiB. That spec's §7 fourth risk names this outcome
and names this change as the consequence.

So the remaining suspect is the SHAPE of the transfer, not the residency of
what it reads.

## 2. Oracle

`llama-cpp` pin `10bf611e5` (b10451), the recorded pin in
`.agents/oracles/llama-cpp.md`. The upload ring is in
`src/llama-model-loader.cpp`:

- `:1440` — `constexpr size_t n_buffers = 4;`
- `:1449` — `const size_t buffer_size = alignment != 1 ? 64 * 1024 * 1024 + 2 * alignment : 1 * 1024 * 1024;`
  under the comment "Buffer size: balance between memory usage and I/O
  efficiency / 64MB works well for NVMe drives".
- `:1496-1516` — allocates `n_buffers` buffers from
  `ggml_backend_dev_host_buffer_type(dev)` (pinned host memory) and one
  `ggml_backend_event_t` per buffer.
- `:1591-1642` — the loop. For each chunk: `ggml_backend_event_synchronize` on
  the buffer about to be reused, fill the buffer, `ggml_backend_tensor_set_async`
  from the pinned buffer to the device, `ggml_backend_event_record`, advance
  `buffer_idx` modulo `n_buffers`.
- `:1664` — the buffers are freed once, after the whole load.

Peak PINNED host residency is therefore `n_buffers * buffer_size` for any model
size. Ours is O(the largest single weight) of pageable range that the driver has
to pin and describe, on every weight.

**ONE HONEST DIFFERENCE, STATED RATHER THAN GLOSSED.** The oracle takes this
ring only when it is NOT using mmap (`:1456-1458`, `if (use_mmap ||
check_tensors) return nullptr;`); with mmap it calls `ggml_backend_tensor_set`
straight from the mapped pointer, which is the same single-shot pageable copy we
do today. So this is not a case where the oracle refuses the shape we ship. What
the oracle supplies is the SHAPE — ring size, buffer size, event discipline, and
the fact that a bounded pinned ring is the sanctioned way to feed a device out of
host storage the device cannot address. We apply it to a source the oracle
reaches by `read()` and we reach by `memcpy` out of a mapping. That adaptation is
the design, and it is why this is its own spec rather than a port.

## 3. Design

A ring of `N` fixed pinned buffers inside `RocmBackend`, lazily created on the
first copy that qualifies.

```
for (off = 0; off < bytes; off += n) {
  n = min(chunk, bytes - off);
  i = next slot
  if (slot[i] has an in-flight chunk) hipEventSynchronize(slot[i].event);
  memcpy(slot[i].host, (const char*)src + off, n);
  hipMemcpyAsync((char*)dst + off, slot[i].host, n, hipMemcpyHostToDevice, stream);
  hipEventRecord(slot[i].event, stream);
}
```

The call still returns with work in flight on the stream, so `Backend::Copy`'s
asynchronous contract is unchanged. What changes is that the SOURCE is fully
consumed by the time `Copy` returns, which is strictly stronger than today.

### 3a. N and the chunk size, justified against the oracle

- **`N = 4`.** The oracle's `n_buffers` at `llama-model-loader.cpp:1440`, taken
  rather than re-derived. Four slots keep one buffer filling while up to three
  drain, which is what a ring buys over a double buffer, and four is the number
  a runtime that loads models for a living ships.
- **`chunk = 64 MiB`.** The oracle's `buffer_size` at `:1449`. We drop its
  `+ 2 * alignment` term, deliberately: that term exists to absorb the
  `read_alignment()` padding of an `O_DIRECT`-style file read (`:1604-1631`
  computes `aligned_offset`, `read_start`, `read_end` and trims the padding back
  off). Our source is an in-memory mapping and our chunk boundaries are exact, so
  there is no padding to absorb and a padded buffer would only waste pinned
  bytes.
- **Total pinned residency: `4 * 64 MiB = 256 MiB`, for any model size.** That
  is the oracle's bound, unchanged.
- **Threshold = one chunk (64 MiB).** Below one chunk the ring degenerates to
  `memcpy` + `hipMemcpyAsync` with no overlap, no second slot ever used, and no
  residency benefit that the driver's own internal staging does not already
  give — it is strictly one extra copy of the bytes. A 4 KiB norm weight must
  not pay a bounce, and at this threshold it does not: it takes the existing
  single call, byte for byte. This boundary is not invented — it is where the
  oracle's own `while` loop stops having more than one iteration.

`VT_ROCM_PINNED_H2D_MIB` overrides the chunk in MiB and `0` disables the path
entirely, restoring `98e2cd7da` behaviour in the same binary. The A/B has to be
available in one binary because the model gate is a 20-minute run on a leased
box and rebuilding between arms is how a build difference gets to masquerade as
the effect.

### 3b. THE REACH, ENUMERATED BEFORE THE CODE

`Backend::Copy` is a shared seam. This change is confined to ONE override of it.
Files touched in `src/` and `include/`:
`src/vt/rocm/rocm_backend.hip` and the new HIP-free
`include/vt/rocm/rocm_pinned_h2d.h`. No other backend's `Copy`, no model, no
loader, no layer.

**Which backends change: ROCm, and only ROCm.** `cuda_backend.cu`,
`cpu_backend.cpp`, `metal_backend.mm`, `vulkan_backend.cpp`, the XPU backend and
the Tenstorrent backend are not edited and their `Copy` is byte-identical. CUDA
is deliberately excluded: `dgx`, `thor` and `orin` produce tokens today, so
nothing there is owed this, and widening to CUDA would put a new host
`cudaEventSynchronize` on a path four measured campaigns depend on. A CUDA arm
is a later row if a CUDA measurement ever asks for one.

**Which ROCm traffic changes.** The staged path is taken only when ALL FIVE
hold, and anything else takes the existing single call unchanged:

1. `bytes >= chunk` (64 MiB by default).
2. `dst` resolves to DEVICE memory. D2H and H2H never bounce, so the sampler's
   pinned-host download and every readback are untouched.
3. `src` is UNREGISTERED host memory. D2D never bounces. A source that is
   already pinned (`hipMemoryTypeHost`) never bounces, because it is already
   DMA-able and the ring exists only to create that property. A MANAGED source
   (`hipMemoryTypeManaged` / `Unified`) never bounces either, because it is
   already device-addressable — which is exactly the `VT_ROCM_MANAGED_ALLOC=1`
   configuration, so that knob's behaviour is unchanged.
4. The stream is NOT capturing a graph. `hipEventSynchronize` inside a capture
   region aborts the capture. The capture contract documented at
   `rocm_backend.hip:334-348` already forbids "host<->device blocking copies"
   inside the region, so a qualifying copy in there is already a contract
   violation — but it would previously have been a silent one and would now be a
   loud one, and changing WHICH failure a misuse produces is still a change. The
   guard keeps capture byte-identical.
5. The ring resolves (`VT_ROCM_PINNED_H2D_MIB != 0` and the pinned allocation
   succeeded). A failed `hipHostMalloc` falls back to the existing single call
   rather than failing the load: 256 MiB of pinned memory is not worth refusing
   a model over.

**So what actually takes it.** Every H2D weight upload of 64 MiB or more on an
AMD board. That is a WIDER set than the release in
`rocm-host-residency-after-upload.md` §4a, and saying so is the point of this
paragraph: that release needed `mmap_fd >= 0`, so it fired only on the GGUF
keep-quant borrow. This predicate asks only "unregistered host source, big
enough", so it ALSO fires on safetensors borrows and on any heap buffer a loader
hands to `Copy`. On ROCm the families that reach it are therefore §4a's five
(Qwen4-Exp, GLM-MoE-DSA, GLM5-Next, Muse-Glimmer, the Qwen3.5 DFlash draft
head's rebound embedding table) PLUS every safetensors model whose weights clear
64 MiB — gemma, phi, minicpm, olmo2, stablelm, commandr, deepseek_v2, dots3,
nemotron_h, qwen3_vl and the rest — PLUS the EXL3 device loader's trellis
uploads, which §4a excluded for want of `mmap_fd`. The behaviour those models
see is identical bytes, bounded pinned host residency, and one extra host
`memcpy` per 64 MiB.

**What is NOT covered by a test, and is recorded rather than claimed.** The only
family this change is MEASURED on is Qwen4-Exp, on one board. The device case in
§5 enters `RocmBackend::Copy` directly with a large pageable source, which is the
seam every one of those families reaches, so the CALL SITE is gated; no family
but Qwen4-Exp gets an end-to-end run here, for the same reason
`rocm-host-residency-after-upload.md` §5 gives — there is no harness that can
drive a second family's production entry point on a device.
`ISSUE-LOCAL-01M2CKN5516AKE7W2JVDV86Z8X` already owns that gap and this change
does not narrow it.

### 3c. Where the decision lives

The PURE parts — the chunk plan, the ring-reuse order, and the predicate that
decides whether to stage — go in `include/vt/rocm/rocm_pinned_h2d.h`, free of
HIP headers, and are table-tested in the ordinary CPU build. That mirrors
`include/vt/rocm/rocm_arch.h` and `ResolveMemoryPolicy` exactly, and for the
reason that header states about `CapabilityFromGcnArch`: the piece a wrong
answer breaks silently is the piece that must be gated on a runner with no AMD
GPU. `rocm_backend.hip` reads two `hipPointerGetAttributes`, one
`hipStreamIsCapturing`, and calls it.

### 3d. The ring is deliberately never freed

The pinned slots and events are allocated on first use and leaked at process
exit. `RocmBackend` instances live in a function-local `static
std::vector<std::unique_ptr<RocmBackend>>` in the registrar, so a destructor
would run during static destruction, and calling `hipHostFree` / `hipEventDestroy`
after the HIP runtime has begun tearing down is a hazard this file does not
have today (`exec_`, the `hipGraphExec_t`, is likewise never destroyed). 256 MiB
returned to the OS at `exit()` buys nothing and a teardown-order crash costs a
measurement. Stated here because "it leaks" must be a decision on the record,
not something a reader discovers.

## 4. Tests — red first

Commit order is the red: the test commit lands before the implementation commit,
so the failing run is reproducible by building the tree at the test commit.

**`tests/vt/test_rocm_pinned_h2d.cpp`** — new, UNCONDITIONAL (no HIP needed),
registered beside `test_rocm_arch`. Drives the pure header with fakes:

1. **The plan.** `bytes = 200 MiB`, `chunk = 64 MiB` gives 4 chunks of
   64/64/64/8 MiB, contiguous offsets summing to `bytes`, and no chunk larger
   than `chunk`. Mutating the chunk size to the whole buffer produces 1 chunk of
   200 MiB and fails this.
2. **The ring waits before it reuses.** With `N = 4` and 9 chunks, slot 0 is
   waited on before chunks 4 and 8 and NOT before chunk 0. The fake records the
   wait/stage/enqueue order; deleting the wait fails it.
3. **The bytes survive.** The fake stage/enqueue pair reassembles the output and
   it is `memcmp`-identical to the input, over a size that is NOT a multiple of
   the chunk.
4. **The predicate truth table.** Staged only for {unregistered host src, device
   dst, `bytes >= chunk`, not capturing, `chunk != 0`}. Every other row of the
   table is direct. Deleting the size term, the src-kind term, the dst term or
   the capture term each flips a row.

**`tests/vt/test_backend_cross_device.cpp`** — one new case, which is the
REACHABILITY conviction. It enters `RocmBackend::Copy` through
`vt::Backend&`, the production seam, with a 200 MiB pageable `std::vector`
source and a device destination, and asserts:

- the downloaded bytes are `memcmp`-identical to the source (bit-exactness is
  this file's declared bar for a pure copy path);
- `PinnedH2DSnapshot()` shows the copy took the ring — `staged_copies` grew by
  one and `chunks` by four — and that `max_chunk_bytes <= 64 MiB`, which is the
  bounded-host-residency assertion in the form this harness can actually make;
- a SMALL copy on the same backend grows `direct_copies` and not
  `staged_copies`, so the threshold is gated in the same case.

Deleting the staged branch from `rocm_backend.hip` leaves `staged_copies` at 0
and fails this case. That is the mutation that proves the wiring, and it is the
one `AGENTS.md` "Nothing lands dead" asks for.

This case measures nothing on a build with no ROCm device, exactly like every
other case in that file, and the gate line below therefore names the board it
was run on.

## 5. Gates

Every selector names its binary and prints its case and assertion counts. This
row has already shipped two selectors that matched nothing and reported
`Status: SUCCESS!`; a selector that matches nothing is indistinguishable from
one that passes.

| binary | selector | cases | assertions |
|---|---|---|---|
| `test_rocm_pinned_h2d` | (none — whole binary) | RECORDED AT §7 | RECORDED AT §7 |
| `test_backend_cross_device` | `-tc=*pinned bounce*` | RECORDED AT §7 | RECORDED AT §7 |
| `test_backend_cross_device` | (none — whole binary) | RECORDED AT §7 | RECORDED AT §7 |
| `test_backend_cross_device` | `-tc=*DSA*` | RECORDED AT §7 | RECORDED AT §7 |

Baseline to hold at `98e2cd7da`: `test_backend_cross_device` whole binary
**60 cases / 84833 assertions**, `-tc=*DSA*` **2 / 273**.

```sh
python3 scripts/check-agent-record.py
python3 scripts/check-commit-style.py --range origin/main..HEAD
python3 scripts/check-commit-trailers.py --range origin/main..HEAD
python3 scripts/check-pr-size.py --base origin/main --head HEAD \
  --branch row/MODEL-MM-QWEN4-EXP-ROCM-CHUNKED-H2D
```

**The model gate is the deliverable.** `/workspace/ckpt/qwen4exp-flash-next-iq1s`
shard 1, `examples/vllm-server` with `--device auto`, `VT_ROCM_MANAGED_ALLOC`
unset, on `strix:gpu0` inside an `rc` lease. Does it produce a token? If it does,
that is this row's G3 and the numbers follow — TTFT, prefill and decode tok/s,
load seconds, peak host `VmHWM` / `RssFile` / `RssAnon`, device memory from
`/sys/class/drm/card*/device/mem_info_vram_used` (`rocm-smi` is not on `PATH` in
the leased container), the exact build and run recipe, revisions, artifact sizes,
environment and contention. Generation is repeated at least three times and
reported as a spread, because gfx1151 fails about two of five identical greedy
runs with an illegal GPU memory access
(`ISSUE-LOCAL-01M2BY2M2ATNVR3XQKV2DB1BJD`) and one green there is unreplicated.
If it does not produce a token, NO number is recorded and the `wchan`
distribution is reported instead, including whether it moved off
`svm_range_set_attr`.

The artifact's compiled feature set is asserted before it is timed: `ldd` for
`libamdhip64.so.7` and the HIP arch it was built for.

## 6. Risks

- **The stall survives this too.** Then the trigger is neither the residency of
  the source nor the shape of the transfer, and the next hypothesis is the
  allocation side — 29.69 GiB of `hipMalloc` on a 33.27 GB board behind a 96 GiB
  carve, or the CIFS mount, which §6a already named as an unseparated confound.
  A negative result with a `wchan` distribution is the reportable outcome, not a
  failure of the change.
- **An extra `memcpy` per 64 MiB slows the load.** Load seconds are recorded on
  both arms of `VT_ROCM_PINNED_H2D_MIB` in one binary, so the cost is measured
  rather than argued.
- **256 MiB of pinned memory on a 31 GiB host.** It is allocated once, it is the
  oracle's own bound, and it replaces an unbounded pinned range the driver was
  creating per copy.
- **A qualifying copy inside a graph capture.** Guarded by term 4, and the guard
  is in the truth table.
- **`hipPointerGetAttributes` leaves a sticky error for an unregistered host
  pointer.** It returns `hipErrorInvalidValue` for one, which is the very answer
  we want, and the last error is cleared with `hipGetLastError()` immediately so
  it cannot poison the next `Check`.

## 7. Outcome

RECORDED ON LANDING.

## 8. Stop conditions

- The ring cannot be placed without changing `Backend::Copy`'s asynchronous
  contract: STOP and return `NEEDS_DECISION`.
- The change would have to widen past `RocmBackend::Copy` to be effective: STOP
  and return `NEEDS_DECISION` rather than editing a second backend.
- `strix:gpu0` is unreachable or the controller is down: the model gate is
  reported UNVERIFIED. It is never replaced by an `ssh` plus a file mutex the
  fleet cannot see.
