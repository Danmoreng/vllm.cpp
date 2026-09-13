# Per-kernel decode-time attribution on gfx1151, 13 September 2026

Row: `BACKEND-ROCM`. Issue: `ISSUE-LOCAL-01M2DCDVCQNGQB4F3AHXQRYK8H`.
Spec: [`rocm-decode-kernel-attribution.md`](../../.agents/specs/rocm-decode-kernel-attribution.md).

**No cross-engine ratio appears in this file.** The `qwen4_exp` ROCm arm has no
declared token-exact gate (`ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG` -- absent,
not failing), so `AGENTS.md` §Gates forbids a performance denominator on it.
Every figure below is this engine against itself.

## 1. The instrument, and why the alternatives lost

`rocprofv3 --kernel-trace`, windowed offline by the per-token sampler dispatch.

| Candidate | Disposition |
|---|---|
| `rocprofv3` / `rocprof` | **CHOSEN.** Works on this board; see §2 |
| HIP events around dispatches | Rejected: serialises the queue it measures, and changes the binary, so the profiled and control arms stop being the same bytes |
| ROCTX ranges | Deferred: same distortion at smaller magnitude, and needs source changes to place the ranges. `rocprofiler-sdk-roctx` is installed, so it stays available |

The deciding property is that **nothing is added to the engine**. The traced
binary is byte-identical to the control binary, so the profiler's cost is a
measurable difference rather than a confound.

## 2. #3040 is real and it is narrower than its summary sentence

[#3040](https://github.com/mudler/vllm.cpp/issues/3040) is the recorded blocker,
and `docs/bench-evidence/strix-kernel-trace-3015-20260907/README.md` states it as:

> Do not derive kernel durations, time shares, or host idle bounds from these
> timestamps.

That sentence is why `cc0e827dd` had to close with "where our decode step time
goes is unestablished". **The measurement under it does not support a blanket
refusal.** Read directly off the committed capture
`diag-kernel-control-asaj36ta--trace--e5367aefafa8--1_kernel_trace.csv.gz`:

| Property | Value |
|---|---:|
| `KERNEL_DISPATCH` rows | 85,737 |
| rows with `End < Start` | **0** |
| rows with `End == Start` | **0** |
| min / median / max duration | 7 ns / 52.301 us / 5.212 ms |
| sum of kernel durations | 11.749 s |
| first-start to last-end span | 12.551 s |
| implied kernel occupancy | **93.6%** |
| timestamp-swap warnings | 62 (**0.072%** of rows) |

A table of unusable timestamps does not sum to 93.6% of its own wall span with
zero inversions. rocprofiler-sdk 1.1.0's
[`profiling_time.hpp:82`](https://github.com/ROCm/rocm-systems/blob/97f5574fe2fdc7bef44fb01545347912ee9f1779/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/tracing/profiling_time.hpp#L82)
**repaired** the 62 rows it warned about and delivered a monotonic table.

**The residual is bounded, and the bound is computed by the tool, not asserted
here.** The CSV does not mark which rows were adjusted, so the worst case is
assumed rather than inspected: if all 62 were maximally wrong and each really
cost the largest duration in the trace, the error is `62 x 5.212 ms = 0.323 s`
against an 11.749 s budget -- **2.75%**. Nothing in this file rests on a margin
narrower than that.

This does not close #3040. #3040 still owns per-row exactness, and it should.
What changes is that a **ranking over tens of thousands of rows** is no longer
refused by it.

## 3. The window, and why it is self-validating

Greedy decode emits one sampler dispatch per token. Slicing between consecutive
`ArgmaxK` dispatches on the committed capture gives:

- 64 marks for 64 generated tokens;
- **63 of 63** segments containing **exactly 1330** dispatches;
- step wall 182.37 to 184.88 ms, a 1.4% spread;
- 1,945 rows before the first mark, excluded as model load and prefill.

The constant dispatch count is the validation. A marker that is not once per
token produces ragged segments, and `scripts/rocm-rank-kernels.py` **refuses**
(exit 3) instead of printing a table. The first decode step is discarded as
well, because our arm stages lazily on the first generation.

This is what the previous clock instrument could not do. Its window carried
~35 s of model load into a ~46 s leg and read 39.1% busy pooled, so an effect
below ~10% was not resolvable. These windows contain decode and nothing else.

## 4. What the instrument cannot see, stated rather than hidden

1. **Kernel execution only.** Whatever the step wall exceeds the kernel-busy sum
   by is host time. The tool prints that percentage rather than attributing the
   gap to kernels.
2. **No overlap.** Durations are summed per name. The printed occupancy is what
   tells a reader whether one queue was in fact serial.
3. **Profiler attach cost.** Measured as an A/B on tok/s, same binary, same
   artifact, alternating legs. See §6.

## 5. Provenance

| Input | Pin |
|---|---|
| vllm.cpp revision built and run | `cc0e827dd9425ee00f3827e2fc36d2ac46560202` (`origin/main`), asserted in-job, clean tree |
| Device | `strix:gpu0`, gfx1151, Radeon 8060S (AMD Ryzen AI MAX+ 395) |
| ROCm | 7.2.4, installed in-job from `repo.radeon.com/rocm/apt/7.2.4` |
| Profiler | `rocprofv3`, `rocprofiler-sdk` 1.1.0 |
| Linked HIP runtime | `libamdhip64.so.7`, `libhipblas.so.3`, `libhipblaslt.so.1`, `librocblas.so.5`; `not_found_libs=0` |
| HIP architecture | `ROCm backend: ENABLED for arch(es) [gfx1151]`, asserted before the build ran |

`VT_ROCM_MANAGED_ALLOC` was unset: the job refuses by name if any
`HSA_`/`ROCR_`/`HIP_`/`GGML_`/`VT_` variable is inherited, and it printed
`inherited_env NONE`.

**One provisioning fact worth keeping.** Ubuntu 24.04 noble ships its own
`rocminfo` 5.7.1, `rocm-cmake` 6.0.0 and `hipcc` 5.7.1 in universe, and apt
prefers them, which makes `rocm-hip-sdk` unsatisfiable with
`rocm-hip-runtime : Depends: rocminfo (= 1.0.0.70204) but 5.7.1-3build1 is to be
installed`. The repair is an apt pin:

```sh
printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1000\n' \
  > /etc/apt/preferences.d/rocm-radeon-first
```

**And one rc fact.** A job script that does `exec >"$LOG" 2>&1` closes the job's
stdout; `rc` then reports `log stream ended: unexpected EOF` and **reaps the
job**. Lease `b4e2330a` died that way before it installed anything. Stream to
`rc`'s stdout instead.

## 6. The measurement, and where it stopped

Four leases on `strix:gpu0`. The instrument's design is sound and its cost is
negligible. **The profiler cannot write a record on this worker.**

### 6.1 What ran

Every leg below is the same binary at `cc0e827dd`, same board, same prompt,
temperature 0, `--max-num-seqs 1`, `VT_ROCM_MANAGED_ALLOC` unset, artifact
sha256 asserted before the run.

| Leg | Artifact | Profiler | Result |
|---|---|---|---|
| `U1` | qwen4_exp UD-IQ1_S, 40 tok x 4 | off | rc=0. **5.162 / 5.157 / 5.189 tok/s** on runs 2-4; run 1 **0.963 tok/s** |
| `P1` | qwen4_exp, 40 tok x 4 | `--kernel-trace` csv | **Never exited.** Ran past `timeout 3600` and ignored SIGTERM; killed with the lease |
| `C27u` | Qwen3.8-27B Q4_K_M, 40 tok x 3 | off | rc=0. 2.999 / 5.333 / 5.316 tok/s |
| `C27p` | 27B, 40 tok x 3 | `--kernel-trace` csv | **rc=134, SIGABRT at 37 s.** Zero-row CSV |
| `Q4Eu` | qwen4_exp, 12 tok x 2 | off | rc=0. 0.315 then **4.823 tok/s** |
| `Q4Ep` | qwen4_exp, 12 tok x 2 | `--kernel-trace` csv | Workload finished in 47.86 s at **4.804 tok/s**; then hung 1860 s and was SIGKILLed. Zero-row CSV |
| `R27` | 27B, 40 tok x 3 | `--kernel-trace` **rocpd** | rc=134. 2.1 MB database, 639 kernel symbols, **0 dispatch rows** |
| `RQ4` | qwen4_exp, 12 tok x 3 | `--kernel-trace` **rocpd** | Workload finished (3.859, 4.992 tok/s); SIGKILLed at 1530 s. 2.4 MB database, **0 dispatch rows** |
| `M27` | 27B, 40 tok x 3 | csv, `vm.max_map_count` raised 262144 -> 2097152 | rc=134 at 45 s. Same failure |

### 6.2 The distortion, measured

The pair that answers it is `Q4Eu` against `Q4Ep`: same binary, same artifact,
same 12 tokens, alternating, one profiled and one not.

| | steady-state tok/s |
|---|---:|
| unprofiled (`Q4Eu` run 2) | 4.823 |
| profiled (`Q4Ep` run 2) | 4.804 |
| **difference** | **-0.4%** |

**Attaching `rocprofv3 --kernel-trace` costs essentially nothing at run time.**
The profiled command completed in 47.86 s by rocprofv3's own timer. The design
goal -- an instrument that does not change what it measures -- is met. What
fails is strictly downstream of the workload.

### 6.3 Why nothing was written, exactly

From `w2-Q4Ep-stderr`, after the generations finished:

```text
W [rocprofv3] '/tmp/kattrib/build/examples/vllm-cli --model ... --repeat 2' :: 47.857123 sec
E output_stream.cpp:111] Opened result file: .../10815_kernel_trace.csv
ring_buffer: munmap failed: Invalid argument
F ring_buffer.cpp:106] mmap failed with errno 22 :: Invalid argument
    ... abort
W tool.cpp:3104] [rocprofv3_error_signal_handler] rocprofv3 caught signal 6...
W tool.cpp:3184] ... executing chained sigaction (SIGINFO)
W tool.cpp:3104] [rocprofv3_error_signal_handler] rocprofv3 caught signal 6...
W [29 minutes later] rocprofv3 caught signal 15...
```

Three facts, in order:

1. **The ring buffer fails to `mmap` with `EINVAL` at output generation**, after
   the traced process has already finished, and rocprofiler-sdk logs it at
   FATAL, which aborts.
2. **Its own signal handler then deadlocks.** It catches signal 6, re-enters,
   and sits there. `timeout`'s SIGTERM arrives 29 minutes later and is also
   survived; only SIGKILL ends it. This is why the first lease consumed an hour
   on one leg.
3. **No dispatch record survives.** Both writers produce a container and no
   content: the CSV has zero rows, and the rocpd database has its 639 kernel
   symbols and an **empty `rocpd_kernel_dispatch` table**.

Reproduced on **both artifacts, both output formats, four leases**, with the
same `ring_buffer.cpp:106` line every time. Raising `vm.max_map_count` eightfold
did not move it.

**This is not the profiler version.** `rocprofv3 --version` reports 1.1.0 at git
revision `97f5574fe2fdc7bef44fb01545347912ee9f1779` -- **byte-identical to the
revision the 2026-09-07 capture pinned**, which wrote 85,737 dispatch rows from
this same board. The profiler is the same; the environment is not. That capture
ran inside a purpose-built podman image (`localhost/vllmcpp-strix-profile`);
this worker is bare Ubuntu 24.04 with the same packages installed by apt, and it
has no `podman` at all. The environment is therefore the live suspect, and
`ulimit -l` on this worker is **8192** (8 MiB), which is the kind of limit a
locked ring buffer would care about.

### 6.4 The answer to the question that was asked

**The qwen4_exp ranked table is UNVERIFIED.** No dispatch record can be obtained
from this worker as provisioned, so where the 5.0-5.3 tok/s goes is still
unestablished. That is a real result: **the next thing to fix is observability
itself**, and it is now a narrow, named repair rather than a shrug -- reproduce
the 2026-09-07 container image, or find which of its properties the bare worker
lacks.

## 7. What the instrument does produce

`scripts/rocm-rank-kernels.py` run against the committed 2026-09-07 capture --
our own ROCm decode, gfx1151, Qwen3.8-27B Q4_K_M, 62 decode-only steps with the
first discarded:

```text
step wall ms          : median 183.82  min 182.37  max 184.88
kernel busy per step  : 178.08 ms (96.9% of step wall; the remainder is HOST time)
#3040 swap warnings   : 62 of 85737 rows (0.0723%); worst-case 2.75% of the budget

kernel                                    share%   total ms  n/step   mean us   min us    max us
KQuantGemmK                               43.01%     4748.1   248.0     308.8     12.8    1248.3
wvSplitKSml                               28.13%     3106.1   144.0     347.9    253.6     499.8
QuantizeQ8KK                               7.65%      844.3   257.0      53.0     34.0     161.9
Cijk_Alik_Bljk_BSS_BH_MT128x32x16_SE_1LD   7.46%      823.2    96.0     138.3     85.3     160.1
GdnScanK                                   6.50%      717.2    48.0     241.0    214.5     278.8
KQuantGemmKCoopQ6K                         3.02%      333.0     9.0     596.9     24.1    5209.8
RmsNormRowKernel                           1.15%      126.8   129.0      15.9      0.9      40.3
GdnPostConvChunkedK                        0.86%       94.7    48.0      31.8     20.9      54.3
AttnQkNormRopeGateK                        0.83%       91.7    16.0      92.5     87.0     108.9
```

The full table is
[`ranked-table-q38-27b-q4km.txt`](rocm-kernel-attrib-gfx1151-20260913/ranked-table-q38-27b-q4km.txt).

**This is a different model from the one the wave was opened on**, so it does not
answer the qwen4_exp question and is not offered as if it did. What it does show
is that the instrument works end to end and what its output looks like.

Three things in it are worth a later row, each stated as an observation and none
of them measured against an oracle:

- **The time is NOT flat, and it is NOT one kernel either.** Two entries carry
  **71.1%**: our own `KQuantGemmK` at 43.01% and AMD's skinny-GEMM
  `wvSplitKSml` at 28.13%. Adding the Tensile `Cijk_*` entry, **35.6% of the
  decode step is inside vendor BLAS** rather than our kernels.
- **`QuantizeQ8KK` runs 257 times per step for 7.65%.** It is the activation
  quantizer feeding the k-quant GEMM, and it is dispatched slightly more often
  than the GEMM it feeds.
- **1330 dispatches per token is the structural number.** Across 63 consecutive
  steps it never varied by one. `KQuantGemmK` 248, `QuantizeQ8KK` 257,
  `wvSplitKSml` 144, `RmsNormRowKernel` 129, `Cijk_*` 96, four `Gdn*`/norm
  families at 48 each. At the measured 96.9% occupancy the step is not
  launch-bound on this model, so dispatch count is an observation here rather
  than a diagnosis -- but it is the first number to re-read once a qwen4_exp
  trace exists, because that arm is 5.0-5.3 tok/s against this one's 5.3.
