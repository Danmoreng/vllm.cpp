# P6: native subgroup GDN post-conv for the FP16 model shape

The existing XPU `GdnPostConv` kernel assigned one work item to each Q/K or
V head. Its Q/K branch summed 128 components serially, then loaded them
again to normalize; the V branch copied 128 components serially. The new
native SYCL path assigns one SG16 subgroup per head. Lanes share the Q/K
squared-sum reduction and copy normalized Q/K and V components in parallel.
One lane still computes the scalar g/beta transforms. The path is limited to
the B70 model shape: FP16 conv and Q/K/V outputs, Hk16, Hv48, Dk/Dv128.
Other shapes retain the previous kernel.
`VT_XPU_GDN_POSTCONV_SUBGROUP=0` selects the old kernel for same-binary A/B.
Python is an external quality oracle only; no Python or Triton runtime is
called by the C++ engine.

The focused GPU FP16 test compared scalar and subgroup paths at one and 256
tokens and passed 30 assertions, including both selected kernel routes.
The existing CPU-reference F32/BF16
post-conv and gated-RMSNorm test passed 100 assertions. CPU `GdnPostConv`
does not support FP16, so the FP16 comparison deliberately uses the old XPU
kernel as its operator oracle.

A diagnostic P4096/O2 profile with the same pinned model and FP8 KV showed
48 post-conv kernel submissions in prefill and 48 in decode:

| Profile phase | Old total GPU event time | Subgroup total GPU event time |
| --- | ---: | ---: |
| Prefill | 136.91 ms | 29.17 ms |
| One decode forward | 4.70 ms | 0.29 ms |

These are event-duration sums from separate profiled runs, not exclusive
wall-clock savings. Unprofiled same-binary full-model A/B used one warmup
and two timed rounds per route, eager batch 1, native GDN and FP8 E4M3 KV
page 1600:

| Case and route | Prefill tokens/s | Decode forwards/s | Peak allocated GPU bytes |
| --- | --- | --- | ---: |
| P4096/D64 scalar | 1399.78, 1406.71 | 24.001, 24.009 | 19,460,738,408 |
| P4096/D64 subgroup | 1432.75, 1435.60 | 25.580, 25.575 | 19,460,738,408 |
| P8192/D64 scalar, 4096+4096 calls | 1334.80, 1329.78 | 21.245, 21.238 | 19,618,024,832 |
| P8192/D64 subgroup, 4096+4096 calls | 1363.50, 1358.71 | 22.450, 22.434 | 19,618,024,832 |

Median gains: **P4096 +2.20% prefill and +6.55% decode**; **P8192 +2.16%
prefill and +5.65% decode**. The 8K benchmark's two prefill model calls
match the Python worker's recorded query shape, although Python request
timing is not treated as equivalent to this synchronized C++ forward scope.

Pinned Python full-vocabulary quality passed the frozen TV, KL, top-10 and
top-1 gates at P4096 prefill and decode 1/64/1024, and at P8192 prefill
and decode 1/64. The P4096 comparison reused four original captures from
the longer Python golden; their hashes and contexts were revalidated.
At P4096, TV was 0.002537/0.014455/0.002367/0.002420; at P8192 it was
0.001646/0.015905/0.002947. See the two quality JSON files. Persistent
GDN state remained FP32 and the GPTQ/FP8 model contract was unchanged.

Compact raw A/B, profile, quality and focused-test logs are under
`raw/p6_postconv_subgroup_*.log.gz`. F32 logits remain outside Git.
Further P6 work includes proving whether prefill-state
copies are removable and examining actual oneDNN primitive costs; neither
is changed here.

## P6 follow-up: actual oneDNN route and state-copy budget

An `ONEDNN_VERBOSE=all` P4096/O2 warmup plus measured request with the pinned
oneDNN 3.13 build recorded 13 matmul shapes. All 1,220 executions were
`gpu,matmul,jit:gemm:any`: GPTQ weights were U4 with `ba` layout, FP16
activations/output used `ab`, group scales used FP16 `128x1` with mask 3,
the scalar weight zero point was S8, and the GPTQ FP math mode was F16.
The dense BA and LM-head weights were FP16 `ba`. The trace contained no
separate reorder execution. Six first-use GEMM kernels missed the kernel
cache and seven hit it; the warmed model held 13 primitives, with no
subsequent creation in the scored request. All reported scratch sizes in
the C++ stage profile were zero. The compressed verbose trace is
`raw/p6_onednn_verbose_4096_o2.log.gz` (source SHA-256
`aa75f6a6dc7fa9cbf4b2d48bc53432e86fbdd8132bbe52816fa6ae1cc6e2e988`).
The verbose queue was not profiling-enabled, so its execution time field is
zero and cannot be used as a latency measurement.

The separate P4096/O2 stage profile above measured 256 GPTQ stream spans:
1936.475 ms during prefill and 24.482 ms for one decode forward. Dense
FP16 spans totalled 9.720 and 4.612 ms respectively (49 invocations in
each phase, including one M=1 LM head). It also measured 96 state gathers
and 96 scatters in prefill, totalling 1.288 and 1.349 ms of GPU event time.
Their combined event duration is under 0.1% of the 2.854 s profiled
prefill wall time; those event durations are not exclusive wall-clock
savings. The gather also applies `has_initial_state` zeroing, and both
operations honor indexed cache slots. Direct cache access would need to
preserve those contracts. These observations do not justify a state-copy
removal or a new GEMM; P6 keeps both unchanged pending a stronger candidate.
