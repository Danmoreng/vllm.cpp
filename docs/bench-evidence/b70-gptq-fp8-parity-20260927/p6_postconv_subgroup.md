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
