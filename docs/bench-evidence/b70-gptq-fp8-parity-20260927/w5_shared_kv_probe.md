# W5: sampled MTP4 cycle attribution and shared-KV probe (2026-09-28)

This is diagnostic evidence, not a performance promotion. The pinned B70
GPTQ/FP8 model ran native C++ eager MTP4 with P4096, T1/p.95/k20, seed 42,
page 1664 after automatic 1600→1664 hybrid adjustment, prefix off and C1.
`VT_B70_PROFILE=1 VT_XPU_PROFILE=1 VT_B70_PROFILE_OUTPUT_TOKENS=32`
profiles the same retrieval prompt used by the P4096/O1024 benchmark. The
profile harness now accepts speculative output bursts, whose callback count
is smaller than their emitted-token count. It records queue events and host
spans to a path outside Git. The first callback includes prefill and the first
speculative cycle; the table below examines the eight later bursts from the
same 32-token trajectory. Summed GPU stages can overlap and are not an
end-to-end denominator.

| Route | Post-first callback wall | GPTQ oneDNN | GDN spec | Attention split + reduce | Sum profiled GPU stages |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing Split-K | 748.2 ms | 251.6 ms | 222.9 ms | 140.1 ms | 693.2 ms |
| Shared KV: 2 queries × 2 GQA heads | 835.9 ms | ~252 ms | ~225 ms | 224.1 ms | 780.0 ms |
| Shared KV: 2 queries × 1 head | 787.5 ms | ~252 ms | ~225 ms | 175.5 ms | 732.1 ms |

Both experimental variants loaded each K/V element once for two adjacent
query positions and applied a separate causal mask per query. They were gated
to C1, Q2–Q5, model head geometry, FP8 E4M3, page 1600/1664 and unit KV
scales; every other shape retained Split-K. The first variant retained the
existing two-head GQA reuse. The second reduced register pressure by handling
one head per workgroup. A focused 16-case Q2–Q5 test across 4K–5K contexts
reported bitwise equal F16 operator output and witnessed both selected routes.
The Q5 synthetic host-timed operator looked favorable (about 2.1→1.7 ms for
the first variant, 2.3→1.3 ms for the second). **Neither result carried to the
real model**: the profiled Attention stage and cycle wall both regressed.

The first variant's complete P4096/O1024 request measured 29.57 derived
decode tok/s versus 42.09 for Split-K. The same-seed streams matched for 173
emitted tokens; token 174 differed, and the subsequent draft acceptance and
text trajectory diverged (701/1283 accepted/proposed versus 770/1012). A
separate P4096/O32 pair matched all 32 token IDs and 23/29 draft acceptance,
which permitted the short same-trajectory profile above. The 1K rate cannot
be used as an isolated kernel-speed comparison after the trajectories split;
the short profile independently rejects both variants on kernel and cycle
time. Their kernel code and tests were removed from the branch. The production
Split-K route and its existing quality behavior remain in place.

The next W5 candidate should address the larger measured costs: about 31 ms
GPTQ and 28 ms GDN per post-first cycle, before returning to a more different
Attention tiling strategy. The profile says little about Python-relative
operator parity; that needs matched source-side traces.
