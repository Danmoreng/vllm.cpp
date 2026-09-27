# P5: bound native Split-K by active KV pages

The P4096/D64 benchmark previously built a compact block table from the
active sequence: three columns at FP8 page size 1600. Its
`VT_B70_BENCH_MAX_CONTEXT=262144` was an admission limit, not a 164-column
table. The old experiment note that described 262400 reserved KV slots in
that full-model run conflated these two settings. The former Split-K plan
therefore used 150 parts (`ceil(4800/32)`), while the rejected exact-active
candidate used 128. The original timing result remains valid; its stated
capacity and old part count have been corrected in `p5_split_reduce.md`.

The benchmark now accepts `VT_B70_BENCH_BLOCK_TABLE_COLS` to pad only the
block table, leaving physical KV allocation and active tokens fixed. Padded
entries point to block zero and are never read for the active context. The
configuration log records the reserved and final active column counts.
With the old kernel plan, same-binary P4096/D64/O65, FP8 E4M3, page 1600,
native GDN, eager batch 1, one warmup and two timed rounds gave:

| Table columns | Planned parts | Decode forwards/s | Prefill tokens/s |
| --- | ---: | --- | --- |
| Dynamic, three at final step | 150 | 24.007, 24.010 | 1399.07, 1404.17 |
| Six reserved | 256 | 23.308, 23.274 | 1399.19, 1401.93 |
| 164 reserved | 256 | 23.287, 23.263 | 1397.49, 1399.88 |

Six and 164 columns have nearly the same decode rate. The larger part count,
rather than table upload width, is the observed performance cliff. The
native Split-K kernel now caps its plan to the **active context rounded up to
a KV page** for the qualified B70 E4M3 Q24/KV4/D256 causal, unit-scale,
page-1600 shape. `VT_XPU_ATTN_SPLIT_ACTIVE_PAGE_CAP=0` restores the old
capacity-based plan for A/B. Other shapes retain their previous behavior.
This keeps the compact table's 150 parts at P4096 while preventing padding
from inflating the plan. `VT_XPU_TRACE_SPLIT_PLAN=1` logs the selected part
count in untimed diagnostic runs.

The focused GPU test compared output with the native reference for tables
with 3, 6 and 164 columns and passed 17 assertions. Its plan trace recorded
150 parts for all three widths with the new selector; the 164-column opt-out
selected 256. Full-model A/B with 164 columns measured:

| Route | Decode forwards/s | Median | Peak GPU bytes |
| --- | --- | ---: | ---: |
| Old capacity plan, opt-out | 23.331, 23.300 | 23.316 | 19,460,739,068 |
| Active-page cap | 24.011, 24.007 | 24.009 | 19,460,739,068 |

The decode gain is **2.97%** for the padded-table case. Prefill medians were
1398.24 and 1399.64 tokens/s. With the default dynamic table, the new route
measured 23.981/23.928 versus 24.007/24.010 decode forwards/s before the
change, within this short run's variation. The part count remained 150.

For quality, C++ P4096/D1024/O1025 used the 164-column table and captured
prefill, decode 1, 64 and 1024 logits. It used the same synthetic prompt
and teacher-forced decode IDs as the pinned Python long-run golden. A subset
manifest selected those four original Python captures; each file hash and
context was revalidated by the comparator. All frozen full-vocabulary TV,
KL, top-10 and top-1 gates passed; see
`p5_split_active_page_cap_quality_4096_1024.json`. Python was the external
oracle only; the engine and Split-K work remain native C++/SYCL.

Small raw operator, model and quality logs are under
`raw/p5_split_active_page_cap_*.log.gz`. Full F32 logits remain outside Git.
This only removes work caused by table padding; per-request split plans,
other contexts and concurrency remain P5 tasks.
