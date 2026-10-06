Read BENCHMARK_REPORT_2026-10-06.txt first. This directory commits the report,
complete scenario/wave summaries, frozen Python baseline, measured source delta,
actual exit/resource receipts, exact historical commands and worker logs.

The opt-in benchmark entry point is in tests/vllm/models/test_xpu_exl3_mtp.cpp:
"XPU EXL3 public engine README: complete serving matrix".
It requires VT_B70_EXL3_MODEL, VT_B70_EXL3_README_WORKLOAD and a new output path
in VT_B70_EXL3_ENGINE_OUTPUT. It is an external-model development probe, not
an automatically runnable unit test. Build/runtime dependencies and admission
limits follow docs/EXL3_XPU.md. Full measured launch arguments are preserved in
*-command.json; paths, model/dependency mounts and output must be adapted on a
new machine. Run GPU workers sequentially with production stopped.

Large raw request traces and tokenized workloads remain outside Git. They are
included, together with original preparation/analysis scripts, in the new Pro
review ZIP under evidence/readme-native-serving-2026-10-06/. EVIDENCE_INDEX.json
records their byte counts and hashes. Paths at the end of the result report refer
to that complete evidence directory. Original scripts and commands retain their
historical local paths; they are evidence, not a portable turnkey launcher.
External weights, compiled executables, build trees and model caches are omitted
from the review ZIP; their identities and commands remain in the receipts.

Validation already performed: focused test_xpu_exl3_mtp compile/link succeeded;
full matrix 183251/183251 assertions; isolated 64K 4707/4707; aligned 16K 4383/4383.
All three actual exits were zero. No new GPU benchmark is needed to package
these frozen observations. The 16-slot smoke failures are included explicitly.

This benchmark covers the full README serving matrix and supplementary 64K
resend at 180 W; coding/media applications and power-limit sweeps were not run.
The previous upstream-readiness report remains historical evidence. Its narrow
primary timing is distinct from this sampled, complete serving matrix.
