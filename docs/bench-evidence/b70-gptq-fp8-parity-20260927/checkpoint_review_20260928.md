# B70 GPTQ/FP8 MTP checkpoint for ChatGPT Pro (2026-09-28)

Branch: `b70-gptq-int4`. Target: Qwen3.8-27B GPTQ INT4/G128, FP16 target and head, FP8 E4M3 KV, eager native C++/SYCL inference on one Arc Pro B70. Python vLLM 0.30.0+xpu, XPU kernels 0.1.15.4, MTP4 is the external reference. C++ does not execute Python or Triton at inference time. The user's P7 follow-up has work packages W0–W8; implementation is incomplete. See the linked evidence files in this directory for commands, shapes, hashes and full results.

## Measured request performance

| Matched request scope | Native C++ | Python reference | C++ / Python | Provenance |
| --- | ---: | ---: | ---: | --- |
| P4096/O1024, sampled eager MTP4, FP8 KV, 1664 page, C1, decode emitted tokens/s | 49.1052 | 71.8318 | 68.4% | C++ five-run median after GDN workgroup change; Python earlier matched-workload run, not simultaneous |
| Same C++ build, optional Xe2 shared-KV verifier, P4096/O1024 | 62.7176 | 71.8318 | 87.3% | One unprofiled candidate run; Python earlier run, different sampled continuation |
| P4096 prefill tokens/s, FP8 KV, 1664 page, non-MTP comparison | 1004.96 | 1586.66 | 63.3% | Earlier page-geometry comparison, different from current MTP4 serving scope |

The MTP decode rate is `1023 / (O1024 wall - separate O1 TTFT)`, an end-to-end client estimate. The accepted GDN launch change raised C++ MTP4 decode from 41.8496 to 49.1052 tokens/s (+17.3%) in alternating route runs with the same token hash and 770/1012 accepted drafts. No new end-to-end prefill result has been measured after that change. Do not mix these emitted-token rates with non-speculative forwards/s or per-kernel timings.

In a fresh same-build C++ A/B, the existing default made 49.3827 decode tokens/s (O1 4.24323 s, O1024 24.959 s, 1012 proposed/770 accepted, output hash `6972477010856893408`); the optional Xe2 verifier made 62.7176 (O1 4.25066 s, O1024 20.5619 s, 1024 proposed/767 accepted, output hash `7809634488016794068`). Both passed 50/50 harness assertions with the same 4096-token prompt hash `4760835697920107937`. This is a one-run **+27.0% request-level candidate**, not a qualified speedup: the sampled continuations differ, and same-prefix Python target-logit quality plus Q2–Q5 route tests are still needed. The default remains Split-K.

## W0–W8 status

| Package | Current status and remaining gate |
| --- | --- |
| W0 references/contracts | Native capability matrix, call-path and benchmark evidence exist. Exact source-to-installed-wheel correspondence remains unproved. |
| W1 graph baseline | C8 case-5 first difference is a near-tie at row 2; graph stays opt-in and capped. Request/state/slot ownership still needs diagnosis. |
| W2 decode attribution | Exact top-20 native sampler qualified with a modest measured win. Non-speculative same-scope Python parity remains open. |
| W3 eager MTP1 | Native draft, conv/GDN, separate cache, speculative state and model smokes exist. EOS/cancellation and complete lifecycle qualification remain open. |
| W4 sampled MTP | Native T1/p.95/k20 MTP4 works at C1. Same-prefix Python/C++ raw target TV is 0.02445/0.02329 on verification rows 0/4, exceeding the frozen 0.02 gate. Prefix-dependent processors and broader distribution/RNG gates remain open. |
| W5 MTP4 speed | Packed INT4 draft weights, workgroup-64 GDN gain and a Torch-free optional Xe2 Q2–Q5 shared-KV attention port exist. New operator replay is near bitwise, but model-level qualification and production promotion remain open. |
| W6 C2/C4 | Functional multi-request coverage exists, but no qualified native sampled MTP4 C2/C4 performance parity. |
| W7 prefix/long context | Recurrent prefix snapshots with speculation and production long-context gates remain open. |
| W8 vision | Deferred by the plan; no native vision implementation claimed. |

## W5 experiments and decisions

- Profiling isolated speculative GDN as a large per-cycle cost. The scoped workgroup-64 launch cut that GPU sum from 222.91 to 102.52 ms in the short profile and produced the measured +17.3% full-request decode gain.
- Two simple shared-KV C++ probes regressed the complete request, so they were not promoted. The Python M04 real-Q5 replay with identical Q/FP8 K/V measured about 0.14–0.16 ms host / 0.165 ms device span. Existing C++ Split-K measured 1.999 ms host, with 1.905 ms partial plus 0.067 ms reduction. C++ Split-K had 0.03114% relative RMS output error against Python.
- A narrow Torch-free port of the pinned Xe2 M04 shared-KV policy now runs through the VT C++ API behind `VT_XPU_XE2_VERIFY=1`. The same-input real Q5 replay passed 34/34 assertions, differed in 5 of 30,720 FP16 outputs, had relative RMS `5.21419e-06` and max absolute error `0.000244141`. Its 20-call synchronized host median was 0.213176 ms. The P4096/O32 smoke kept the default's 32 emitted token IDs and 23/29 draft acceptance. The full one-run A/B is recorded above. The default path remains Split-K until the broader quality and Q2–Q5 gates pass.
- The real first-layer 4K pre-quantization K/V source already differed between Python and C++ by 0.0965%/0.1091% relative RMS. With identical Python FP16 source, the native FP8 writer matched Python cache bytes exactly. The attention crossed replay attributed more of the Q5 output gap to cache content than to the attention arithmetic. A writer profile bounded prefill cache-writing opportunity at <0.84% of observed TTFT. No cache writer change was promoted.
- A native-GDN prefill opt-in reduced one O1 TTFT (4.0730 to 3.5625 s), but the O1024 sampled request slowed (49.2570 to 43.7578 tokens/s) with a different trajectory. It stays opt-in.

## Highest-value next decisions for the revised plan

1. Resolve the source difference before the first full-attention K/V cache write: compare the same real GDN-to-QKV activation and GPTQ projection/QK norm/RoPE outputs in the two engines. This also addresses the sampled row-0/4 TV failures.
2. Qualify the new native M04-like path at Q2–Q5, then run paired P4096/O1024 C1 route A/B and report selected-route witnesses, token hashes, accepted/proposed counts, TTFT, decode rate and device/host cost. Keep it opt-in if it regresses or changes target quality.
3. Complete W1 graph ownership diagnosis, W4 prefix-dependent processors/RNG coverage, then C2/C4 and recurrent prefix/long-context work. Retain separate plans for vision and production multimodal quality.

Raw model weights, Q/K/V captures, logit arrays and long benchmark logs are deliberately outside this review archive. The small committed evidence files identify the exact local captures and hashes where available.
