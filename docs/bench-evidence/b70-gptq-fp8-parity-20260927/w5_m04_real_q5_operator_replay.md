# W5: Python M04 Q5 attention replay on native C++ Split-K (2026-09-28)

The layer bisect in `w5_same_prefix_layer_bisect.md` showed growing target
differences at full-attention layers. This test isolates the attention operator
from upstream projections, RoPE and already different cache contents. A
temporary probe in the installed production `b70_attention.py` captured the
first real five-row M04 verification call after a 4096-token retrieval prompt.
The probe was excluded from the committed runtime and the temporary Python
server was stopped after one eight-token request.

The Python image was
`local/b70-qwen38-vllm:vllm-0.30.0-xpu-kernels-0.1.15.4`, with the pinned
GPTQ-G128 model, eager MTP4, T1/p.95/k20, FP8 E4M3 KV, no prefix reuse,
and M04 shared-KV operator SHA-256
`784916abd42b614794becac634a6a154f854c73d6dfcb1a50cd939b563020c6a`.
The request reported exactly 4096 prompt tokens. M04 dispatched with Q5,
KV length 4101, 1664-token physical pages, unit K/V scales, softmax scale
0.0625 and 16 splits. Its Q was F16 `[5,24,256]`; the three active physical
K/V pages were copied in logical block-table order as E4M3 bytes. The raw
capture remains outside Git in `/tmp/b70_m04_real_replay` (~10 MiB). SHA-256:

| File | SHA-256 |
| --- | --- |
| `metadata.json` | `055d3114d3812e54e700ef32c052b2387b37ecc61c8173921dd43bf9aff09e53` |
| `q.bin` | `2360959279560c8b8796cd30e8f0d0c47e5509f26d5c2c2468cb3db50a2b58e1` |
| `k.bin` | `e25846c9cb395a6e949c1990183ec47749ba478f38bbe686212777e30f3d606c` |
| `v.bin` | `c4df2f5c697816210a7ed863a88218bea77994acee3a0faea5c536afb3f9a2d8` |
| `output.bin` | `51685e65deab1be2d75b2fb9fc67e936288d375c6489cc48392377f22323b0c1` |

The focused native test in `test_xpu_attention_fast.cpp` repacks only the K/V
memory layout for the public VT cache contract, retaining each FP8 value and
logical token order. It feeds the captured Python Q and cache bytes to C++
`vt::PagedAttention` with the same causal positions and scale, and checks that
the `attention_split_partial` route was selected. The pinned oneAPI builder
compiled the target, and the B70 test passed **29/29 assertions** with no
reference-tier hits.

| Verification row | Relative RMS difference | Max absolute difference |
| ---: | ---: | ---: |
| 0 | 0.02965% | 0.000976562 |
| 1 | 0.03259% | 0.000976562 |
| 2 | 0.03169% | 0.000976562 |
| 3 | 0.03148% | 0.000976562 |
| 4 | 0.03014% | 0.000976562 |
| All rows | **0.03114%** | **0.000976562** |

This is a same-input operator comparison, not an end-to-end target-quality
pass or a performance claim. It rules out a large intrinsic arithmetic gap
between these two attention kernels at this Q5 input. It does not rule out
different Q/K/V generation, cache quantization, positions, or small errors
amplified by downstream layers. The next discriminating capture is the native
C++ Q and active FP8 KV from the same prompt and first verification layer,
compared directly with the Python operands before revisiting shared-KV tiling.

## Native input capture and crossed replay

That next capture was completed on the same 4096-token prompt (token-ID FNV64
`4760835697920107937`) with C++ eager MTP4, FP8 KV and the 1664-token
effective page. The first C++ Q5 verification also had KV length 4101 and
unit scales. A temporary source probe captured its first full-attention
layer's Q, output, and three active K/V pages. The short P4096/O32 native
request passed 50/50 assertions and the capture instrumentation was removed
and the model target rebuilt. The raw files remain outside Git at
`/tmp/b70_cpp_m04_inputs` (~10 MiB). This capture did not separately record
the draft IDs; the earlier same-prefix probability probe did, while these
new Q tensors remain close across all five rows.

| Operand, active positions only | Python/C++ comparison |
| --- | ---: |
| Q F16 relative RMS | 0.0736% |
| K E4M3 relative RMS | 0.6824% |
| V E4M3 relative RMS | 0.7238% |
| K bytes equal | 96.42% |
| V bytes equal | 97.04% |
| Own complete attention outputs, relative RMS | 0.5293% |

The optional `VT_B70_M04_CPP_REPLAY_DIR` branch of the focused test swaps one
real operand group at a time, always comparing with the captured Python M04
output. The native C++ output is reproduced **bit-for-bit** from its captured
Q and KV, so the replay accurately models the production Split-K call.

| Q / K / V source | Relative RMS against Python M04 output |
| --- | ---: |
| Python / Python / Python through C++ Split-K | 0.0311% |
| C++ / Python / Python | 0.0550% |
| Python / C++ / Python | 0.3322% |
| Python / Python / C++ | 0.4091% |
| Python / C++ / C++ | 0.5285% |
| C++ / C++ / C++ | 0.5293% |

The crossed replays show that the existing cache values dominate this
first-layer attention-output gap; both K and V contribute, with V somewhat
larger for this request. They do not determine whether the cache differences
come from upstream projections, accumulated prefill differences, or FP8
storage semantics. A separate diagnostic converted every finite FP16 value
through the C++ host E4M3 codec and PyTorch's E4M3 cast: all 63,488 finite
encodings matched. The focused native XPU codec test also compares device and
host encoding across ties and scales. These checks weaken a simple rounding
rule hypothesis but do not yet compare the two engines' actual cache writers
on identical pre-quantization operands.

## First 4K prefill layer before FP8 conversion

The first full-attention prefill layer's post-projection, post-RoPE K/V was
captured before the native cache writer in both engines. Python identified
`language_model.model.layers.3.self_attn`; both captures had 4096 rows,
four KV heads, head size 256 and FP16 storage. The same request produced
prompt hash `4760835697920107937` and the 1664-token effective page.
Python first performs a **zero-valued 4096-row startup profile**, which is
not a real request. The diagnostic probe was corrected to arm only after
`/health` and the real request was submitted. The zero profile was excluded
from every number below. Neither capture probe remains in committed model
runtime code; the C++ focused model target was rebuilt after removal.

| First prefill layer, 4096 active tokens | K | V |
| --- | ---: | ---: |
| FP16 pre-quantization relative RMS, Python vs C++ | 0.0965% | 0.1091% |
| Exact FP16 values | 21.87% | 20.70% |
| Mismatched E4M3 bytes, actual Python vs C++ cache | 150,196 | 124,368 |
| Python FP16 source cast to E4M3 vs actual Python cache | **0** | **0** |
| C++ FP16 source cast to E4M3 vs actual C++ cache | **0** | **0** |
| Native C++ FP8 writer fed the captured Python FP16 source vs Python cache | **0** | **0** |

The byte counts cover 4,194,304 values per K or V. The C++ writer replay is
an optional branch of `test_xpu_attention_fast.cpp`, selected with
`VT_B70_M04_PREQUANT_DIR`; with the previous M04 and crossed-replay fixtures,
it passed **52/52 assertions**, zero reference-tier hits. It wrote all 4096
token slots using `ReshapeAndCacheFp8` on the B70. Thus the cache writers are
bitwise compatible **on these identical real FP16 inputs and unit scales**.
The source tensors already differ by about one tenth of a percent before FP8;
quantization turns those small differences into roughly 3–4% changed cache
bytes. The earlier crossed replay shows that this cache difference dominates
the first Q5 attention-output gap. This does not establish the cause of the
pre-quantization source difference, nor does it close the two sampled-logit
quality failures.

## Same-input Q5 operator timing

The captured Python Q and FP8 K/V were replayed on the B70 in fresh
single-process, 20-call warm measurements. Python used the production M04
adapter and native shared-KV operator. C++ used `vt::PagedAttention` with the
current Split-K route, identical logical Q/K/V and causal positions. The
Python result reproduced the captured M04 output exactly; the C++ replay
retained its 0.03114% relative RMS error and passed 31/31 focused assertions.

| Scope | Python M04 | C++ Split-K |
| --- | ---: | ---: |
| Synchronized adapter/call host median | 0.141–0.159 ms | 1.999 ms |
| Device timed span / kernel stages | 0.165 ms adapter span | 1.905 ms partial + 0.067 ms reduction |

The C++ host samples were bimodal, roughly 1.02–2.38 ms; the device partial
median confirms that the slow path is real GPU work, not solely a host wait.
The Python device figure is an event span around the entire adapter, whereas
the C++ figures are two individual kernel events. They show the size and
location of the opportunity, but are not a full-model speedup prediction.
The existing C++ path shares K/V across pairs of GQA heads but processes five
verification rows separately; M04 packs those rows for shared historical KV
loading. The two earlier simple shared-KV C++ experiments regressed the full
request, so the next candidate must follow the packed/tiled M04 structure and
pass the same-input output and full P4096/O1024 gates.

The separate real P4096/O32 profile ruled out the FP8 KV cache writer as a
material next target: 180 writer events totaled 34.74 ms on GPU, of which
the first 17 prefill writes used 33.68 ms. Even eliminating the entire prefill
writer would save less than 0.84% of the observed 4.014 s first-token time;
the other 163 writes totaled about 1.06 ms. This is an upper bound on that
profile, not an accepted model-level optimization.

The two source captures (~16 MiB each) remain outside Git in
`/tmp/b70_python_prequant` and `/tmp/b70_cpp_prequant`:

| Source file | SHA-256 |
| --- | --- |
| Python K | `37074f3c014303fdc5d16a557ac721f28075b24a9259bbcf07847426982403af` |
| Python V | `e4cdfbc03fff3126849c739b3f03946d06933ba5d3076d2b767d7d1fc4f5bdad` |
| C++ K | `0155ef03f6c69018680bd1ecdefe7aa2ea66bf157f75af1276a45feb8ef8e328` |
| C++ V | `748dcc9b4d89c5b7511d77282ca26288b1bf2f41b6fdd9e00d42d2fa91fc8a5a` |

Next isolate the input to this first full-attention projection after the
three GDN layers and compare the GPTQ QKV projection and Q/K norm/RoPE stages
on identical inputs. The FP8 writer and M04/Split-K arithmetic need no
correctness change based on this capture. W5's verification throughput,
graphs and broader sampled-quality gate remain open.

## Torch-free Xe2 M04-like verification candidate

A narrowly adapted native copy of the pinned Xe2 shared-KV policy now builds
with C++/SYCL-TLA, without Torch or Python at inference time. It accepts the
27B native FP8 page layout for Q2–Q5, 1664-token physical pages and unit
scales. The C++ adapter packs Q for shared historical K/V loading and unpacks
the output. It is **off by default**; set `VT_XPU_XE2_VERIFY=1` to try it under
`VT_XPU_ATTENTION=auto`. Unsupported shapes retain the existing route.

On the captured real Python Q5 input with the same logical FP8 K/V bytes, the
focused B70 replay passed 34/34 assertions. Native output differs in 5 of
30,720 FP16 entries from Python M04; relative RMS is `5.21419e-06` and
maximum absolute difference is `0.000244141`. A warm 20-call synchronized
host median was **0.213176 ms** with the actual VT cache layout, versus
**1.999 ms** for the existing Split-K replay and approximately 0.141–0.159 ms
for the Python M04 adapter. These are operator-only timings. The C++ public
KV layout remains head-contiguous and was not changed for the experiment.

The P4096/O32 sampled eager MTP4 full-model smoke passed 50/50 assertions,
emitted the same 32 token IDs as the earlier route, and kept 23/29 draft
acceptance. Its separate O1 TTFT was 4.23372 s, full O32 wall 4.73164 s,
and derived short-request decode 62.2585 tokens/s. This short measurement is
for correctness and route smoke, not the promotion benchmark. The full
P4096/O1024 C++ A/B passed 50/50 assertions on each route: default Split-K
49.3827 versus optional Xe2 verifier 62.7176 derived emitted decode
tokens/s, with O1024 wall 24.959 versus 20.5619 s. The prompt hash matched;
the output hashes and acceptance counts differed (default 770/1012 accepted,
candidate 767/1024). This single-run +27.0% candidate is **not promoted**:
the differing sampled continuation needs same-prefix Python target-logit
comparison, and Q2–Q5 route coverage remains incomplete. See
`checkpoint_review_20260928.md` for the exact full-request figures and
remaining W0–W8 work.
