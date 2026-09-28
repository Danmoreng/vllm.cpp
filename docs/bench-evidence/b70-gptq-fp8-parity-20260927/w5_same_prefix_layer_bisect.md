# W5: same-prefix five-row target-layer bisect (2026-09-28)

The verifier-probability probe in `w4_same_prefix_mtp_quality.md` found two
rows above the frozen 2% TV target. To locate where the gap grows, the first
P4096 MTP4 verification call was captured in both engines with the same
prompt hash `4760835697920107937`, first token 271 and four draft tokens
`90979,78,248046,198`. Python's target decoder reported MRoPE positions
`[4096,4097,4098,4099,4100]` in each of its three streams. Both engines
used eager C1, FP8 KV and a 1664-token effective hybrid page.

Temporary probes saved only the five-token FP16 `hidden` and `residual`
outputs at all 64 target decoder-layer boundaries. Python was instrumented
in the installed `Qwen3NextDecoderLayer.forward`; C++ in
`DenseForwardLayers`. The Python probe excluded the separate full-attention
draft layer at index 0, which otherwise overwrites the target layer-0 file.
Both short model requests completed; the C++ probed request emitted the
same 32 IDs as its uninstrumented control. Raw activation arrays remain outside Git under
`/tmp/b70_{cpp,python}_spec_stages`.

For each layer, the table computes relative RMS difference of the combined
stream `hidden + residual` over all five rows and 5120 channels. Python and
C++ values were converted to F32 for comparison, with F64 accumulation of
squared errors.

| Layer | Type | Relative RMS difference |
| ---: | --- | ---: |
| 0 | GDN | 0.0105% |
| 1 | GDN | 0.0161% |
| 2 | GDN | 0.0299% |
| 3 | Full attention | 0.0761% |
| 15 | Full attention | 0.2113% |
| 31 | Full attention | 0.5165% |
| 35 | Full attention | 0.8004% |
| 43 | Full attention | 1.4055% |
| 46 | GDN | 1.4989% |
| 47 | Full attention | 1.9645% |
| 51 | Full attention | 3.2813% |
| 55 | Full attention | 4.7525% |
| 63 | Full attention | 4.8331% |

Across all 16 full-attention layers, the mean increase over the preceding
layer's relative error was `0.001575`, versus `0.000490` for the 47 GDN
transitions. Fifteen of 16 attention transitions increased the error. These
are **propagation measurements**, not isolated per-kernel error: later
layers receive already different inputs.

Additional captures at an early and a late full-attention layer show where
the extra difference appears. Relative RMS is computed against the Python
tensor at each named stage, not against the combined residual stream.

| Layer | Post-input norm | Attention block output | Post-attention norm | MLP output |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.0309% | 0.1103% | 0.2424% | 0.2406% |
| 47 | 1.5750% | 7.8242% | 2.6367% | 3.7114% |

Layer 3 starts from a very close normalized input and shows an increase
through attention. At layer 47 the input is already different, so the much
larger block-output difference cannot yet be called an attention-kernel bug.
The next discriminating test is an operator replay that feeds the **same**
captured norm/Q/K/V and historical FP8 KV to both Python's M04 shared-KV
verification and C++ attention at Q5. The current C++ Split-K path and the
earlier slower experimental shared-KV kernels remain unchanged until such a
same-input quality and complete-cycle performance result exists.

All source instrumentation was removed after capture, and the focused C++
model target was rebuilt successfully. No raw activation file is committed.
