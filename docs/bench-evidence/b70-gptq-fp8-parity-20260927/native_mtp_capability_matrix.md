# W0 native MTP inventory (2026-09-28)

This inventory checks the new ChatGPT Pro W0 brief against the
`b70-gptq-int4` checkpoint `077b4c4bd` and records the first W3 correctness
increments. It is not a claim of production MTP support. Python is an external oracle;
the intended inference runtime remains C++ plus native GPU kernels.

## Two reference configurations

| | Frozen non-MTP oracle | Production MTP oracle |
| --- | --- | --- |
| Source | Python vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` plus archived local patch | Local deployment repository `intel-b70-qwen38-vllm` at `906be5bbcdd7756b514b32e3043fb1060bf867cd` |
| Runtime recipe | P4096/D64 worker-path diagnostic; eager; FP8 E4M3 KV; 1600 effective page; no prefix or MTP | README/Dockerfile: vLLM 0.30.0+xpu, XPU kernels 0.1.15.4, MTP4, separate INT4 draft head and five INT4 draft linears, full vocabulary, graph, FP8 KV, prefix reuse, at most C4, 6656 max batched tokens, 180 W card cap |
| Sampler | Teacher-forced / greedy quality and timing gates | Temperature 1, top-p .95, top-k 20, 1024 output tokens in the fixed-length source-review benchmark |
| Provenance limit | Previous `baseline_manifest.json` and `timing_contract.md`; installed wheel's exact native source unavailable | Local README and Dockerfile match the reviewed Git commit; neither proves the image currently running or its active dispatch. The service was inactive at inventory time. |

Local production-recipe SHA-256: Dockerfile
`0bf6c5b1bcc6a03c452865c5bbe46c609e1605a97c0195c66332348c2b7ac5c0`,
README `bd52fe6aad9826ff3ce1a96ca18bc9ca99c9e9d40502f5eb96e82532b5d7e3e8`.
The Dockerfile pins XPU wheel SHA-256
`4ec262f7afdd07c62defc8286dc9f357569627b996c09b410f7972b653696fe7`;
this is a recipe pin, not a hash freshly read from a running container.

## Model tensor inventory

The local `models-legacy/Qwen3.8-27B-GPTQ-G128` safetensors index SHA-256 is
`d511a969b32a16f1890326dadd4d04039779cebce7458b8e5d2ce7f3ce2550ed`.
Shard 5's safetensors header contains 15 `mtp.*` tensors, all BF16, for one
predictor layer; their exact names, shapes and index-to-shard mapping are in
`model_tensor_inventory.json`. The index and header were read without loading
weight payloads. The local config contains `mtp_num_hidden_layers=1` and
`vocab_size=248320`. The model revision in the production README is declared
as `a47b0c6f0d756bc394c4cc629d5b0ded1acc7001`; the local header/index
alone do not independently attest to that revision.

## Native C++ capability and blocking contracts

| Area | Current source evidence | W3/W5 requirement |
| --- | --- | --- |
| Generic draft | `qwen3_5_mtp.cpp` loads BF16 MTP tensors; a focused C++ load of the actual GPTQ checkpoint's shard 5 passes 6/6 shape/layer assertions. The real draft weights with FP16 activations and FP16 target-shared embedding/head complete native XPU paged forward and full-vocabulary logits with no reference-tier hit. The input tap in this gate is synthetic FP16. | Validate proposal tokens and draft KV continuation against the Python oracle with an actual target-generated tap. |
| Target tap | GPTQ target activations and hidden tap are FP16. `ForwardPaged` now keeps the XPU draft activations, hidden carry and logits input FP16, with no per-forward FP16→BF16 buffer copy. BF16 remains the generic CPU/draft route. The production Python speculator allocates its hidden buffer in model dtype (FP16 here); its INT4 draft linear converts its input to FP16 only if needed. | Real GPTQ target plus loaded draft quality gate; capture actual production intermediate dtypes before claiming complete reference parity. |
| Speculative recurrent ops | Both were absent from XPU registration at `077b4c4bd`. The W3 step now adds native XPU conv and GDN kernels; focused CPU/XPU tests pass, including 27B FP16/GDN dimensions. See `w3_native_xpu_spec_recurrent.md`. | Full-model accepted-prefix/rollback, lifecycle and batch qualification remain. |
| Graph | `ForwardQwen3_5Dense` returns through eager `ForwardDeviceTap` when a hidden tap is requested. | Qualify eager MTP first, then persistent graph-compatible post-final-norm tap. |
| Draft weights/head | GPTQ/XPU MTP load now packs BF16 FC, merged QKV, attention output, merged gate/up and down into symmetric INT4/G128, plus a separate FP16-source INT4 draft head. Native oneDNN W4A16 consumes the packed owners; the target head stays FP16. Focused pack test 6/6 and real-checkpoint XPU draft test 29/29 pass. The synthetic tap produced different dense vs packed draft top-1 (14 vs 71262), while the head-only pack kept top-1 at 14; this is a quality finding, not an acceptance-rate result. | Compare C++ packed outputs against the installed Python INT4 draft on identical actual target taps; measure acceptance and decode speed. |
| Verification sampler | `src/vllm/v1/spec_decode/rejection_sampler.cpp` and `include/.../rejection_sampler.h` implement the greedy acceptance path; sampled residual rejection is explicitly deferred. | Exact sampled proposal/target distributions and residual correction, plus request-local RNG. |
| Prefix | `runner.cpp` rejects `recurrent_prefix_snapshots_ && spec_on()`. | Keep prefix off for W3; design committed-only target/draft snapshot publication for W7. |
| Attention | Native FP8 attention supports qualified non-speculative cases; no dedicated shared-KV 2–5-row verification route. | Correct per-row causal masking and measured verification kernel. |
| Platform | `src/vllm/platforms/xpu.cpp` documents a native text path without speculative draft execution. | Enable only after the above contracts pass on B70. |

W1's independent eager-oracle and first-divergence result are in
`w1_graph_case5_isolation.md`. Graph remains opt-in with a batch cap of four.
The target and XPU draft now carry FP16 activations without a per-token
FP16/BF16 bridge. The GPTQ draft linears and its separate head are packed once
at load time. A real-weight draft forward, two-step FP16-KV continuation and
FP8 E4M3 KV write/read pass on XPU; FP16-KV continuation equals a causal
two-row forward (max absolute difference 0), and the FP8-vs-FP16 KV maximum
difference is 0.425781 on this synthetic tap. The recurrent ops and draft
forward alone do not enable MTP1. The next W3 work is
target-generated hidden input, Q=2 target verification
and committed-state/greedy-token comparison. The
generic draft retains its existing dense head path for non-GPTQ models.
