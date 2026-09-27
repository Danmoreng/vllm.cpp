# P4 real 4K GDN layer-0 replay and Python state comparison

The opt-in `VT_B70_GDN_CAPTURE` path captured the first GDN layer of the pinned
Qwen3.8-27B GPTQ model during a P4096/O1 eager C++ forward with FP16
activations and FP8 E4M3 KV cache (effective page 1600). The Python V2 worker
ran the same periodic token IDs (`9705ea213a213545`) and captured the first
4K call of the installed `_xpu_C.gdn_attention` operator after model
initialization. It used the unmodified wheel implementation. Neither capture
run is suitable for throughput scoring.

The C++ dump includes real normalized Q/K, V, G/beta, sequence offsets, core
output, FP32 SSM state before/after and FP16 conv cache before/after. The
manifest with dtype, shape and byte lengths is in `p4_real_layer0_manifest.tsv`.
The raw 142 MiB C++ tensors and the compact 54 MiB Python tensor snapshot live
outside Git at `/home/sebastian/LocalLLM/b70-gdn-real-layer-20260927/`.
SHA-256: Python compact snapshot
`f25d3774ff74ccf6b37935e667cf9c4b9be9a763eb36dd0d18e6112c9bee092d`;
C++ Q tensor
`4b7864dbf253bf08be6509d2da5a652fc71d5320aeb36c1905d99adaa472b5e7`;
C++ core output
`d1309fa072e67f28ec2d0b05dae947796d273e60d241d8ef96badc5a03e804b6`.

The focused XPU test replays the real C++ core inputs from fresh state and
reproduces output and final state byte-for-byte. A second replay divides the
same activation sequence at token 2048; its second half starts from the
nonzero state produced by the first half and still reproduces the full output
and final state byte-for-byte. The test passed 45/45 assertions.

`p4_real_layer0_compare.json` contains the cross-engine results. Python used
state slot 1, while the C++ single-request harness used slot 0. Python marked
the fresh request with `has_initial_state=false`; its pre-existing cache bytes
are ignored. The compact snapshot stores logical zero initial tensors instead
of unused allocation contents.
After prefill, the live conv-cache row is bit-identical across engines. The
relative RMS differences are 0.002218 for the FP32 SSM state and 0.001439 for
the FP16 core output. All pass `p4_state_thresholds.json`, frozen before any P4
kernel candidate. These thresholds supplement the existing long-run logit gate;
they do not qualify decode continuation or a deeper layer yet.

The selected C++ core uses chunked FP16 GDN; the Python trace proves the native
XPU GDN route. Exact wheel 0.1.15.4 native source remains unavailable, so this
is a numerical comparison of executed paths rather than proof of identical
kernel code. Next capture a nonempty-state continuation and then test a
structural GDN scheduling candidate against this replay and the model gates.
The compact Python hook was rerun independently; it produced the same state
and output comparison metrics and passed the fixed gate again.
