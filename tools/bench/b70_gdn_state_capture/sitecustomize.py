"""Capture the first armed 4K Python XPU GDN layer for P4 quality analysis.

The driver creates B70_GDN_CAPTURE_ARM after model initialization. The hook is
loaded in the actual V2 worker through PYTHONPATH; it wraps the installed
_xpu_C operator without replacing its implementation. Capture runs are never
throughput measurements.
"""

import os
from pathlib import Path


_out_arg = os.environ.get("B70_GDN_CAPTURE_OUT")
if _out_arg:
    import torch
    import vllm._xpu_ops  # Registers the installed XPU custom operator.

    _out = Path(_out_arg)
    _out.mkdir(parents=True, exist_ok=True)
    _arm = Path(os.environ["B70_GDN_CAPTURE_ARM"])
    _native = torch.ops._xpu_C.gdn_attention
    _captured = False

    def _cpu(tensor):
        return tensor.detach().cpu().contiguous().clone()

    def _capture_gdn(*args, **kwargs):
        global _captured
        projected = args[2]
        selected = (not _captured and _arm.exists() and
                    projected.shape[0] == 4096 and
                    kwargs.get("num_prefills") == 1 and
                    kwargs.get("num_decodes") == 0)
        if selected:
            _captured = True
            slot = int(kwargs["non_spec_state_indices_tensor"][0].item())
            has_initial_state = bool(kwargs["has_initial_state"][0].item())
            def initial_state(name):
                row = kwargs[name][slot]
                return _cpu(row) if has_initial_state else torch.zeros(
                    tuple(row.shape), dtype=row.dtype, device="cpu")
            before = {
                "conv_state": initial_state("conv_state"),
                "ssm_state": initial_state("ssm_state"),
            }
            metadata = {
                "state_slot": slot,
                "projected_qkvz_shape": tuple(projected.shape),
                "projected_qkvz_dtype": str(projected.dtype),
                "num_prefills": kwargs["num_prefills"],
                "num_decodes": kwargs["num_decodes"],
                "num_actual_tokens": kwargs["num_actual_tokens"],
                "num_k_heads": args[4],
                "num_v_heads": args[5],
                "head_k_dim": args[6],
                "head_v_dim": args[7],
            }
            for name in ("non_spec_state_indices_tensor",
                         "non_spec_query_start_loc", "has_initial_state"):
                value = kwargs.get(name)
                metadata[name] = _cpu(value) if value is not None else None
        result = _native(*args, **kwargs)
        if selected:
            payload = {
                "metadata": metadata,
                "before": before,
                "after": {
                    "conv_state": _cpu(kwargs["conv_state"][slot]),
                    "ssm_state": _cpu(kwargs["ssm_state"][slot]),
                    "core_out": _cpu(args[0]),
                },
            }
            target = _out / "python_layer0_4096.pt"
            temporary = _out / "python_layer0_4096.pt.tmp"
            torch.save(payload, temporary)
            temporary.replace(target)
            print(f"B70_GDN_CAPTURE {target} pid={os.getpid()}", flush=True)
        return result

    torch.ops._xpu_C.gdn_attention = _capture_gdn
