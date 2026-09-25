"""Opt-in layer capture for the pinned B70 GPTQ Python reference.

Loaded only when GPTQ_CAPTURE_DIR is set. It records selected W4A16 and dense
FP16 module calls after the reference implementation has produced its output.
The hook deliberately synchronizes only those selected calls and is not a
benchmark mode.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
import threading
from pathlib import Path


_ROOT = os.environ.get("GPTQ_CAPTURE_DIR")
if _ROOT:
    _ROOT_PATH = Path(_ROOT)
    _CONTEXT_PATH = Path(
        os.environ.get("GPTQ_CAPTURE_CONTEXT_FILE", str(_ROOT_PATH / "current_prompt.json"))
    )
    _LOCK = threading.Lock()
    _CAPTURED: set[tuple[str, int]] = set()
    _ROUTED: set[tuple] = set()
    _PATCHED: set[str] = set()
    _CATEGORIES = set(filter(None, os.environ.get("GPTQ_CAPTURE_CATEGORIES", "").split(",")))

    def _wanted(category: str) -> bool:
        if category in {"gdn_out", "gdn_norm", "gdn_metadata"}:
            return category in _CATEGORIES
        return not _CATEGORIES or category in _CATEGORIES

    def _category(prefix: str) -> str | None:
        if ".linear_attn.in_proj_qkvz" in prefix:
            return "gdn_qkvz"
        if ".linear_attn.out_proj" in prefix:
            return "gdn_out"
        if ".mlp.gate_up_proj" in prefix:
            return "mlp_gate_up"
        if ".mlp.down_proj" in prefix:
            return "mlp_down"
        if ".self_attn.qkv_proj" in prefix:
            return "attention_qkv"
        if ".linear_attn.in_proj_ba" in prefix:
            return "dense_ba"
        if prefix.endswith("lm_head") or ".lm_head." in prefix:
            return "dense_lm_head"
        return None

    def _context() -> dict:
        try:
            return json.loads(_CONTEXT_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _cpu(tensor):
        return tensor.detach().contiguous().cpu()

    def _sha256(tensor) -> str:
        raw = _cpu(tensor).view(__import__("torch").uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def _write_json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def _write_routes(record: dict) -> None:
        key = (
            record["prefix"],
            record["kernel_class"],
            record["input_dtype"],
            record["output_dtype"],
            str(record["input_shape"]),
        )
        with _LOCK:
            if key in _ROUTED:
                return
            _ROUTED.add(key)
            with (_ROOT_PATH / "routes.jsonl").open("a") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")

    def _save_safetensors(path: Path, tensors: dict, metadata: dict) -> None:
        from safetensors.torch import save_file

        if path.exists() or path.with_suffix(".json").exists():
            raise RuntimeError(f"refusing to overwrite capture: {path}")
        save_file({key: _cpu(value) for key, value in tensors.items()}, str(path))
        _write_json(path.with_suffix(".json"), metadata)

    def _capture_quantized(kernel, layer, x, bias, output) -> None:
        prefix = str(getattr(layer, "prefix", ""))
        category = _category(prefix)
        if category not in {"gdn_qkvz", "gdn_out", "mlp_gate_up", "mlp_down", "attention_qkv"} or not _wanted(category):
            return
        context = _context()
        request_id = context.get("request_id")
        if not request_id:
            return

        import torch

        activation = x.reshape(-1, x.shape[-1])
        m, k = map(int, activation.shape)
        n = int(output.shape[-1])
        w_q, w_s, w_zp, w_gidx = kernel._get_weight_params(layer)
        record = {
            "category": category,
            "prefix": prefix,
            "kernel_class": type(kernel).__name__,
            "torch_operator": str(torch.ops._xpu_C.int4_gemm_w4a16.default._schema),
            "group_size": int(kernel.config.group_size),
            "input_shape": [m, k],
            "output_shape": list(map(int, output.shape)),
            "input_dtype": str(activation.dtype),
            "weight_storage_dtype": str(w_q.dtype),
            "weight_storage_shape": list(map(int, w_q.shape)),
            "scales_dtype": str(w_s.dtype),
            "scales_shape": list(map(int, w_s.shape)),
            "zero_point_dtype": str(w_zp.dtype),
            "zero_point_values": _cpu(w_zp).reshape(-1).tolist(),
            "g_idx_present": w_gidx is not None,
            "output_dtype": str(output.dtype),
            "request_id": request_id,
        }
        _write_routes(record)
        capture_key = (category, m)
        if m not in {1, 16, 256}:
            return
        with _LOCK:
            if capture_key in _CAPTURED:
                return
            _CAPTURED.add(capture_key)

        stem = f"{category}_m{m}"
        weight_path = _ROOT_PATH / f"{category}_weights.safetensors"
        if not weight_path.exists():
            weight_tensors = {
                "qweight_nt_int32": w_q,
                "scales_f16": w_s,
                "effective_zero_point_i8": w_zp,
            }
            if w_gidx is not None:
                weight_tensors["g_idx_i32"] = w_gidx
            _save_safetensors(
                weight_path,
                weight_tensors,
                {
                    "category": category,
                    "prefix": prefix,
                    "layout": "qweight [N,K/8] contiguous; scales as post-load",
                    "checkpoint_revision": "a47b0c6f0d756bc394c4cc629d5b0ded1acc7001",
                },
            )
        tensors_path = _ROOT_PATH / f"{stem}.safetensors"
        tensors = {"activation_fp16": activation, "output_fp16": output}
        if bias is not None:
            tensors["bias_fp16"] = bias
        metadata = {
            **record,
            "fixture_kind": "captured_reference_model_operation",
            "request_context": context,
            "weights_file": weight_path.name,
            "activation_sha256": _sha256(activation),
            "output_sha256": _sha256(output),
            "scope": "Captured selected operation only; no full-model parity claim.",
        }
        _save_safetensors(tensors_path, tensors, metadata)

    def _capture_dense(method, layer, x, output, capture_site="linear_apply") -> None:
        prefix = str(getattr(layer, "prefix", ""))
        category = _category(prefix)
        weight = layer.weight
        if category is None and tuple(map(int, weight.shape)) == (248320, 5120):
            category = "dense_lm_head"
        if category not in {"dense_ba", "dense_lm_head"} or not _wanted(category):
            return
        context = _context()
        request_id = context.get("request_id")
        if not request_id:
            return

        import torch

        activation = x.reshape(-1, x.shape[-1])
        m, k = map(int, activation.shape)
        n = int(weight.shape[0])
        record = {
            "category": category,
            "prefix": prefix,
            "kernel_class": type(method).__name__,
            "operator": f"{type(method).__name__}.apply",
            "capture_site": capture_site,
            "input_shape": [m, k],
            "weight_shape": list(map(int, weight.shape)),
            "output_shape": list(map(int, output.shape)),
            "input_dtype": str(activation.dtype),
            "weight_dtype": str(weight.dtype),
            "output_dtype": str(output.dtype),
            "request_id": request_id,
        }
        _write_routes(record)
        prompt_key = os.environ.get("GPTQ_CAPTURE_PROMPT_KEY") == "1"
        prompt_length = int(context.get("prompt_length", 0))
        capture_key = (category, m, prompt_length if prompt_key else 0)
        if m not in {1, 16, 256}:
            return
        with _LOCK:
            if capture_key in _CAPTURED:
                return
            _CAPTURED.add(capture_key)

        stem = (f"{category}_p{prompt_length}_m{m}" if prompt_key
                else f"{category}_m{m}")
        tensors = {"activation_fp16": activation, "output_reference": output}
        if category == "dense_ba":
            tensors["weight_fp16_nk"] = weight
            weight_capture = "full dense weight saved as [N,K]"
        elif os.environ.get("GPTQ_CAPTURE_HEAD_OUTPUT_ONLY") == "1":
            weight_capture = "omitted for the bounded full-model comparison"
        else:
            # The ignored local evidence can hold the 2.54-GB dense checkpoint
            # tensor. Keeping all rows lets the native GPTQ-01 probe exercise
            # the actual [5120, 248320] head rather than only a slice.
            tensors["weight_full_fp16_nk"] = weight
            weight_capture = "full dense head copied from the pinned checkpoint"
        tensors_path = _ROOT_PATH / f"{stem}.safetensors"
        metadata = {
            **record,
            "fixture_kind": "captured_reference_model_operation",
            "request_context": context,
            "weight_capture": weight_capture,
            "weight_sha256": _sha256(weight) if category == "dense_ba" or
                os.environ.get("GPTQ_CAPTURE_HEAD_OUTPUT_ONLY") != "1" else None,
            "activation_sha256": _sha256(activation),
            "output_sha256": _sha256(output),
            "scope": "Captured selected operation only; no full-model parity claim.",
        }
        _save_safetensors(tensors_path, tensors, metadata)

    def _capture_gdn_norm(layer, x, z, output) -> None:
        if z is None or not _wanted("gdn_norm"):
            return
        context = _context()
        if not context.get("request_id") or x.numel() not in {48 * 128, 16 * 48 * 128}:
            return
        m = x.numel() // (48 * 128)
        capture_key = ("gdn_norm", m)
        with _LOCK:
            if capture_key in _CAPTURED:
                return
            _CAPTURED.add(capture_key)
        _save_safetensors(
            _ROOT_PATH / f"gdn_norm_m{m}.safetensors",
            {"core_fp16": x, "gate_fp16": z, "output_fp16": output,
             "weight_fp16": layer.weight},
            {"category": "gdn_norm", "request_context": context,
             "eps": layer.eps, "activation": layer.activation,
             "norm_before_gate": layer.norm_before_gate},
        )

    def _capture_gdn_metadata(layer) -> None:
        if not _wanted("gdn_metadata"):
            return
        context = _context()
        if not context.get("request_id") or not str(layer.prefix).endswith("layers.0.linear_attn"):
            return
        m = int(context.get("prompt_length", 0))
        capture_key = ("gdn_metadata", m)
        with _LOCK:
            if capture_key in _CAPTURED:
                return
            _CAPTURED.add(capture_key)
        from vllm.forward_context import get_forward_context
        metadata = get_forward_context().attn_metadata[layer.prefix]
        slots = metadata.non_spec_state_indices_tensor
        slot = int(_cpu(slots).reshape(-1)[0])
        state = layer.kv_cache[1][slot]
        conv_state = layer.kv_cache[0][slot]
        initial = metadata.has_initial_state
        _write_json(
            _ROOT_PATH / f"gdn_metadata_m{m}.json",
            {"request_context": context,
             "has_initial_state": None if initial is None else _cpu(initial).reshape(-1).tolist(),
             "state_slot": slot, "state_shape": list(state.shape),
             "state_abs_max_before": float(state.abs().max().item()),
             "state_first_before": _cpu(state.reshape(-1)[:8]).tolist(),
             "conv_abs_max_before": float(conv_state.abs().max().item()),
             "conv_first_before": _cpu(conv_state.reshape(-1)[:8]).tolist()},
        )

    def _install_patches() -> None:
        gdn_module = sys.modules.get(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
        )
        if _wanted("gdn_metadata") and gdn_module is not None and "gdn_metadata" not in _PATCHED:
            gdn_class = getattr(gdn_module, "QwenGatedDeltaNetAttention", None)
            if gdn_class is not None:
                original = gdn_class.forward_xpu

                def forward_xpu(self, hidden_states):
                    _capture_gdn_metadata(self)
                    return original(self, hidden_states)

                gdn_class.forward_xpu = forward_xpu
                _PATCHED.add("gdn_metadata")

        layernorm_module = sys.modules.get("vllm.model_executor.layers.layernorm")
        if _wanted("gdn_norm") and layernorm_module is not None and "gdn_norm" not in _PATCHED:
            norm_class = getattr(layernorm_module, "RMSNormGated", None)
            if norm_class is not None:
                original = norm_class.forward_xpu

                def forward_xpu(self, x, z=None):
                    output = original(self, x, z)
                    _capture_gdn_norm(self, x, z, output)
                    return output

                norm_class.forward_xpu = forward_xpu
                _PATCHED.add("gdn_norm")

        xpu_module = sys.modules.get(
            "vllm.model_executor.kernels.linear.mixed_precision.xpu"
        )
        if xpu_module is not None and "xpu" not in _PATCHED:
            kernel_class = getattr(xpu_module, "XPUwNa16LinearKernel", None)
            if kernel_class is not None:
                original = kernel_class.apply_weights

                def apply_weights(self, layer, x, bias=None):
                    output = original(self, layer, x, bias)
                    _capture_quantized(self, layer, x, bias, output)
                    return output

                kernel_class.apply_weights = apply_weights
                _PATCHED.add("xpu")

        linear_module = sys.modules.get("vllm.model_executor.layers.linear")
        if linear_module is not None and "linear" not in _PATCHED:
            method_class = getattr(linear_module, "UnquantizedLinearMethod", None)
            if method_class is not None:
                original = method_class.apply

                def apply(self, layer, x, bias=None):
                    output = original(self, layer, x, bias)
                    _capture_dense(self, layer, x, output, "unquantized_linear_method")
                    return output

                method_class.apply = apply
                _PATCHED.add("linear")

        logits_module = sys.modules.get(
            "vllm.model_executor.layers.logits_processor"
        )
        if logits_module is not None and "logits" not in _PATCHED:
            processor_class = getattr(logits_module, "LogitsProcessor", None)
            if processor_class is not None:
                original = processor_class._apply_head

                def apply_head(self, lm_head, hidden_states, embedding_bias):
                    output = original(self, lm_head, hidden_states, embedding_bias)
                    _capture_dense(
                        lm_head.quant_method,
                        lm_head,
                        hidden_states,
                        output,
                        "logits_processor",
                    )
                    return output

                processor_class._apply_head = apply_head
                _PATCHED.add("logits")

    _original_import = builtins.__import__

    def _capture_import(name, globals=None, locals=None, fromlist=(), level=0):
        result = _original_import(name, globals, locals, fromlist, level)
        _install_patches()
        return result

    builtins.__import__ = _capture_import
