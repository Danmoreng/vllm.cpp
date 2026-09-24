#!/usr/bin/env python3
"""Export one B70 GPTQ W4A16 reference operation from the pinned checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_tensor(model_dir: Path, shard: str, name: str) -> torch.Tensor:
    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name).contiguous()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise SystemExit(f"Refusing to overwrite fixture: {output} or {sidecar}")
    if not torch.xpu.is_available() or torch.xpu.device_count() != 1:
        raise SystemExit("Expected exactly one available Intel XPU device")

    source_lines = (model_dir / "SOURCE-REVISION.txt").read_text().splitlines()
    config = json.loads((model_dir / "config.json").read_text())
    text_config = config.get("text_config", config)
    quant_config = config.get(
        "quantization_config", text_config.get("quantization_config", {})
    )
    expected_source = [
        "mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16",
        "a47b0c6f0d756bc394c4cc629d5b0ded1acc7001",
    ]
    if source_lines != expected_source:
        raise SystemExit(f"Unexpected checkpoint source pin: {source_lines!r}")
    expected_quant = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "lm_head": False,
    }
    for key, expected in expected_quant.items():
        if quant_config.get(key) != expected:
            raise SystemExit(
                f"Checkpoint contract mismatch for {key}: "
                f"{quant_config.get(key)!r} != {expected!r}"
            )
    if config.get("dtype", text_config.get("dtype")) != "float16":
        raise SystemExit("Pinned checkpoint must declare float16 model dtype")

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    names = {suffix: f"{args.prefix}.{suffix}" for suffix in ("qweight", "scales")}
    for name in names.values():
        if name not in weight_map:
            raise SystemExit(f"Missing checkpoint tensor: {name}")
    shard = weight_map[names["qweight"]]
    if weight_map[names["scales"]] != shard:
        raise SystemExit("qweight and scales are not in the same shard")

    qweight_disk = load_tensor(model_dir, shard, names["qweight"])
    scales = load_tensor(model_dir, shard, names["scales"])
    if qweight_disk.dtype != torch.int32 or qweight_disk.ndim != 2:
        raise SystemExit(
            f"Unexpected qweight dtype/shape: {qweight_disk.dtype} "
            f"{tuple(qweight_disk.shape)}"
        )
    k, n = qweight_disk.shape[0] * 8, qweight_disk.shape[1]
    if scales.shape != (k // 128, n) or scales.dtype != torch.float16:
        raise SystemExit(
            f"Unexpected scales dtype/shape: {scales.dtype} {tuple(scales.shape)}"
        )
    if (k, n) != (5120, 1024):
        raise SystemExit(
            f"This first fixture is pinned to K=5120, N=1024; got K={k}, N={n}"
        )

    bias_name = f"{args.prefix}.bias"
    bias = (
        load_tensor(model_dir, weight_map[bias_name], bias_name)
        if bias_name in weight_map
        else None
    )
    if bias is not None and (bias.dtype != torch.float16 or bias.shape != (n,)):
        raise SystemExit(f"Unexpected bias dtype/shape: {bias.dtype} {tuple(bias.shape)}")

    import vllm_xpu_kernels  # noqa: F401  # registers the installed _xpu_C operator
    from vllm.model_executor.kernels.linear import (
        MPLinearLayerConfig,
        choose_mp_linear_kernel,
    )
    from vllm.scalar_type import scalar_types

    layer_config = MPLinearLayerConfig(
        full_weight_shape=(k, n),
        partition_weight_shape=(k, n),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )
    kernel_cls = choose_mp_linear_kernel(layer_config)
    if kernel_cls.__name__ != "XPUwNa16LinearKernel":
        raise SystemExit(f"Unexpected reference kernel selection: {kernel_cls.__name__}")

    layer = torch.nn.Module()
    layer.register_parameter(
        "qweight",
        torch.nn.Parameter(qweight_disk.to("xpu"), requires_grad=False),
    )
    layer.register_parameter(
        "scales",
        torch.nn.Parameter(scales.to("xpu"), requires_grad=False),
    )
    if bias is not None:
        layer.register_parameter(
            "bias",
            torch.nn.Parameter(bias.to("xpu"), requires_grad=False),
        )
    kernel = kernel_cls(layer_config, "qweight", "scales", "qzeros", "g_idx")
    kernel.process_weights_after_loading(layer)
    if layer.qweight.shape != (n, k // 8) or not layer.qweight.is_contiguous():
        raise SystemExit("Unexpected post-load qweight layout")
    if layer.qzeros.dtype != torch.int8 or layer.qzeros.tolist() != [8]:
        raise SystemExit("Reference did not install the expected scalar effective zero point 8")
    if layer.g_idx is not None:
        raise SystemExit("Reference unexpectedly retained g_idx for desc_act=false")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    activation_cpu = torch.randn((1, k), generator=generator, dtype=torch.float32).to(
        torch.float16
    )
    activation = activation_cpu.to("xpu")
    bias_xpu = getattr(layer, "bias", None)

    # First call creates the oneDNN primitive; the saved result is the next call.
    kernel.apply_weights(layer, activation, bias_xpu)
    torch.xpu.synchronize()
    output_xpu = kernel.apply_weights(layer, activation, bias_xpu)
    torch.xpu.synchronize()
    if output_xpu.dtype != torch.float16 or tuple(output_xpu.shape) != (1, n):
        raise SystemExit(
            f"Unexpected output dtype/shape: {output_xpu.dtype} "
            f"{tuple(output_xpu.shape)}"
        )
    if not torch.isfinite(output_xpu).all().item():
        raise SystemExit("Reference operation returned non-finite values")

    tensors = {
        "activation_fp16": activation_cpu.contiguous(),
        "qweight_nt_int32": layer.qweight.detach().cpu().contiguous(),
        "scales_f16": layer.scales.detach().cpu().contiguous(),
        "effective_zero_point_i8": layer.qzeros.detach().cpu().contiguous(),
        "output_fp16": output_xpu.detach().cpu().contiguous(),
    }
    if bias_xpu is not None:
        tensors["bias_fp16"] = bias_xpu.detach().cpu().contiguous()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output))

    op = torch.ops._xpu_C.int4_gemm_w4a16.default
    runtime_versions = {
        name: importlib.metadata.version(name)
        for name in ("vllm", "vllm-xpu-kernels", "torch")
    }
    metadata = {
        "fixture_kind": "isolated_operator_with_synthetic_activation",
        "scope": "No model forward, token IDs, positions, or full-model quality claim.",
        "checkpoint_repository": source_lines[0],
        "checkpoint_revision": source_lines[1],
        "model_dtype": "float16",
        "quantization": expected_quant,
        "matrix_prefix": args.prefix,
        "matrix_shape": {"M": 1, "K": k, "N": n},
        "disk_qweight": {"dtype": "int32", "shape": list(qweight_disk.shape)},
        "post_load_qweight": {
            "dtype": "int32",
            "shape": list(layer.qweight.shape),
            "physical_layout": "[N, K/8], contiguous",
        },
        "scales": {"dtype": "float16", "shape": list(layer.scales.shape)},
        "runtime_zero_point": {"dtype": "int8", "values": [8]},
        "disk_qzeros_audit": "evidence/local_aux_v1.json; offset 1 verified for all matrices",
        "g_idx": "omitted by reference because desc_act=false; disk values audited separately",
        "bias_present": bias is not None,
        "activation_seed": args.seed,
        "activation_sha256": tensor_sha256(tensors["activation_fp16"]),
        "output_sha256": tensor_sha256(tensors["output_fp16"]),
        "selected_kernel_class": kernel_cls.__name__,
        "torch_operator": str(op._schema),
        "one_dnn_verbose": os.environ.get("DNNL_VERBOSE"),
        "onednn_verbose": os.environ.get("ONEDNN_VERBOSE"),
        "device": torch.xpu.get_device_name(0),
        "runtime_versions": runtime_versions,
        "reference_image_id": args.image_id,
        "fixture_tensors_file": output.name,
    }
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote fixture {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
