#!/usr/bin/env python3
"""Derive real K/V subprojection fixtures from a captured fused QKV route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file, save_file


REVISION = "a47b0c6f0d756bc394c4cc629d5b0ded1acc7001"
PROMPT_LENGTHS = (1, 16, 256)


def sha256(tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().view(-1).numpy().tobytes()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    args = parser.parse_args()
    capture_dir = args.capture_dir.resolve()

    weight_path = capture_dir / "attention_qkv_weights.safetensors"
    weights = load_file(str(weight_path), device="cpu")
    qweight = weights["qweight_nt_int32"]
    scales = weights["scales_f16"]
    zero_points = weights["effective_zero_point_i8"]
    if tuple(qweight.shape) != (14336, 640):
        raise SystemExit(f"unexpected merged QKV qweight shape: {tuple(qweight.shape)}")
    if tuple(scales.shape) != (40, 14336) or zero_points.tolist() != [8]:
        raise SystemExit("unexpected merged QKV scales or effective zero point")

    # Q includes the per-head gate: 24 heads * 256 channels * 2. The remaining
    # K and V segments each have 4 KV heads * 256 channels.
    split = {
        "attention_k_proj": (12288, 13312),
        "attention_v_proj": (13312, 14336),
    }
    for family, (first, last) in split.items():
        out_weight = capture_dir / f"{family}_weights.safetensors"
        weight_json = out_weight.with_suffix(".json")
        if out_weight.exists() or weight_json.exists():
            raise SystemExit(f"refusing to overwrite {out_weight}")
        wq_slice = qweight[first:last].contiguous()
        scale_slice = scales[:, first:last].contiguous()
        save_file(
            {
                "qweight_nt_int32": wq_slice,
                "scales_f16": scale_slice,
                "effective_zero_point_i8": zero_points.contiguous(),
            },
            str(out_weight),
        )
        weight_json.write_text(
            json.dumps(
                {
                    "fixture_kind": "post_load_fused_weight_component",
                    "source_route": "attention_qkv",
                    "source_weights_file": weight_path.name,
                    "checkpoint_revision": REVISION,
                    "matrix_prefix": "language_model.model.layers.3.self_attn.qkv_proj",
                    "component": family,
                    "merged_output_rows": [first, last],
                    "shape": {"K": 5120, "N": last - first},
                    "group_size": 128,
                    "qweight_sha256": sha256(wq_slice),
                    "scales_sha256": sha256(scale_slice),
                    "effective_zero_point": [8],
                    "g_idx": "omitted at runtime; desc_act=false",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        for m in PROMPT_LENGTHS:
            source_path = capture_dir / f"attention_qkv_m{m}.safetensors"
            source_meta = json.loads(source_path.with_suffix(".json").read_text())
            source = load_file(str(source_path), device="cpu")
            output = source["output_fp16"][:, first:last].contiguous()
            out_path = capture_dir / f"{family}_m{m}.safetensors"
            out_meta = out_path.with_suffix(".json")
            if out_path.exists() or out_meta.exists():
                raise SystemExit(f"refusing to overwrite {out_path}")
            save_file(
                {
                    "activation_fp16": source["activation_fp16"].contiguous(),
                    "output_fp16": output,
                },
                str(out_path),
            )
            out_meta.write_text(
                json.dumps(
                    {
                        "fixture_kind": "captured_fused_attention_component",
                        "source_route": "attention_qkv",
                        "source_operation_file": source_path.name,
                        "source_weights_file": weight_path.name,
                        "checkpoint_revision": REVISION,
                        "component": family,
                        "merged_output_rows": [first, last],
                        "matrix_shape": {"M": m, "K": 5120, "N": last - first},
                        "group_size": 128,
                        "request_context": source_meta["request_context"],
                        "activation_sha256": sha256(source["activation_fp16"]),
                        "output_sha256": sha256(output),
                        "scope": "Actual no-MTP model call; component is a row slice of fused QKV.",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            print(f"wrote {out_path.name} and {out_weight.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
