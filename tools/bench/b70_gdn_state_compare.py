"""Compare one real 4K GDN layer captured by the pinned Python and C++ runs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def stats(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")
    a = reference.astype(np.float64).ravel()
    b = candidate.astype(np.float64).ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite GDN values")
    diff = a - b
    return {
        "elements": int(a.size),
        "rms_error": float(np.sqrt(np.mean(diff * diff))),
        "relative_rms_error": float(np.linalg.norm(diff) / max(np.linalg.norm(a), 1e-30)),
        "max_abs_error": float(np.max(np.abs(diff))),
        "exact_fraction": float(np.mean(a == b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cpp-dir", type=Path, required=True)
    parser.add_argument("--compact-python", type=Path)
    parser.add_argument("--thresholds", type=Path)
    args = parser.parse_args()
    data = torch.load(args.python, map_location="cpu", weights_only=True)
    meta = data["metadata"]
    slot = int(meta.get("state_slot", meta["non_spec_state_indices_tensor"][0]))
    if tuple(meta["non_spec_query_start_loc"].tolist()) != (0, 4096):
        raise ValueError("expected one unchunked 4K prefill")
    for phase in ("before", "after"):
        for name in ("conv_state", "ssm_state"):
            tensor = data[phase][name]
            if tensor.ndim > 3 and name == "ssm_state":
                data[phase][name] = tensor[slot].clone()
            elif tensor.ndim > 2 and name == "conv_state":
                data[phase][name] = tensor[slot].clone()
    meta["state_slot"] = slot
    if args.compact_python:
        torch.save(data, args.compact_python)

    def cxx(name: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
        path = args.cpp_dir / f"s-1_l0_{name}.bin"
        arr = np.fromfile(path, dtype=dtype)
        if arr.size != int(np.prod(shape)):
            raise ValueError(f"unexpected {path} length: {arr.size}")
        return arr.reshape(shape)

    # The focused C++ one-request harness uses slot 0; the Python scheduler
    # reserves slot 0 and placed this request in slot 1. Compare live rows.
    cpp_slot = 0
    has_initial_state = bool(meta["has_initial_state"][0])
    result = {"python_state_slot": slot, "cpp_state_slot": cpp_slot,
              "python_projected_shape": meta["projected_qkvz_shape"],
              "has_initial_state": has_initial_state,
              "initial_cache_bytes_ignored": not has_initial_state}
    for phase in ("before", "after"):
        if phase == "before" and not has_initial_state:
            continue
        py_ssm = data[phase]["ssm_state"].numpy()
        cpp_ssm = cxx(f"ssm_state_{phase}", "<f4", (1, 48, 128, 128))[0]
        result[f"ssm_{phase}"] = stats(py_ssm, cpp_ssm)
        py_conv = data[phase]["conv_state"].numpy().T
        cpp_conv = cxx(f"conv_cache_{phase}", "<f2", (2, 10240, 3))[cpp_slot]
        result[f"conv_{phase}"] = stats(py_conv, cpp_conv)
    py_out = data["after"]["core_out"].numpy()
    cpp_out = cxx("core_out", "<f2", (4096, 48, 128))
    result["core_out"] = stats(py_out, cpp_out)
    if args.thresholds:
        limits = json.loads(args.thresholds.read_text(encoding="utf-8"))
        failures = []
        for stage in ("core_out", "ssm_after", "conv_after"):
            for name, bound in limits[stage].items():
                metric = name.removesuffix("_max").removesuffix("_min")
                value = result[stage][metric]
                if name.endswith("_max") and value > bound or (
                    name.endswith("_min") and value < bound
                ):
                    failures.append(f"{stage}.{name}: {value} vs {bound}")
        result["quality_gate"] = "pass" if not failures else "fail"
        result["quality_failures"] = failures
    print(json.dumps(result, indent=2))
    if result.get("quality_gate") == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
