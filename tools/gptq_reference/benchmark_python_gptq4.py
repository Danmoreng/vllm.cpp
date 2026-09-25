#!/usr/bin/env python3
"""Measure the installed Python control on a captured GPTQ-01 fixture."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shlex
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from safetensors import safe_open


def tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as source:
        if name not in source.keys():
            raise SystemExit(f"{path.name} is missing tensor {name}")
        return source.get_tensor(name)


def median(values: list[float]) -> float:
    return statistics.median(values)


def pointer_alignment(tensor: torch.Tensor) -> dict[str, int]:
    address = tensor.data_ptr()
    return {f"mod_{alignment}": address % alignment for alignment in (64, 128, 256)}


class ProfiledQueue:
    def __init__(self, original_stream: torch.xpu.Stream) -> None:
        source = Path(__file__).with_name("profile_queue.cpp")
        library = Path(tempfile.gettempdir()) / "vllmcpp_gptq_profile_queue.so"
        if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
            configured = os.environ.get("CXX")
            compiler = shlex.split(configured) if configured else []
            if not compiler:
                compiler_path = shutil.which("icpx") or "/opt/intel/oneapi/compiler/latest/bin/icpx"
                compiler = [compiler_path]
            command = compiler + [
                "-fsycl",
                "-fPIC",
                "-shared",
                str(source),
                "-o",
                str(library),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)

        self.library = ctypes.CDLL(str(library))
        self.library.gptq4_create_profiled_queue.argtypes = [ctypes.c_void_p]
        self.library.gptq4_create_profiled_queue.restype = ctypes.c_void_p
        self.library.gptq4_destroy_profiled_queue.argtypes = [ctypes.c_void_p]
        self.library.gptq4_submit_timer_marker.argtypes = [ctypes.c_void_p]
        self.library.gptq4_submit_timer_marker.restype = ctypes.c_void_p
        self.library.gptq4_timer_interval_us.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.library.gptq4_timer_interval_us.restype = ctypes.c_double

        self.queue = self.library.gptq4_create_profiled_queue(
            ctypes.c_void_p(original_stream.sycl_queue)
        )
        if not self.queue:
            raise RuntimeError("failed to create a profiling-enabled SYCL queue")
        self.stream = torch.xpu.get_stream_from_external(
            self.queue, torch.xpu.current_device()
        )

    def marker(self) -> ctypes.c_void_p:
        marker = self.library.gptq4_submit_timer_marker(self.queue)
        if not marker:
            raise RuntimeError("failed to enqueue a SYCL timing marker")
        return marker

    def interval_us(self, start: ctypes.c_void_p, end: ctypes.c_void_p) -> float:
        return float(self.library.gptq4_timer_interval_us(start, end))

    def close(self) -> None:
        if self.queue:
            torch.xpu.synchronize()
            self.library.gptq4_destroy_profiled_queue(self.queue)
            self.queue = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gptq4", "dense"))
    parser.add_argument("operation", type=Path)
    parser.add_argument("weights", type=Path, nargs="?")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    operation_path = args.operation.resolve()
    weights_path = (args.weights or args.operation).resolve()
    if args.iterations < 3:
        raise SystemExit("at least three timed iterations are required")
    if not torch.xpu.is_available() or torch.xpu.device_count() != 1:
        raise SystemExit("expected one available Intel XPU")

    activation = tensor(operation_path, "activation_fp16").to("xpu").contiguous()
    try:
        expected_cpu = tensor(operation_path, "output_fp16")
    except SystemExit:
        expected_cpu = tensor(operation_path, "output_reference")
    expected = expected_cpu.to("xpu")
    bias = None
    try:
        bias = tensor(weights_path, "bias_fp16").to("xpu").contiguous()
    except SystemExit:
        pass

    if args.mode == "gptq4":
        import vllm_xpu_kernels  # noqa: F401
        from vllm.model_executor.kernels import linear as _linear_registration  # noqa: F401

        qweight = tensor(weights_path, "qweight_nt_int32").to("xpu").contiguous()
        scales = tensor(weights_path, "scales_f16").to("xpu").contiguous()
        zero_points = tensor(weights_path, "effective_zero_point_i8").to("xpu").contiguous()
        if activation.dtype != torch.float16 or qweight.dtype != torch.int32:
            raise SystemExit("GPTQ fixture must contain F16 activations and I32 packed qweight")
        if tuple(qweight.shape) != (expected.shape[1], activation.shape[1] // 8):
            raise SystemExit("qweight shape does not match activation and output geometry")
        if tuple(scales.shape) != (activation.shape[1] // 128, expected.shape[1]):
            raise SystemExit("scales shape does not match G128 geometry")
        if zero_points.dtype != torch.int8 or zero_points.tolist() != [8]:
            raise SystemExit("expected the pinned symmetric runtime zero point 8")
        operator = torch.ops._xpu_C.int4_gemm_w4a16.default
        input_alignment = {
            "activation": pointer_alignment(activation),
            "weights": pointer_alignment(qweight),
            "scales": pointer_alignment(scales),
            "zero_points": pointer_alignment(zero_points),
        }

        def invoke() -> torch.Tensor:
            return operator(activation, qweight.t(), bias, scales, zero_points, 128, None)

    else:
        weight_name = None
        with safe_open(str(weights_path), framework="pt", device="cpu") as source:
            for name in ("weight_fp16_nk", "weight_full_fp16_nk"):
                if name in source.keys():
                    weight_name = name
                    break
            if weight_name is None:
                raise SystemExit("dense fixture is missing an FP16 [N,K] weight")
            weight_cpu = source.get_tensor(weight_name)
        weight = weight_cpu.to("xpu").contiguous()
        if activation.dtype != torch.float16 or weight.dtype != torch.float16:
            raise SystemExit("dense fixture must contain FP16 activations and weights")
        if tuple(weight.shape) != (expected.shape[1], activation.shape[1]):
            raise SystemExit("dense weight shape does not match activation and output geometry")

        def invoke() -> torch.Tensor:
            return torch.nn.functional.linear(activation, weight, bias)

        input_alignment = {
            "activation": pointer_alignment(activation),
            "weights": pointer_alignment(weight),
        }

    profiled_queue = ProfiledQueue(torch.xpu.current_stream())
    try:
        with torch.xpu.stream(profiled_queue.stream):
            result = invoke()
            torch.xpu.synchronize()
            if result.dtype != torch.float16 or tuple(result.shape) != tuple(expected.shape):
                raise SystemExit(f"unexpected output: {result.dtype} {tuple(result.shape)}")
            delta = result - expected
            error = delta.abs()
            screen = error <= (0.02 + 0.01 * expected.abs())
            outside = int((~screen).sum().item())

            warmup_count = 768
            warmup_begin = profiled_queue.marker()
            for index in range(warmup_count):
                invoke()
                if (index + 1) % 64 == 0:
                    torch.xpu.synchronize()
            warmup_end = profiled_queue.marker()
            warmup_gpu_ms = profiled_queue.interval_us(warmup_begin, warmup_end) / 1000
            torch.xpu.synchronize()
            host_us: list[float] = []
            gpu_us: list[float] = []
            for _ in range(args.iterations):
                start = profiled_queue.marker()
                host_start = time.perf_counter_ns()
                invoke()
                host_end = time.perf_counter_ns()
                end = profiled_queue.marker()
                gpu_us.append(profiled_queue.interval_us(start, end))
                host_us.append((host_end - host_start) / 1000.0)
    finally:
        profiled_queue.close()

    report = {
        "mode": args.mode,
        "M": int(activation.shape[0]),
        "K": int(activation.shape[1]),
        "N": int(expected.shape[1]),
        "warmup_count": warmup_count,
        "warmup_gpu_ms": warmup_gpu_ms,
        "median_host_enqueue_us": median(host_us),
        "median_gpu_interval_us": median(gpu_us),
        "gpu_interval_timer": "SYCL single-task queue markers; includes host submission gaps",
        "pointer_alignment": {
            **input_alignment,
            "output": pointer_alignment(result),
            "expected": pointer_alignment(expected),
        },
        "host_enqueue_us": host_us,
        "gpu_interval_us": gpu_us,
        "max_abs_error": float(error.max().item()),
        "rms_error": float(delta.float().square().mean().sqrt().item()),
        "values_outside_tolerance": outside,
        "runtime": {
            "torch": torch.__version__,
            "torch_xpu_device": torch.xpu.get_device_name(0),
            "reference_operator": "torch.ops._xpu_C.int4_gemm_w4a16"
            if args.mode == "gptq4"
            else "torch.nn.functional.linear",
        },
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(serialized)
    print(serialized, end="")
    return 0 if outside == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
