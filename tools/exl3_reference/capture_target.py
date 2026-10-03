#!/usr/bin/env python3
"""Bounded cold eager P128/D1 or P128/D64 original-target captures.

Reference reproducibility evidence only. No changed arithmetic or native pass.
"""
import argparse
import dataclasses
import functools
import heapq
import json
import math
from pathlib import Path
import os
import struct

from capture_block import BlockCapture
from capture_projection import IMAGE
from capture_runtime_layout import PROFILE, verify_inputs
from compare_projection import blob, metrics
from extract_projection import digest, headers
from runtime_layout import tensor_layout


def observe_logits(original, save):
    """Delegate exactly once, preserving arguments and the result object."""
    @functools.wraps(original)
    def observed(hidden_states, *args, **kwargs):
        result = original(hidden_states, *args, **kwargs)
        save(hidden_states, result)
        return result
    return observed


def target_phases(decode_steps):
    headers.require(decode_steps in (1, 64), "only focused D1/D64 captures are supported")
    return ["p128"] + [f"d{i}" for i in range(1, decode_steps + 1)]


def load_trace(path, decode_steps):
    record = json.loads(path.read_text())
    ids = record["output_ids"]
    headers.require(len(ids) == 1 and len(ids[0]) == decode_steps + 1 and
                    all(type(t) is int and 0 <= t < 248320 for t in ids[0]),
                    "trace must contain one complete bounded target token sequence")
    return ids[0]


def validate_prefix_witnesses(witnesses, prompt, outputs):
    phases = ["p128"] + [f"d{i}" for i in range(1, len(outputs))]
    headers.require(list(witnesses) == phases, "missing or reordered actual input witnesses")
    for step, phase in enumerate(phases):
        ids = prompt if step == 0 else outputs[step - 1:step]
        positions = list(range(len(prompt))) if step == 0 else [len(prompt) + step - 1]
        headers.require(witnesses[phase]["token_ids"] == ids and
                        witnesses[phase]["positions"] == [positions] * 3,
                        "actual input tokens/positions differ from recorded prefix: " + phase)


def probability_metrics(actual, expected):
    headers.require(len(actual) == len(expected) and len(actual) > 0,
                    "logit lengths must match and be nonempty")
    headers.require(all(math.isfinite(x) for row in (actual, expected) for x in row),
                    "nonfinite logits")
    def log_probs(row):
        maximum = max(row)
        normalizer = math.log(math.fsum(math.exp(x - maximum) for x in row))
        return [x - maximum - normalizer for x in row]
    a, e = log_probs(actual), log_probs(expected)
    pa, pe = [math.exp(x) for x in a], [math.exp(x) for x in e]
    top = min(10, len(actual))
    ai = set(heapq.nlargest(top, range(len(actual)), key=lambda i: actual[i]))
    ei = set(heapq.nlargest(top, range(len(expected)), key=lambda i: expected[i]))
    tv = 0.5 * math.fsum(abs(x - y) for x, y in zip(pa, pe))
    kl = math.fsum(p * (x - y) for p, x, y in zip(pe, e, a))
    overlap = len(ai & ei)
    return {"TV": tv, "KL_reference_to_actual": kl, "top10_overlap": overlap,
            "investigation_trigger": tv > 0.02 or kl > 0.002 or overlap < min(9, top)}


class TargetCapture(BlockCapture):
    def install_target_capture(self, output, decode_steps=1):
        import torch
        installed = self.install_block_capture(output)
        self._target_phases = []
        self._target_expected_phases = target_phases(decode_steps)
        self._target_layouts = []
        self._target_witnesses, self._target_hooks = {}, []
        model = self.model_runner.model
        modules = dict(model.named_modules())
        layer = modules[installed["layer"]]
        embedding = modules[installed["layer"].rsplit("layers.", 1)[0] + "embed_tokens"]
        def witness(module, args, kwargs, tokens=False):
            step = len(self._target_phases)
            headers.require(step < len(self._target_expected_phases), "extra model input witness")
            phase = self._target_expected_phases[step]
            values = dict(zip(("input_ids",) if tokens else ("positions",), args)) | kwargs
            value = values["input_ids" if tokens else "positions"]
            headers.require(value.dtype in (torch.int32, torch.int64) and
                            value.numel() <= (128 if tokens else 384), "unbounded actual input witness")
            row = self._target_witnesses.setdefault(phase, {})
            key = "token_ids" if tokens else "positions"
            headers.require(key not in row, "duplicate actual input witness")
            row[key] = value.detach().cpu().tolist()
        self._target_hooks.append(embedding.register_forward_pre_hook(
            lambda module, args, kwargs: witness(module, args, kwargs, True), with_kwargs=True))
        self._target_hooks.append(layer.register_forward_pre_hook(witness, with_kwargs=True))
        self._target_original_logits = model.compute_logits
        def save(hidden, logits):
            step = len(self._target_phases)
            headers.require(step < len(self._target_expected_phases), "unexpected extra target-head step")
            phase = self._target_expected_phases[step]
            if step < 2:
                headers.require(self._block_records[-1]["phase"] == phase, "unexpected block/head order")
            headers.require(hidden.ndim == logits.ndim == 2 and
                            hidden.shape == (1, 5120) and logits.shape == (1, 248320),
                            "requires gathered final prompt row and full target vocabulary")
            self._target_layouts.append({"phase": phase, "hidden": tensor_layout(hidden),
                                         "logits": tensor_layout(logits)})
            for label, value in (("logits_hidden", hidden), ("logits", logits)):
                host = value.detach().cpu().contiguous()
                dtype = {torch.float16: "F16", torch.float32: "F32"}.get(host.dtype)
                headers.require(dtype is not None and torch.isfinite(host).all().item(),
                                "invalid target-head boundary")
                self._block_tensors[phase + "_" + label] = (
                    dtype, list(host.shape), host.numpy().tobytes())
            self._target_phases.append(phase)
            if len(self._target_phases) == 2:
                # The existing bounded block observer covers only P128/D1.
                # Later decode steps retain head observations, not block states.
                for hook in self._block_hooks:
                    hook.remove()
                self._block_hooks.clear()
        model.compute_logits = observe_logits(self._target_original_logits, save)
        return installed | {"target_observer": "compute_logits delegates original once; bounded active copies"}

    def finish_target_capture(self):
        self.model_runner.model.compute_logits = self._target_original_logits
        for hook in self._target_hooks:
            hook.remove()
        headers.require(self._target_phases == self._target_expected_phases, "missing full-target head calls")
        return self.finish_block_capture() | {"target_layouts": self._target_layouts,
                                              "actual_input_witnesses": self._target_witnesses}


def compare_repeats(paths, decode_steps=1):
    reports = [headers.read_shard_header(p) for p in paths]
    comparisons = []
    for j in range(1, len(paths)):
        for i in range(j):
            stages = {}
            for phase in target_phases(decode_steps):
                labels = ("ba_output", "ssm_state_after", "hidden_out") if phase in ("p128", "d1") else ()
                for label in labels + ("logits_hidden", "logits"):
                    key = phase + "_" + label
                    a, raw_a = blob(paths[j], reports[j], key)
                    e, raw_e = blob(paths[i], reports[i], key)
                    headers.require(a["shape"] == e["shape"] and a["dtype"] == e["dtype"],
                                    "repeat stage contract differs")
                    result = metrics(raw_a, raw_e, a["dtype"])
                    headers.require(result["finite"], "nonfinite repeat stage")
                    if label in ("logits", "ssm_state_after"):
                        fmt = "<f" if a["dtype"] == "F32" else "<e"
                        av = [v for (v,) in struct.iter_unpack(fmt, raw_a)]
                        ev = [v for (v,) in struct.iter_unpack(fmt, raw_e)]
                        if label == "logits":
                            result["distribution"] = probability_metrics(av, ev)
                        else:
                            result["fixed_pointwise_failures"] = sum(
                                abs(x-y) > 1e-5 + 1e-4 * abs(y) for x,y in zip(av, ev))
                    stages[key] = result
            comparisons.append({"reference_repeat": i, "actual_repeat": j, "stages": stages})
    return comparisons


def capture(args):
    target_phases(args.decode_steps)
    headers.require(1 <= args.repeats <= 3, "repeat count must be bounded by three")
    headers.require(not args.output_dir.exists(), "refusing to overwrite target repeats")
    reference = verify_inputs(args.reference_manifest, args.model_dir, args.image_identity)
    import yaml
    profile = yaml.safe_load(PROFILE.read_text())
    for key, value in profile["env"].items():
        if value is not None:
            os.environ[key] = str(value).replace("{model_dir}", str(PROFILE.parent))
    args.output_dir.mkdir(parents=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["EXL3_LOADER_REPORT_DIR"] = str(args.output_dir / "loader")
    from vllm import LLM, SamplingParams
    from vllm.config import ReasoningConfig
    from vllm.engine.arg_utils import EngineArgs
    import torch
    headers.require(torch.__version__ == "2.13.0+xpu" and "B70" in torch.xpu.get_device_name(0),
                    "requires pinned Torch XPU and B70")
    allowed = {f.name for f in dataclasses.fields(EngineArgs)}
    kwargs = {k: v for k, v in profile["vllm"].items() if k in allowed}
    if isinstance(kwargs.get("reasoning_config"), dict):
        kwargs["reasoning_config"] = ReasoningConfig(**kwargs["reasoning_config"])
    overrides = {"enforce_eager": True, "speculative_config": None,
                 "compilation_config": {"mode": 0, "cudagraph_mode": "NONE", "custom_ops": ["none"]}}
    trace = load_trace(args.trace_capture_json, args.decode_steps) if args.trace_capture_json else None
    if trace is not None:
        overrides["enable_trace_replay"] = True
    kwargs.update(overrides, model=str(args.model_dir), worker_extension_cls="capture_target.TargetCapture")
    tokens = [1000 + (i * 37) % 4096 for i in range(128)]
    params = SamplingParams(temperature=0, max_tokens=args.decode_steps + 1, ignore_eos=True,
                            trace_decode_token_ids=trace)
    llm = None
    paths, records = [], []
    try:
        llm = LLM(**kwargs)
        ordinary = llm.generate({"prompt_token_ids": tokens}, params, use_tqdm=False)
        ordinary_ids = [list(o.outputs[0].token_ids) for o in ordinary]
        for repeat in range(args.repeats):
            headers.require(llm.reset_prefix_cache(), "prefix reset failed")
            path = args.output_dir / f"repeat-{repeat}.safetensors"
            llm.collective_rpc("install_target_capture", timeout=60, args=(str(path), args.decode_steps))
            observed = llm.generate({"prompt_token_ids": tokens}, params, use_tqdm=False)
            ids = [list(o.outputs[0].token_ids) for o in observed]
            result = llm.collective_rpc("finish_target_capture", timeout=60)
            headers.require(len(result) == 1, "requires one worker")
            result = result[0] | {"output_ids": ids, "ordinary_ids_exact": ids == ordinary_ids}
            with path.with_suffix(".json").open("x") as stream:
                json.dump(result, stream, indent=2); stream.write("\n")
            validate_prefix_witnesses(result["actual_input_witnesses"], tokens, ids[0])
            if args.decode_steps == 1:
                headers.require(ids == ordinary_ids, "different continuation prevents matched-prefix D1 comparison")
            if trace is not None:
                headers.require(ids == [trace], "original trace replay did not follow the requested prefix")
            if records:
                headers.require(ids == records[0]["output_ids"],
                                "different repeat continuations prevent matched-prefix comparisons")
            paths.append(path); records.append(result)
            print("TARGET_REPEAT_DONE", repeat, ids, flush=True)
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
    report = {"schema": 1, "kind": f"original_eager_target_P128_D{args.decode_steps}_reproducibility",
              "decode_steps": args.decode_steps,
              "sampling_mode": "original_trace_token_replay" if trace is not None else "original_greedy",
              "trace_source_sha256": digest(args.trace_capture_json.read_bytes()) if trace is not None else None,
              "trace_token_ids": trace,
              "image": IMAGE, "checkpoint": reference["checkpoint"]["identity"],
              "profile_sha256": digest(PROFILE.read_bytes()), "s1_overrides": overrides,
              "capture_tool_sha256": digest(Path(__file__).read_bytes()),
              "prompt_token_ids": tokens, "ordinary_output_ids": ordinary_ids,
              "observed_output_ids": [r["output_ids"] for r in records],
              "repeats": [{"path": str(p), "sha256": digest(p.read_bytes())} for p in paths],
              "comparisons": compare_repeats(paths, args.decode_steps),
              "scope": "Bounded original full-target capture; block states only P128/D1. Not a native pass or a new numerical envelope."}
    with (args.output_dir / "comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print("TARGET_REPEAT_CAPTURE_DONE", args.output_dir, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--decode-steps", type=int, choices=(1, 64), default=1)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--trace-capture-json", type=Path)
    capture(parser.parse_args())
