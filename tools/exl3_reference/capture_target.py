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
from capture_runtime_layout import PROFILE, verify_inputs, active_metadata
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


def deterministic_ba_forward(original, read_setting, write_setting):
    """Scope the original dense matmul policy; delegate once and restore on failure."""
    @functools.wraps(original)
    def forward(*args, **kwargs):
        previous = read_setting()
        write_setting((True, False))
        try:
            return original(*args, **kwargs)
        finally:
            write_setting(previous)
    return forward


def target_phases(decode_steps):
    headers.require(decode_steps in (1, 29, 64), "only focused D1/D29/D64 captures are supported")
    return ["p128"] + [f"d{i}" for i in range(1, decode_steps + 1)]


def selected_detail_kind(block_step, detail_layer):
    headers.require(type(detail_layer) is int and detail_layer in (-1, 1, 3, 21) and
                    (detail_layer == -1 or block_step == 29),
                    "detailed observation supports D29 GDN1/GDN21 or attention3 only")
    return {-1: "none", 1: "gdn", 3: "attention", 21: "gdn"}[detail_layer]


def selected_block_layers(modules, first_layer):
    prefix = first_layer.rsplit(".", 1)[0] + "."
    layers = [(int(name[len(prefix):]), module) for name, module in modules.items()
              if name.startswith(prefix) and name[len(prefix):].isdigit()]
    headers.require(sorted(index for index, _ in layers) == list(range(64)),
                    "selected-step observation requires all 64 target blocks")
    return sorted(layers)


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
    def install_deterministic_ba(self):
        import torch
        headers.require(not hasattr(self, "_deterministic_ba_originals"), "BA policy already installed")
        modules = [(name, module) for name, module in self.model_runner.model.named_modules()
                   if name.endswith("linear_attn.in_proj_ba")]
        headers.require(len(modules) == 48 and all(module.weight.dtype == torch.float16 and
                        module.weight.shape == (96, 5120) and
                        type(module.quant_method).__name__ == "UnquantizedLinearMethod"
                        for _, module in modules), "requires all48 actual unquantized target BA modules")
        def read():
            return (torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled())
        def write(setting):
            torch.use_deterministic_algorithms(setting[0], warn_only=setting[1])
        self._deterministic_ba_originals = [(module, module.forward) for _, module in modules]
        record = []
        for name, module in modules:
            record.append({"name": name, "weight_layout": tensor_layout(module.weight),
                           "quant_method": type(module.quant_method).__name__})
            module.forward = deterministic_ba_forward(module.forward, read, write)
        return {"kind": "diagnostic_original_deterministic_BA_only", "modules": record,
                "scope": "Original forwards/operands unchanged; deterministic policy only inside each BA forward, restored before return. Not a replacement of frozen qualification."}

    def restore_deterministic_ba(self):
        for module, original in self._deterministic_ba_originals:
            module.forward = original
        del self._deterministic_ba_originals
        return {"restored_modules": 48}

    def install_target_capture(self, output, decode_steps=1, block_step=-1, detail_layer=-1):
        import torch
        headers.require(block_step == -1 or (decode_steps == 29 and block_step == 29),
                        "selected block observation is bounded to D29")
        detail_kind = selected_detail_kind(block_step, detail_layer)
        installed = self.install_block_capture(output)
        self._target_phases = []
        self._target_expected_phases = target_phases(decode_steps)
        self._target_layouts = []
        self._target_witnesses, self._target_hooks = {}, []
        model = self.model_runner.model
        modules = dict(model.named_modules())
        layer = modules[installed["layer"]]
        self._selected_block_hooks = []
        self._selected_block_counts = {}
        if block_step >= 0:
            def save_boundary(key, value):
                headers.require(key not in self._block_tensors and
                                value.dtype == torch.float16 and value.shape == (1, 5120),
                                "duplicate or unbounded D29 block boundary: " + key)
                host = value.detach().cpu().contiguous()
                headers.require(torch.isfinite(host).all().item(), "nonfinite D29 block boundary")
                self._block_tensors[key] = ("F16", [1, 5120], host.numpy().tobytes())
            for index, block in selected_block_layers(modules, installed["layer"]):
                def before(module, args, kwargs, index=index):
                    if len(self._target_phases) != block_step:
                        return
                    values = dict(zip(("positions", "hidden_states", "residual"), args)) | kwargs
                    save_boundary(f"d29_l{index}_hidden_in", values["hidden_states"])
                    residual = values.get("residual")
                    if residual is not None:
                        save_boundary(f"d29_l{index}_residual_in", residual)
                def after(module, args, kwargs, output, index=index):
                    if len(self._target_phases) != block_step:
                        return
                    headers.require(isinstance(output, tuple) and len(output) == 2,
                                    "unexpected target block result")
                    save_boundary(f"d29_l{index}_hidden_out", output[0])
                    save_boundary(f"d29_l{index}_residual_out", output[1])
                    self._selected_block_counts[index] = self._selected_block_counts.get(index, 0) + 1
                self._selected_block_hooks.append(block.register_forward_pre_hook(before, with_kwargs=True))
                self._selected_block_hooks.append(block.register_forward_hook(after, with_kwargs=True))
                for label, norm in (("post_input_norm", block.input_layernorm),
                                    ("post_attn_norm", block.post_attention_layernorm)):
                    def norm_after(module, args, kwargs, output, index=index, label=label):
                        if len(self._target_phases) == block_step:
                            value = output[0] if isinstance(output, tuple) else output
                            save_boundary(f"d29_l{index}_{label}", value)
                    self._selected_block_hooks.append(norm.register_forward_hook(norm_after, with_kwargs=True))
        self._selected_block_step = block_step
        self._selected_detail_record = None
        self._selected_attention_capture = None
        if detail_kind == "attention":
            from capture_attention import AttentionCapture
            observer = AttentionCapture()
            observer.model_runner = self.model_runner
            path = Path(output)
            observer.install_block_capture(path.with_name(path.stem + "-attention3.safetensors"),
                                           selected_step=29, step_counter=lambda: len(self._target_phases))
            self._selected_attention_capture = observer
        if detail_kind == "gdn":
            from vllm.forward_context import get_forward_context
            block = modules[installed["layer"].rsplit(".", 1)[0] + f".{detail_layer}"]
            headers.require(block.layer_type == "linear_attention", "selected layer must be GDN")
            mixer = block.linear_attn
            def save_detail(label, value):
                key = f"d29_l{detail_layer}_detail_" + label
                headers.require(key not in self._block_tensors and value.numel() <= 1_000_000,
                                "duplicate or unbounded D29 selected GDN detail")
                host = value.detach().cpu().contiguous()
                dtype = {torch.float16: "F16", torch.float32: "F32"}.get(host.dtype)
                headers.require(dtype is not None and torch.isfinite(host).all().item(),
                                "invalid D29 selected GDN detail")
                self._block_tensors[key] = (dtype, list(host.shape), host.numpy().tobytes())
            def mixer_before(module, args, kwargs):
                if len(self._target_phases) != block_step:
                    return
                meta = get_forward_context().attn_metadata[mixer.prefix]
                indices = meta.non_spec_state_indices_tensor
                headers.require(meta.num_actual_tokens == 1 and not meta.num_prefills and
                                indices.numel() == 1, "requires one actual D29 decode state")
                slot = int(indices.detach().cpu().item())
                headers.require(0 <= slot < mixer.kv_cache[0].shape[0] and
                                self._selected_detail_record is None, "invalid or repeated active D29 slot")
                self._selected_detail_record = {"layer": detail_layer, "step": 29, "slot": slot,
                    "metadata": active_metadata(meta),
                    "conv_layout": tensor_layout(mixer.kv_cache[0]),
                    "ssm_layout": tensor_layout(mixer.kv_cache[1])}
                save_detail("conv_before", mixer.kv_cache[0][slot])
                save_detail("ssm_before", mixer.kv_cache[1][slot])
            def mixer_after(module, args, kwargs, output):
                if len(self._target_phases) != block_step:
                    return
                slot = self._selected_detail_record["slot"]
                save_detail("mixer_output", output[0] if isinstance(output, tuple) else output)
                save_detail("conv_after", mixer.kv_cache[0][slot])
                save_detail("ssm_after", mixer.kv_cache[1][slot])
            self._selected_block_hooks.append(mixer.register_forward_pre_hook(mixer_before, with_kwargs=True))
            self._selected_block_hooks.append(mixer.register_forward_hook(mixer_after, with_kwargs=True))
            for label, module in (("qkvz", mixer.in_proj_qkvz), ("ba", mixer.in_proj_ba),
                                  ("gated_norm", mixer.norm), ("out_proj", mixer.out_proj)):
                def detail_after(module, args, kwargs, output, label=label):
                    if len(self._target_phases) == block_step:
                        save_detail(label + "_output", output[0] if isinstance(output, tuple) else output)
                self._selected_block_hooks.append(module.register_forward_hook(detail_after, with_kwargs=True))
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
        attention = self._selected_attention_capture
        attention_record = attention.finish_block_capture() if attention is not None else None
        self.model_runner.model.compute_logits = self._target_original_logits
        for hook in self._target_hooks:
            hook.remove()
        for hook in self._selected_block_hooks:
            hook.remove()
        if self._selected_block_step >= 0:
            headers.require(self._selected_block_counts == dict.fromkeys(range(64), 1),
                            "missing, repeated or reordered D29 block observations")
        headers.require(self._target_phases == self._target_expected_phases, "missing full-target head calls")
        return self.finish_block_capture() | {"selected_block_step": self._selected_block_step,
                                              "selected_attention_record": attention_record,
                                              "selected_detail_record": self._selected_detail_record,
                                              "selected_block_counts": self._selected_block_counts,
                                              "target_layouts": self._target_layouts,
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
    ba_policy = None
    paths, records = [], []
    try:
        llm = LLM(**kwargs)
        if args.deterministic_ba:
            ba_policy = llm.collective_rpc("install_deterministic_ba", timeout=60)
        ordinary = llm.generate({"prompt_token_ids": tokens}, params, use_tqdm=False)
        ordinary_ids = [list(o.outputs[0].token_ids) for o in ordinary]
        for repeat in range(args.repeats):
            headers.require(llm.reset_prefix_cache(), "prefix reset failed")
            path = args.output_dir / f"repeat-{repeat}.safetensors"
            llm.collective_rpc("install_target_capture", timeout=60,
                               args=(str(path), args.decode_steps, args.block_step, args.detail_layer))
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
            try:
                if ba_policy is not None:
                    restored = llm.collective_rpc("restore_deterministic_ba", timeout=60)
                    headers.require(restored == [{"restored_modules": 48}], "BA policy restoration failed")
            finally:
                llm.llm_engine.engine_core.shutdown()
    kind_prefix = "diagnostic_deterministic_BA_" if args.deterministic_ba else ""
    report = {"schema": 1, "kind": f"{kind_prefix}original_eager_target_P128_D{args.decode_steps}_reproducibility",
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
              "selected_block_step": args.block_step,
              "selected_detail_layer": args.detail_layer,
              "diagnostic_deterministic_ba": bool(args.deterministic_ba), "ba_policy": ba_policy,
              "scope": "Bounded original full-target capture; layer0 states P128/D1, optional read-only D29 block/norm boundaries. Not a native pass or a new numerical envelope."}
    with (args.output_dir / "comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print("TARGET_REPEAT_CAPTURE_DONE", args.output_dir, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--decode-steps", type=int, choices=(1, 29, 64), default=1)
    parser.add_argument("--block-step", type=int, choices=(-1, 29), default=-1)
    parser.add_argument("--detail-layer", type=int, choices=(-1, 1, 3, 21), default=-1)
    parser.add_argument("--deterministic-ba", action="store_true",
                        help="separate diagnostic policy: original BA matmul deterministic; frozen gates unchanged")
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--trace-capture-json", type=Path)
    capture(parser.parse_args())
