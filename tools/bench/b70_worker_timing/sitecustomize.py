"""Diagnostic XPU V2 runner timings, loaded through PYTHONPATH.

Set B70_WORKER_TIMING_PATH to a mounted JSONL file. This adds a synchronization
after compute_logits and writes per step. It is deliberately unsuitable for
scored throughput runs; use it to establish the active timing boundaries.
"""

import functools
import json
import os
import time
import atexit


_path = os.environ.get("B70_WORKER_TIMING_PATH")
if _path:
    import torch

    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.xpu_model_runner import XPUModelRunnerV2
    from vllm.v1.worker.xpu_worker import XPUWorker

    _schedule = Scheduler.schedule
    _execute = XPUModelRunnerV2.execute_model
    _sample = XPUModelRunnerV2.sample_tokens
    _shutdown = XPUWorker.shutdown
    _pending = {}
    _buffered = os.environ.get("B70_WORKER_TIMING_BUFFER") == "1"
    _records = []
    _schedule_records = []
    _schedule_path = os.environ.get("B70_WORKER_SCHED_PATH")

    def _write(path, records):
        if not records:
            return
        payload = "".join(json.dumps(record, separators=(",", ":")) + "\n"
                          for record in records).encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view):]
        finally:
            os.close(fd)

    def _flush():
        if _records:
            _write(_path, _records)
            _records.clear()
        if _schedule_path and _schedule_records:
            _write(_schedule_path, _schedule_records)
            _schedule_records.clear()

    atexit.register(_flush)

    @functools.wraps(_schedule)
    def _timed_schedule(self, *args, **kwargs):
        output = _schedule(self, *args, **kwargs)
        calls = []
        for request_id, query_tokens in output.num_scheduled_tokens.items():
            request = self.requests[request_id]
            context_after = request.num_computed_tokens
            context_before = context_after - query_tokens
            prompt_query_tokens = min(
                query_tokens, max(0, request.num_prompt_tokens - context_before)
            )
            calls.append({
                "request_id": request_id,
                "query_tokens": query_tokens,
                "context_before": context_before,
                "context_after": context_after,
                "prompt_query_tokens": prompt_query_tokens,
                "decode_query_tokens": query_tokens - prompt_query_tokens,
            })
        _schedule_records.append({
            "event": "b70_scheduler_call",
            "pid": os.getpid(),
            "calls": calls,
        })
        return output

    @functools.wraps(_execute)
    def _timed_execute(self, scheduler_output, *args, **kwargs):
        query_tokens = scheduler_output.total_num_scheduled_tokens
        if query_tokens == 0:
            return _execute(self, scheduler_output, *args, **kwargs)
        record = {
            "event": "b70_v2_worker_timing",
            "pid": os.getpid(),
            "query_tokens": query_tokens,
            "query_tokens_by_request": scheduler_output.num_scheduled_tokens,
            "execute_start_ns": time.perf_counter_ns(),
            "model_spans": [],
        }
        _pending[id(self)] = record
        model = self.model
        original_forward = model.forward

        @functools.wraps(original_forward)
        def timed_forward(*forward_args, **forward_kwargs):
            start = time.perf_counter_ns()
            output = original_forward(*forward_args, **forward_kwargs)
            record["model_spans"].append({
                "start_ns": start,
                "submit_end_ns": time.perf_counter_ns(),
            })
            return output

        model.forward = timed_forward
        try:
            return _execute(self, scheduler_output, *args, **kwargs)
        finally:
            record["execute_end_ns"] = time.perf_counter_ns()
            model.forward = original_forward

    @functools.wraps(_sample)
    def _timed_sample(self, *args, **kwargs):
        record = _pending.pop(id(self), None)
        if record is None:
            return _sample(self, *args, **kwargs)
        record["sample_start_ns"] = time.perf_counter_ns()
        model = self.model
        original_logits = model.compute_logits

        @functools.wraps(original_logits)
        def timed_logits(*logit_args, **logit_kwargs):
            record["logits_start_ns"] = time.perf_counter_ns()
            output = original_logits(*logit_args, **logit_kwargs)
            torch.xpu.synchronize()
            record["logits_complete_ns"] = time.perf_counter_ns()
            return output

        model.compute_logits = timed_logits
        try:
            return _sample(self, *args, **kwargs)
        finally:
            record["sample_end_ns"] = time.perf_counter_ns()
            model.compute_logits = original_logits
            if _buffered:
                _records.append(record)
            else:
                _write(_path, [record])

    @functools.wraps(_shutdown)
    def _timed_shutdown(self, *args, **kwargs):
        _flush()
        return _shutdown(self, *args, **kwargs)

    XPUModelRunnerV2.execute_model = _timed_execute
    XPUModelRunnerV2.sample_tokens = _timed_sample
    XPUWorker.shutdown = _timed_shutdown
    if _schedule_path:
        Scheduler.schedule = _timed_schedule
