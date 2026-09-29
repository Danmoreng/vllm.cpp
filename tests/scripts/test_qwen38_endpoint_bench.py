import json
from pathlib import Path

import pytest

from tools.bench.qwen38_endpoint_bench import (
    HTTPTransport,
    comparison_verdict,
    load_corpus,
    run_workload,
    summarize,
)


def test_load_corpus_accepts_utf8_text_and_messages(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([
        {"id": "text", "text": "Caffè 東京", "max_tokens": 3},
        {"id": "chat", "messages": [{"role": "user", "content": "Olá 👋"}], "max_tokens": 2},
    ], ensure_ascii=False), encoding="utf-8")
    corpus = load_corpus(path)
    assert corpus[0]["text"] == "Caffè 東京"
    assert corpus[1]["messages"][0]["content"] == "Olá 👋"


class FakeTransport:
    tokenizer_identity = {"name": "fixture-tokenizer", "revision": "abc123"}

    def __init__(self):
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        n = payload["max_tokens"]
        return {
            "events": [
                {"offset_s": 0.01 * (i + 1), "token_ids": [100 + i], "text": chr(97 + i)}
                for i in range(n)
            ],
            "prompt_tokens": 7,
            "generated_tokens": n,
            "cache": {"hits": 2, "queries": 3},
            "draft": {"accepted": 1, "proposed": 2},
            "memory_samples": [{"offset_s": 0, "bytes": 1234}],
        }


def test_workload_is_canonical_closed_loop_and_preserves_raw_repetitions():
    transport = FakeTransport()
    config = {
        "endpoint": "http://example.invalid/v1/completions",
        "model": "runtime-selected-model",
        "corpus": [{"id": "p", "text": "héllo", "max_tokens": 3}],
        "draft": "on",
        "concurrency": 2,
        "waves": 5,
    }
    result = run_workload(config, transport)
    assert len(result["samples"]) == 10
    assert [s["wave"] for s in result["samples"]].count(0) == 2
    assert result["raw_repetitions"] == result["samples"]
    assert result["tokenizer_identity"] == transport.tokenizer_identity
    assert len(result["canonical_payload_hash"]) == 64
    assert all(s["token_fingerprint"] for s in result["samples"])
    assert all(s["ttft_s"] == pytest.approx(0.01) for s in result["samples"])
    assert all(s["tpot_s"] == pytest.approx(0.01) for s in result["samples"])
    assert all(s["itl_s"] == pytest.approx([0.01, 0.01]) for s in result["samples"])
    assert all(s["cache"] == {"hits": 2, "queries": 3} for s in result["samples"])
    payload = transport.payloads[0]
    assert payload["model"] == "runtime-selected-model"
    assert payload["seed"] == 0
    assert payload["temperature"] == 0
    assert payload["ignore_eos"] is True
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert "draft" not in payload


def test_serving_refuses_fewer_than_five_waves():
    with pytest.raises(ValueError, match="five"):
        run_workload({
            "endpoint": "http://localhost/v1/completions", "model": "m",
            "corpus": [{"text": "x", "max_tokens": 1}], "waves": 4,
        }, FakeTransport())


def test_summary_uses_interpolated_percentiles_and_measured_makespan():
    samples = [
        {"ttft_s": x, "tpot_s": x / 10, "e2e_s": x + 1,
         "prompt_tokens": 2, "generated_tokens": 2, "itl_s": [x / 10]}
        for x in (1.0, 2.0, 3.0, 4.0)
    ]
    out = summarize(samples, workload_makespan_s=20)
    assert out["request_count"] == 4
    assert out["metrics"]["ttft_s"]["p50"] == pytest.approx(2.5)
    assert out["metrics"]["ttft_s"]["p99"] == pytest.approx(3.97)
    assert out["workload_makespan_s"] == 20
    assert out["request_rate"] == pytest.approx(4 / 20)
    assert out["token_throughput"] == pytest.approx(8 / 20)
    assert out["raw_repetitions"] == samples


def test_workload_promotes_sample_refusal_and_records_full_run_makespan(monkeypatch):
    clock = iter([10.0, 10.0, 10.2, 10.2, 10.5, 10.5, 11.0,
                  11.0, 11.4, 11.4, 12.0, 12.0])
    monkeypatch.setattr("tools.bench.qwen38_endpoint_bench.time.perf_counter", lambda: next(clock))

    class RefusingTransport:
        tokenizer_identity = {"name": "t", "revision": "1"}
        def __call__(self, payload):
            return {"events": [], "refusal_reason": "usage missing", "e2e_s": 0.1}

    result = run_workload({"endpoint": "x", "model": "m", "corpus": [{"text": "x"}],
                           "waves": 5, "concurrency": 1}, RefusingTransport())
    assert result["workload_makespan_s"] == pytest.approx(2.0)
    assert result["refusal_reason"] == "one or more samples were refused"
    assert result["request_rate"] is None
    assert result["token_throughput"] is None


def test_comparison_fails_closed_on_all_canonical_evidence():
    sample = {"refusal_reason": None, "payload_hash": "payload", "prompt_tokens": 7,
              "token_fingerprint": "tokens"}
    base = {"refusal_reason": None, "canonical_payload_hash": "run-payload",
            "tokenizer_identity": {"name": "tok", "revision": "1"},
            "samples": [sample], "token_fingerprints": ["tokens"]}
    assert comparison_verdict(base, json.loads(json.dumps(base)))["verdict"] == "matched"
    mutations = [
        ("top refusal", lambda x: x.update(refusal_reason="no")),
        ("sample refusal", lambda x: x["samples"][0].update(refusal_reason="no")),
        ("payload", lambda x: x.update(canonical_payload_hash="other")),
        ("tokenizer", lambda x: x.update(tokenizer_identity=None)),
        ("prompt tokens", lambda x: x["samples"][0].update(prompt_tokens=8)),
        ("consumed token", lambda x: x["samples"][0].update(token_fingerprint="other")),
        ("fingerprint list", lambda x: x.update(token_fingerprints=["other"])),
    ]
    for _, mutate in mutations:
        right = json.loads(json.dumps(base))
        mutate(right)
        assert comparison_verdict(base, right)["verdict"] == "refused"


def test_canonical_hashes_exclude_runtime_model_names_and_adapter_extras():
    config = {"endpoint": "x", "corpus": [{"text": "same", "max_tokens": 2}],
              "waves": 5}
    left = run_workload({**config, "model": "engine-a/model"}, FakeTransport())
    right = run_workload({**config, "model": "engine-b-alias"}, FakeTransport())
    assert left["canonical_payload_hash"] == right["canonical_payload_hash"]
    assert [s["payload_hash"] for s in left["samples"]] == [
        s["payload_hash"] for s in right["samples"]
    ]
    assert comparison_verdict(left, right)["verdict"] == "matched"

    http_a = HTTPTransport("x", tokenizer_identity="tok@rev", extra_json={"draft_model": "a"})
    http_b = HTTPTransport("x", tokenizer_identity="tok@rev", extra_json={"draft_model": "b"})
    payload = {"model": "runtime", "prompt": "same"}
    assert http_a.prepare_payload(payload) != http_b.prepare_payload(payload)


def test_workload_refuses_missing_or_comparison_different_tokenizer_identity():
    transport = FakeTransport()
    transport.tokenizer_identity = None
    config = {"endpoint": "x", "model": "m", "corpus": [{"text": "x", "max_tokens": 1}],
              "waves": 5}
    missing = run_workload(config, transport)
    assert "tokenizer identity" in missing["refusal_reason"]

    left = run_workload(config, FakeTransport())
    different_transport = FakeTransport()
    different_transport.tokenizer_identity = {"name": "other", "revision": "abc123"}
    right = run_workload(config, different_transport)
    verdict = comparison_verdict(left, right)
    assert verdict["verdict"] == "refused"
    assert "tokenizer" in verdict["refusal_reason"]


def test_http_payload_extra_json_is_explicit_and_cannot_override_canonical_fields():
    transport = HTTPTransport("http://example.invalid", tokenizer_identity="tok@revision",
                              extra_json={"draft_model": "d"})
    assert transport.prepare_payload({"model": "m", "stream": True}) == {
        "model": "m", "stream": True, "draft_model": "d"
    }
    with pytest.raises(ValueError, match="override"):
        HTTPTransport("x", tokenizer_identity="tok", extra_json={"model": "other"}).prepare_payload({"model": "m"})


def test_sample_refuses_usage_without_reliable_generated_token_accounting():
    class NoUsageTransport:
        tokenizer_identity = {"name": "t"}
        def __call__(self, payload):
            return {"events": [{"offset_s": .1, "text": "one"},
                               {"offset_s": .2, "text": " chunk"}]}
    result = run_workload({"endpoint": "x", "model": "m", "corpus": [{"text": "x"}],
                           "waves": 5}, NoUsageTransport())
    assert all(s["generated_tokens"] is None for s in result["samples"])
    assert all(s["tpot_s"] is None for s in result["samples"])
    assert result["token_throughput"] is None
