import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/bench/run_qwen38_tensorfold_gap.sh"
TF_PIN = "191188075bca56a7c71074a79375eb4c1cb22e1c"
RECIPE_PIN = "856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e"


def sh(command, cwd=None):
    return subprocess.run(command, cwd=cwd or ROOT, text=True, capture_output=True)


def make_repo(path: Path, revision: str | None = None):
    path.mkdir(parents=True)
    sh(["git", "init", "-q"], path)
    sh(["git", "config", "user.email", "fixture@example.invalid"], path)
    sh(["git", "config", "user.name", "Fixture"], path)
    (path / "tracked").write_text("clean\n")
    sh(["git", "add", "tracked"], path)
    sh(["git", "commit", "-qm", "fixture"], path)
    actual = sh(["git", "rev-parse", "HEAD"], path).stdout.strip()
    return revision or actual


def fixture(tmp_path: Path):
    source = tmp_path / "source"
    revision = make_repo(source)
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"measured fixture")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "artifacts.sha256"
    manifest.write_text(f"{digest}  {artifact.name}\n")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:       999999999 kB\n")
    clock = tmp_path / "clock.py"
    clock.write_text("# fixture clock sampler\n")
    env = tmp_path / "arm.env"
    env.write_text(
        f"ENGINE=vllm-cpp\nSOURCE_DIR={source}\nEXPECTED_REVISION={revision}\n"
        f"ARTIFACT_MANIFEST={manifest}\nENDPOINT=http://127.0.0.1:9/v1/completions\n"
        "ENDPOINT_READY_URL=http://127.0.0.1:9/health\nMODEL=fixture\n"
        "TOKENIZER_IDENTITY=fixture@revision\nLAUNCH_COMMAND='printf launched'\n"
        "MIN_FREE_MEMORY_KIB=1\n"
    )
    process_env = {
        **os.environ,
        "RC_DEVICE": "fixture:gpu0",
        "RC_JOB_ID": "lease-1",
        "RC_TOKEN": "not-a-real-token",
        "QWEN38_MEMINFO_PATH": str(meminfo),
        "QWEN38_CLOCK_SAMPLER": str(clock),
        "QWEN38_ENDPOINT_PROBE": "true",
    }
    return env, source, manifest, process_env


def run_check(env_file, process_env):
    return subprocess.run([str(RUNNER), "check", str(env_file)], cwd=ROOT,
                          env=process_env, text=True, capture_output=True)


def test_examples_are_portable_pinned_templates_without_secrets_or_fake_hashes():
    tf = (ROOT / "benchmarks/manifests/qwen38_tensorfold/tensorfold.env.example").read_text()
    cpp = (ROOT / "benchmarks/manifests/qwen38_tensorfold/vllm_cpp.env.example").read_text()
    assert TF_PIN in tf and RECIPE_PIN in tf
    assert "EXPECTED_REVISION=" in cpp
    assert "ARTIFACT_MANIFEST=" in tf and "ARTIFACT_MANIFEST=" in cpp
    for text in (tf, cpp):
        assert "/home/" not in text and "/workspace/" not in text
        assert "RC_TOKEN=" not in text
        assert "ARTIFACT_SHA256=" not in text


def test_check_accepts_clean_exact_revision_and_does_not_launch(tmp_path):
    env_file, _, _, process_env = fixture(tmp_path)
    result = run_check(env_file, process_env)
    assert result.returncode == 0, result.stderr
    assert "launched" not in result.stdout
    assert "check PASS" in result.stdout


def test_dirty_revision_fails_closed(tmp_path):
    env_file, source, _, process_env = fixture(tmp_path)
    (source / "tracked").write_text("dirty\n")
    result = run_check(env_file, process_env)
    assert result.returncode != 0
    assert "dirty" in result.stderr.lower()


def test_wrong_revision_and_missing_artifact_hash_fail_closed(tmp_path):
    env_file, _, manifest, process_env = fixture(tmp_path)
    text = env_file.read_text().replace("EXPECTED_REVISION=", "EXPECTED_REVISION=" + "0" * 40 + " # ")
    env_file.write_text(text)
    assert run_check(env_file, process_env).returncode != 0

    env_file, _, manifest, process_env = fixture(tmp_path / "second")
    manifest.write_text("weights.bin\n")
    result = run_check(env_file, process_env)
    assert result.returncode != 0
    assert "hash" in result.stderr.lower() or "manifest" in result.stderr.lower()


def test_missing_each_required_lease_variable_fails_closed(tmp_path):
    env_file, _, _, process_env = fixture(tmp_path)
    for variable in ("RC_DEVICE", "RC_JOB_ID", "RC_TOKEN"):
        candidate = dict(process_env)
        candidate.pop(variable)
        result = run_check(env_file, candidate)
        assert result.returncode != 0
        assert variable in result.stderr


def test_capture_comparison_refuses_mismatched_token_fingerprints_and_emits_no_ratio(tmp_path):
    left = {
        "refusal_reason": None, "canonical_payload_hash": "payload", "run_identity": "run",
        "tokenizer_identity": "tok", "samples": [{"refusal_reason": None,
        "payload_hash": "sample", "prompt_tokens": 1, "prompt_token_fingerprint": "left"}],
        "prompt_token_fingerprints": ["left"],
    }
    right = json.loads(json.dumps(left))
    right["samples"][0]["prompt_token_fingerprint"] = "right"
    right["prompt_token_fingerprints"] = ["right"]
    left_path, right_path = tmp_path / "left.json", tmp_path / "right.json"
    left_path.write_text(json.dumps(left)); right_path.write_text(json.dumps(right))
    out = tmp_path / "comparison.json"
    result = subprocess.run([str(RUNNER), "compare-results", str(left_path), str(right_path),
                             "artifact-a", "artifact-b", str(out)], cwd=ROOT,
                            text=True, capture_output=True)
    assert result.returncode != 0
    comparison = json.loads(out.read_text())
    assert comparison["verdict"] == "REFUSED"
    assert "ratio" not in comparison


def test_capture_source_has_timestamped_evidence_and_harness_contract():
    source = RUNNER.read_text()
    for command in ("check", "tensorfold", "vllm-cpp", "capture"):
        assert command in source
    assert "qwen38_endpoint_bench.py" in source
    assert "date -u +%Y%m%dT%H%M%SZ" in source
    assert "comparison_verdict" in source
    assert "PROFILE_COMPARISON" in source
