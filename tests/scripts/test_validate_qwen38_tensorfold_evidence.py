import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/bench/validate_qwen38_tensorfold_evidence.py"
SPEC = importlib.util.spec_from_file_location("qwen38_evidence", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
SOURCE = ROOT / ".agents/evidence/bench-qwen38-tensorfold-gap/20260929T180547Z"


def materialize(tmp_path: Path, file: str = "summary.json", mutate=None) -> Path:
    for source in SOURCE.iterdir():
        if source.is_file():
            shutil.copyfile(source, tmp_path / source.name)
    path = tmp_path / file
    value = json.loads(path.read_text())
    if mutate:
        mutate(value)
    path.write_text(json.dumps(value))
    return tmp_path


def test_committed_blocker_validates():
    MODULE.validate(SOURCE)


@pytest.mark.parametrize(
    ("file", "mutate"),
    [
        ("summary.json", lambda x: x.update(numbers_published=True)),
        ("summary.json", lambda x: x.update(cross_engine_ratio=1.2)),
        ("summary.json", lambda x: x["artifacts"]["tensorfold"].update(present=True)),
        ("summary.json", lambda x: x["workloads"].update(serial_decode="CAPTURED")),
        ("summary.json", lambda x: x["mtp"].update(verdict="LOADABLE")),
        ("summary.json", lambda x: x["task_outcomes"].update(task_5="BLOCKED")),
        ("lease-discovery.json", lambda x: x["jobs"][0].update(state="failed")),
        ("lease-discovery.json", lambda x: x["jobs"][0].update(job_id="raw-id")),
        ("lease-discovery.json", lambda x: x["jobs"][0].update(note="0f3f7d24-805f-4bc0-a8d5-d86c5fa564ce")),
        ("artifact-discovery.json", lambda x: x["expected_vllm_cpp_directories"][0].update(present=True)),
        ("artifact-discovery.json", lambda x: x.update(download_performed=True)),
        ("checks.json", lambda x: x["focused_tests"].update(passed=0)),
        ("checks.json", lambda x: x["focused_tests"].update(status="FAILED", failed=1)),
        ("checks.json", lambda x: x.update(approved_runner_source_sha="0" * 40)),
    ],
)
def test_blocked_schema_mutations_fail_closed(tmp_path, file, mutate):
    with pytest.raises(MODULE.EvidenceError):
        MODULE.validate(materialize(tmp_path, file, mutate))


def test_summary_lease_mismatch_fails_closed(tmp_path):
    def mutate(x):
        x["lease_checks"][0]["job_fingerprint_sha256"] = "a" * 64

    with pytest.raises(MODULE.EvidenceError, match="summary leases disagree"):
        MODULE.validate(materialize(tmp_path, "summary.json", mutate))


def test_summary_scan_mismatch_fails_closed(tmp_path):
    def mutate(x):
        x["scan_assertions"]["workspace_tensorfold_artifact_candidates"] = 1

    with pytest.raises(MODULE.EvidenceError, match="scan assertions"):
        MODULE.validate(materialize(tmp_path, "summary.json", mutate))


def test_credential_like_value_fails_closed(tmp_path):
    def mutate(x):
        x["reason"] = "Bearer abc.def"

    with pytest.raises(MODULE.EvidenceError, match="credential-like"):
        MODULE.validate(materialize(tmp_path, "checks.json", mutate))


def test_measured_schema_fails_closed_without_raw_capture(tmp_path):
    def measured(x):
        x.update(status="MEASURED", numbers_published=True, tensorfold_source_present=True)
        for item in x["artifacts"].values():
            item.update(present=True, hash_manifest_present=True)
        x["workloads"] = {key: "CAPTURED" for key in x["workloads"]}
        x["profiles"] = {key: "CAPTURED" for key in x["profiles"]}

    directory = materialize(tmp_path, "summary.json", measured)
    checks = json.loads((directory / "checks.json").read_text())
    checks.update(correctness_gate="PASSED", timing_gate="CAPTURED", profile_gate="CAPTURED")
    (directory / "checks.json").write_text(json.dumps(checks))
    with pytest.raises(MODULE.EvidenceError, match="comparison.json"):
        MODULE.validate(directory)
