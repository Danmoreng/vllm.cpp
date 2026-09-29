import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/bench/validate_qwen38_tensorfold_evidence.py"
SPEC = importlib.util.spec_from_file_location("qwen38_evidence", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
SOURCE = ROOT / ".agents/evidence/bench-qwen38-tensorfold-gap/20260929T180547Z"


def materialize(tmp_path: Path, mutate=None) -> Path:
    for source in SOURCE.iterdir():
        if source.is_file():
            (tmp_path / source.name).write_bytes(source.read_bytes())
    summary = json.loads((tmp_path / "summary.json").read_text())
    if mutate:
        mutate(summary)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    return tmp_path


def test_committed_blocker_validates():
    MODULE.validate(SOURCE)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x.update(numbers_published=True),
        lambda x: x.update(cross_engine_ratio=1.2),
        lambda x: x["lease_checks"][0].update(job_id="raw-id"),
        lambda x: x["artifacts"]["tensorfold"].update(present=True),
        lambda x: x["workloads"].update(serial_decode="CAPTURED"),
        lambda x: x["mtp"].update(verdict="LOADABLE"),
    ],
)
def test_blocked_schema_fails_closed(tmp_path, mutate):
    with pytest.raises(MODULE.EvidenceError):
        MODULE.validate(materialize(tmp_path, mutate))


def test_measured_schema_fails_closed_without_raw_capture(tmp_path):
    def measured(x):
        x.update(status="MEASURED", numbers_published=True, tensorfold_source_present=True)
        for item in x["artifacts"].values():
            item.update(present=True, hash_manifest_present=True)
        x["workloads"] = {key: "CAPTURED" for key in x["workloads"]}
        x["profiles"] = {key: "CAPTURED" for key in x["profiles"]}

    with pytest.raises(MODULE.EvidenceError, match="comparison.json"):
        MODULE.validate(materialize(tmp_path, measured))
