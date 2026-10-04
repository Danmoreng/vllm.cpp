import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "b70_resource_receipt", Path(__file__).resolve().parents[2] /
    "tools/bench/b70_resource_receipt.py")
receipt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receipt)


class ResourceReceiptTests(unittest.TestCase):
    def test_observed_limits_deltas_and_unavailable_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "memory.max").write_text("12884901888\n")
            (root / "memory.swap.max").write_text("max\n")
            (root / "memory.events").write_text("high 2\noom 0\n")
            (root / "cpu.stat").write_text("nr_throttled 3\nthrottled_usec 70\n")
            before = receipt.snapshot(root)
            (root / "cpu.stat").write_text("nr_throttled 5\nthrottled_usec 140\n")
            after = receipt.snapshot(root)
            self.assertEqual(before["memory.max"]["raw"], "12884901888")
            self.assertEqual(before["memory.swap.max"]["raw"], "max")
            self.assertIn("unavailable", before["memory.swap.peak"])
            self.assertEqual(receipt.counter_deltas(before, after), {
                "memory.events": {"high": 0, "oom": 0},
                "cpu.stat": {"nr_throttled": 2, "throttled_usec": 70}})

    def test_worker_failure_is_preserved_and_receipt_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "receipt.json"
            self.assertEqual(receipt.run([sys.executable, "-c", "raise SystemExit(77)"],
                                         output, root), 77)
            recorded = json.loads(output.read_text())
            self.assertEqual(recorded["worker_exit_code"], 77)
            self.assertGreater(recorded["worker_pid"], 0)
            self.assertEqual(len(recorded["worker_executable_sha256"]), 64)
            original = output.read_bytes()
            marker = root / "should-not-run"
            with self.assertRaises(FileExistsError):
                receipt.run([sys.executable, "-c",
                             "from pathlib import Path; Path(%r).touch()" % str(marker)],
                            output, root)
            self.assertEqual(output.read_bytes(), original)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
