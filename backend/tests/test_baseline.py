"""S4d：baseline 任命规则与条件指纹可比性校验的确定性单测。"""

import json
import tempfile
import unittest
from pathlib import Path

from acceptance.baseline import appoint_baseline, compare_with_baseline
from acceptance.conditions import fingerprint_differences


def _sc(total=90.0, partial=False, valid=True):
    dims = {k: {"score": 90 + i, "weight": w}
            for i, (k, w) in enumerate([("retrieval", 25), ("answer", 30),
                                        ("refusal", 20), ("latency", 15), ("cost", 10)])}
    if partial:
        dims.pop("cost")
    return {"total": total, "valid": valid, "partial": partial, "dimensions": dims}


def _report(total=90.0, **kw):
    return {"generated_at": "2026-10-25T12:00:00", "scorecard": _sc(total, **kw)}


def _cond(sha="abc123", chunks=1696, ph="h1"):
    return {"git_sha": sha,
            "promptmeta": {"prompt_hashes": {"classify": ph}, "safety_wordlist_hash": "sw1"},
            "index_stats": {"docs": 149, "chunks": chunks},
            "testset_versions": {"scenarios": "2026-10-25"},
            "suites_executed": ["scenarios"]}


class TestAppointBaseline(unittest.TestCase):
    def test_full_run_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "baseline.json"
            rec = appoint_baseline(_report(), _cond(), out)
            self.assertTrue(out.exists())
            self.assertEqual(rec["scorecard"]["total"], 90.0)

    def test_partial_run_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as cm:
                appoint_baseline(_report(partial=True), _cond(), Path(td) / "b.json")
            self.assertIn("全量", str(cm.exception))

    def test_invalid_scorecard_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                appoint_baseline(_report(valid=False), _cond(), Path(td) / "b.json")


class TestFingerprintDifferences(unittest.TestCase):
    def test_same_conditions_empty(self):
        self.assertEqual(fingerprint_differences(_cond(), _cond()), [])

    def test_git_diff_recorded_not_vetoed(self):
        # 校准（2026-10-25）：git_sha 仅记录不否决——否则优化提交后无法前后对比
        self.assertEqual(fingerprint_differences(_cond(), _cond(sha="def456")), [])

    def test_prompt_diff_detected(self):
        d = fingerprint_differences(_cond(), _cond(ph="h2"))
        self.assertTrue(any("prompt" in x for x in d))

    def test_index_diff_detected(self):
        d = fingerprint_differences(_cond(), _cond(chunks=1700))
        self.assertTrue(any("index_stats" in x for x in d))

    def test_suites_not_compared(self):
        a, b = _cond(), _cond()
        b["suites_executed"] = ["coverage"]
        self.assertEqual(fingerprint_differences(a, b), [])


class TestCompareWithBaseline(unittest.TestCase):
    def _with_baseline(self, td, total=90.0):
        out = Path(td) / "baseline.json"
        appoint_baseline(_report(total=total), _cond(), out)
        return out

    def test_same_conditions_diff_computed(self):
        with tempfile.TemporaryDirectory() as td:
            b = self._with_baseline(td, total=90.0)
            result = compare_with_baseline(_report(total=92.5), _cond(), b)
            self.assertTrue(result["comparable"])
            self.assertEqual(result["total_delta"], 2.5)
            self.assertIn("retrieval", result["dimension_deltas"])

    def test_changed_conditions_forbid_diff(self):
        with tempfile.TemporaryDirectory() as td:
            b = self._with_baseline(td)
            result = compare_with_baseline(_report(total=99.0), _cond(chunks=1700), b)
            self.assertFalse(result["comparable"])
            self.assertIn("需重定基线", result["reason"])
            self.assertNotIn("total_delta", result)

    def test_no_baseline_file(self):
        with tempfile.TemporaryDirectory() as td:
            result = compare_with_baseline(_report(), _cond(), Path(td) / "none.json")
            self.assertFalse(result["comparable"])


if __name__ == "__main__":
    unittest.main()
