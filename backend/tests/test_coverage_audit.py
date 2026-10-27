"""testing/tools/coverage_audit.py 的覆盖口径与引用校验测试。"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "coverage_audit", ROOT / "testing" / "tools" / "coverage_audit.py")
coverage_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage_audit)


def _slot(slot_id="S-1", minimum=1, cases=None, units=None, risk="normal"):
    return {
        "id": slot_id, "layer": "system_capability", "title": slot_id,
        "requirements": ["F8"], "risk": risk, "min_cases": minimum,
        "cases": cases or [], "unit_tests": units or [], "next_action": "补测",
    }


def _write_matrix(root, slots):
    (root / "testing" / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "testing" / "coverage_matrix.json").write_text(json.dumps({
        "name": "test_matrix", "version": "v1",
        "policy": {"stable_design_target": 0.9, "critical_design_target": 1.0},
        "slots": slots,
    }), encoding="utf-8")


def _write_dataset(root, name, state):
    (root / "testing" / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "testing" / "datasets" / f"{name}.json").write_text(json.dumps({
        "name": name, "version": "v1", "review_status": state,
        "cases": [{"id": "C-1", "question": "q"}],
    }), encoding="utf-8")


class TestCoverageAudit(unittest.TestCase):
    def test_repository_matrix_references_resolve(self):
        result = coverage_audit.audit(ROOT)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["overall"]["total"], 40)
        self.assertEqual(result["overall"]["design_covered"], 40)

    def test_draft_is_design_only_not_stable_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(root, "stable", "approved")
            _write_dataset(root, "draft", "draft")
            _write_matrix(root, [
                _slot("S-1", cases=["stable:C-1"]),
                _slot("S-2", cases=["draft:C-1"]),
                _slot("S-3", minimum=2, cases=["draft:C-1"]),
            ])
            result = coverage_audit.audit(root)
        self.assertEqual(result["overall"]["stable_covered"], 1)
        self.assertEqual(result["overall"]["design_covered"], 2)
        self.assertEqual([row["status"] for row in result["slots"]],
                         ["covered", "draft_only", "gap"])

    def test_case_review_status_can_be_promoted_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(root, "mixed", "draft")
            _write_matrix(root, [_slot(cases=["mixed:C-1"])])
            path = root / "testing" / "datasets" / "mixed.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["cases"][0]["review_status"] = "approved"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = coverage_audit.audit(root)
        self.assertTrue(result["slots"][0]["stable_covered"])

    def test_unit_reference_can_target_individual_test_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_matrix(root, [_slot(units=[
                "frontend/App.test.js::alpha works",
                "frontend/App.test.js::beta works",
            ], minimum=2)])
            (root / "frontend").mkdir()
            (root / "frontend" / "App.test.js").write_text(
                "test('alpha works')\ntest('beta works')\n", encoding="utf-8")
            result = coverage_audit.audit(root)
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["slots"][0]["stable_covered"])

    def test_report_does_not_invent_execution_or_pass_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(root, "draft", "draft")
            _write_matrix(root, [_slot(cases=["draft:C-1"])])
            result = coverage_audit.audit(root)
        report = coverage_audit.render_markdown(result)
        self.assertIn("实际执行覆盖率 / 通过率 | N/A", report)
        self.assertIn("仅草稿", report)


if __name__ == "__main__":
    unittest.main()
