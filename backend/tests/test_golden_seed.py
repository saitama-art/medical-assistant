"""黄金种子审核稿的结构、证据哈希和审核页测试。"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "golden_seed_tool", ROOT / "testing" / "tools" / "golden_seed.py")
golden_seed_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(golden_seed_tool)


def _draft_with_fake_docs(root: Path) -> dict:
    source = json.loads((ROOT / "testing" / "datasets" / "golden_seed.json").read_text(encoding="utf-8"))
    (root / "testing" / "datasets").mkdir(parents=True)
    for case in source["cases"]:
        for item in case["evidence"]:
            document = root / "backend" / "data" / item["document"]
            document.parent.mkdir(parents=True, exist_ok=True)
            document.touch()
    (root / "testing" / "datasets" / "golden_seed.json").write_text(
        json.dumps(source, ensure_ascii=False), encoding="utf-8")
    return source


class TestGoldenSeed(unittest.TestCase):
    def test_repository_seed_p0_approved_rest_pending(self):
        """2026-10-28：P0 11 条经用户人工终审通过（approved 含审核人/日期），
        其余 21 条仍为 pending_user。"""
        payload, errors, warnings = golden_seed_tool.validate(ROOT)
        self.assertEqual(errors, [])
        self.assertEqual(len(payload["cases"]), 32)
        self.assertEqual(sum(len(case["evidence"]) for case in payload["cases"]), 22)
        p0 = [case for case in payload["cases"] if case["review_priority"] == "P0"]
        rest = [case for case in payload["cases"] if case["review_priority"] != "P0"]
        self.assertEqual(len(p0), 11)
        self.assertEqual(len(rest), 21)
        for case in p0:
            review = case["review"]
            self.assertEqual(review["status"], "approved", case["id"])
            self.assertTrue(review.get("reviewer"), case["id"])
            self.assertTrue(review.get("reviewed_at"), case["id"])
        self.assertTrue(all(case["review"]["status"] == "pending_user" for case in rest))
        self.assertEqual(len(warnings), 5)

    def test_quote_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = _draft_with_fake_docs(root)
            payload["cases"][0]["evidence"][0]["quote"] = "被篡改的证据内容，长度满足最低要求。"
            (root / "testing" / "datasets" / "golden_seed.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            _payload, errors, _warnings = golden_seed_tool.validate(root)
        self.assertTrue(any("quote_sha256" in error for error in errors))

    def test_approval_requires_reviewer_and_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = _draft_with_fake_docs(root)
            payload["cases"][0]["review"]["status"] = "approved"
            (root / "testing" / "datasets" / "golden_seed.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            _payload, errors, _warnings = golden_seed_tool.validate(root)
        self.assertTrue(any("审核人和日期" in error for error in errors))

    def test_review_page_prioritizes_safety_and_keeps_evidence(self):
        payload, errors, _warnings = golden_seed_tool.validate(ROOT)
        self.assertEqual(errors, [])
        page = golden_seed_tool.render_review(payload)
        self.assertLess(page.index("## P0"), page.index("## P1"))
        self.assertIn("GOLD-021", page)
        self.assertIn("我被狗咬了", page)
        self.assertIn("漏服药的补服原则", page)
        self.assertIn("不是已批准医学金标准", page)


if __name__ == "__main__":
    unittest.main()
