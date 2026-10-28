"""抽取器评测种子集的结构校验：只验数据集自身，不测抽取逻辑（终端 2 范围）。

确定性、无 LLM、无数据库：枚举合法、字段一致、虚构标记、时间语义提示完整。
"""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "testing" / "datasets" / "case_extraction_seed.json"


def _load():
    return json.loads(SEED.read_text(encoding="utf-8"))


class TestCaseExtractionSeed(unittest.TestCase):
    def test_structure_and_count(self):
        payload = _load()
        cases = payload["cases"]
        self.assertEqual(len(cases), payload["case_count"])
        ids = [c["id"] for c in cases]
        self.assertEqual(len(set(ids)), len(ids))          # ID 唯一

    def test_all_cases_marked_fictional(self):
        for case in _load()["cases"]:
            self.assertTrue(case["fictional"], case["id"])
            self.assertIn("虚构", case["raw_note"], case["id"])

    def test_expected_enums_legal(self):
        payload = _load()
        fact_types = set(payload["fact_type_enum"])
        polarities = set(payload["polarity_enum"])
        certainties = set(payload["certainty_enum"])
        med_types = set(payload["medication_event_type_enum"])
        for case in payload["cases"]:
            exp = case["expected"]
            for fact in exp["facts"]:
                self.assertIn(fact["fact_type"], fact_types, case["id"])
                self.assertIn(fact["polarity"], polarities, case["id"])
                self.assertIn(fact["certainty"], certainties, case["id"])
            for med in exp["medications"]:
                self.assertIn(med["event_type"], med_types, case["id"])

    def test_semantic_focus_cases_present(self):
        """关键语义考察点必须在集内：否定、可能、停药、仅提及、文本观察值。"""
        cases = _load()["cases"]
        all_facts = [f for c in cases for f in c["expected"]["facts"]]
        all_meds = [m for c in cases for m in c["expected"]["medications"]]
        all_obs = [o for c in cases for o in c["expected"]["observations"]]
        self.assertTrue(any(f["polarity"] == "negative" for f in all_facts))
        self.assertTrue(any(f["certainty"] == "possible" for f in all_facts))
        self.assertTrue(any(m["event_type"] == "reported_stopped" for m in all_meds))
        self.assertTrue(any(m["event_type"] == "mentioned" for m in all_meds))
        self.assertTrue(any(m["event_type"] == "prescribed" for m in all_meds))
        self.assertTrue(any("value_text" in o for o in all_obs))
        self.assertTrue(any(o.get("reference_high") for o in all_obs))

    def test_ce008_no_fabricated_drug_name(self):
        """CE-008 药名缺失：expected 允许 None，但必须有不得编造的说明。"""
        case = next(c for c in _load()["cases"] if c["id"] == "CE-008")
        med = case["expected"]["medications"][0]
        self.assertIsNone(med["medication_name"])
        self.assertIn("不得编造", med["note"])


if __name__ == "__main__":
    unittest.main()
