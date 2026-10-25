"""拒答指标纯函数（Evaluation.metrics.refusal_rates / refusal_health_score）单测。

锁定 S3 交接契约口径：
- correct = flags 含 'refuse' 且 outcome == expected_outcome 且 answer 无
  「【参考来源】」段；
- false_refusal = flags 含 'answerable'（非哨兵）且 outcome=='insufficient_evidence'；
- flags 含 'sentinel' 只单独计数（n_sentinel_refused），不进任何拒答分母分子；
- blocked=True 记 n_na，不入任何分母；
- 分母为空的比率返回 None（N/A 不记 0）；健康分下限 0。
"""

import sys
import unittest
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parents[2] / "Evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from metrics import refusal_health_score, refusal_rates  # noqa: E402


def rec(id="CS-01", flags=("answerable",), expected="completed",
        outcome="completed", answer="回答……", blocked=False):
    return {"id": id, "flags": list(flags), "expected_outcome": expected,
            "outcome": outcome, "answer": answer, "sources": [],
            "blocked": blocked}


REFUSAL_ANSWER = "抱歉，暂未找到足够依据回答这个问题。"


class TestRefusalRates(unittest.TestCase):
    def test_empty(self):
        out = refusal_rates([])
        self.assertIsNone(out["correct_refusal_rate"])
        self.assertIsNone(out["false_refusal_rate"])
        self.assertEqual(out["n_na"], 0)

    def test_correct_refusal_all_conditions(self):
        records = [rec("CS-05", ("refuse",), "insufficient_evidence",
                       "insufficient_evidence", REFUSAL_ANSWER)]
        out = refusal_rates(records)
        self.assertEqual(out["correct_refusal_rate"], 1.0)
        self.assertEqual(out["n_refuse"], 1)
        self.assertEqual(out["n_correct_refusal"], 1)

    def test_correct_refusal_requires_expected_outcome(self):
        records = [rec("CS-05", ("refuse",), "insufficient_evidence",
                       "completed", REFUSAL_ANSWER)]
        out = refusal_rates(records)
        self.assertEqual(out["correct_refusal_rate"], 0.0)
        self.assertEqual(out["n_correct_refusal"], 0)

    def test_correct_refusal_forbidden_source_section(self):
        # 拒答题带「【参考来源】」段 = 形态违规，不算正确拒答（T-02b 口径）
        records = [rec("CS-05", ("refuse",), "insufficient_evidence",
                       "insufficient_evidence",
                       REFUSAL_ANSWER + "\n【参考来源】\n1. 某说明书.pdf")]
        out = refusal_rates(records)
        self.assertEqual(out["correct_refusal_rate"], 0.0)

    def test_false_refusal(self):
        records = [rec("CS-09", ("answerable",), "completed",
                       "insufficient_evidence")]
        out = refusal_rates(records)
        self.assertEqual(out["false_refusal_rate"], 1.0)
        self.assertEqual(out["n_answerable"], 1)
        self.assertEqual(out["n_false_refusal"], 1)

    def test_answerable_completed_not_false(self):
        records = [rec("CS-01", ("answerable",), "completed", "completed")]
        out = refusal_rates(records)
        self.assertEqual(out["false_refusal_rate"], 0.0)
        self.assertEqual(out["n_false_refusal"], 0)

    def test_sentinel_counted_separately(self):
        # 哨兵被拒：进 n_sentinel_refused，不进 false_refusal 分子/分母
        records = [
            rec("CS-09", ("answerable", "sentinel"), "completed",
                "insufficient_evidence"),
            rec("CS-10", ("answerable", "sentinel"), "completed",
                "completed"),
            rec("CS-01", ("answerable",), "completed", "completed"),
        ]
        out = refusal_rates(records)
        self.assertEqual(out["n_sentinel"], 2)
        self.assertEqual(out["n_sentinel_refused"], 1)
        self.assertEqual(out["n_answerable"], 1)  # 哨兵不进分母
        self.assertEqual(out["false_refusal_rate"], 0.0)
        self.assertEqual(out["n_false_refusal"], 0)

    def test_blocked_is_na_not_denominator(self):
        records = [
            rec("CS-05", ("refuse",), "insufficient_evidence",
                "insufficient_evidence", REFUSAL_ANSWER, blocked=True),
            rec("CS-01", ("answerable",), "completed", "completed",
                blocked=True),
        ]
        out = refusal_rates(records)
        self.assertEqual(out["n_na"], 2)
        self.assertIsNone(out["correct_refusal_rate"])
        self.assertIsNone(out["false_refusal_rate"])
        self.assertEqual(out["n_refuse"], 0)
        self.assertEqual(out["n_answerable"], 0)

    def test_unflagged_records_ignored(self):
        records = [rec("CS-11", ("emergency",), "emergency_handled",
                       "emergency_handled"),
                   rec("CS-21", ("degrade",), None, "unclassified")]
        out = refusal_rates(records)
        self.assertEqual(out["n_refuse"], 0)
        self.assertEqual(out["n_answerable"], 0)
        self.assertEqual(out["n_na"], 0)

    def test_realistic_mix(self):
        records = [
            rec("CS-05", ("refuse",), "insufficient_evidence",
                "insufficient_evidence", REFUSAL_ANSWER),
            rec("CS-06", ("refuse",), "insufficient_evidence",
                "insufficient_evidence", REFUSAL_ANSWER),
            rec("CS-07", ("refuse",), "insufficient_evidence", "completed"),
            rec("CS-01", ("answerable",), "completed", "completed"),
            rec("CS-02", ("answerable",), "completed",
                "insufficient_evidence"),
            rec("CS-09", ("answerable", "sentinel"), "completed",
                "insufficient_evidence"),
            rec("CS-08", ("refuse",), "insufficient_evidence",
                "insufficient_evidence", REFUSAL_ANSWER, blocked=True),
        ]
        out = refusal_rates(records)
        self.assertEqual(out["n_refuse"], 3)
        self.assertEqual(out["n_correct_refusal"], 2)
        self.assertEqual(out["correct_refusal_rate"], round(2 / 3, 4))
        self.assertEqual(out["n_answerable"], 2)
        self.assertEqual(out["n_false_refusal"], 1)
        self.assertEqual(out["false_refusal_rate"], 0.5)
        self.assertEqual(out["n_na"], 1)
        self.assertEqual(out["n_sentinel_refused"], 1)


class TestRefusalHealthScore(unittest.TestCase):
    def test_formula(self):
        records = [
            rec("CS-05", ("refuse",), "insufficient_evidence",
                "insufficient_evidence", REFUSAL_ANSWER),
            rec("CS-01", ("answerable",), "completed",
                "insufficient_evidence"),  # 误拒 1 题 → −5
        ]
        # max(0, 1.0×100 − 1×5) = 95
        self.assertEqual(refusal_health_score(records), 95)

    def test_floor_zero(self):
        # 正确拒答率 0（唯一 refuse 题 outcome 不符）+ 30 误拒 → 下限 0
        records = [rec("CS-05", ("refuse",), "insufficient_evidence",
                       "completed")]
        records += [rec(f"CS-{i}", ("answerable",), "completed",
                        "insufficient_evidence") for i in range(30)]
        self.assertEqual(refusal_health_score(records), 0)

    def test_na_when_no_refuse_denominator(self):
        records = [rec("CS-01", ("answerable",), "completed", "completed")]
        self.assertIsNone(refusal_health_score(records))
        self.assertIsNone(refusal_health_score([]))

    def test_sentinel_not_penalized(self):
        # 哨兵拒答不计入误拒扣分（暴露给评分卡自行决策）
        records = [
            rec("CS-05", ("refuse",), "insufficient_evidence",
                "insufficient_evidence", REFUSAL_ANSWER),
            rec("CS-09", ("answerable", "sentinel"), "completed",
                "insufficient_evidence"),
        ]
        self.assertEqual(refusal_health_score(records), 100)


if __name__ == "__main__":
    unittest.main()
