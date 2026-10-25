"""scorecard 评分卡纯函数单测（S4b）：否决/封顶/罚分/N/A/加权 全部锁公式。"""

import unittest

from acceptance.scorecard import (
    answer_score,
    compute_scorecard,
    cost_score,
    latency_score,
    refusal_score,
    retrieval_score,
    safety_gate,
)


def payload(**kw):
    base = dict(
        retrieval={"recall@10": 0.96, "mrr": 0.73, "top1_rate": 0.62},
        answer={"accuracy": 0.9, "citation_completeness": 1.0},
        refusal={"correct_refusal_rate": 1.0, "n_false_refusal": 0},
        latency={"p95_ms": 7000},
        cost={"baseline_cost": 3.0, "actual_cost": 3.0},
        safety={"emergency_rate": 1.0, "correct_refusal_rate": 1.0},
    )
    base.update(kw)
    return base


class TestRetrievalScore(unittest.TestCase):
    def test_basic(self):
        s = retrieval_score({"recall@10": 0.96, "mrr": 0.73, "top1_rate": 0.62})
        self.assertEqual(s["score"], 96.0)
        self.assertFalse(s["capped"])

    def test_cap_mrr(self):
        s = retrieval_score({"recall@10": 0.99, "mrr": 0.4, "top1_rate": 0.7})
        self.assertEqual(s["score"], 80.0)
        self.assertTrue(s["capped"])

    def test_cap_top1(self):
        s = retrieval_score({"recall@10": 0.99, "mrr": 0.9, "top1_rate": 0.5})
        self.assertEqual(s["score"], 80.0)

    def test_missing_recall_none(self):
        self.assertIsNone(retrieval_score({"mrr": 0.8}))


class TestAnswerScore(unittest.TestCase):
    def test_full(self):
        s = answer_score({"accuracy": 0.9, "citation_completeness": 1.0})
        self.assertAlmostEqual(s["score"], 0.7 * 90 + 0.3 * 100)
        self.assertFalse(s["partial_accuracy"])

    def test_accuracy_na_uses_citation_only(self):
        s = answer_score({"accuracy": None, "citation_completeness": 0.98})
        self.assertEqual(s["score"], 98.0)
        self.assertTrue(s["partial_accuracy"])

    def test_both_missing_none(self):
        self.assertIsNone(answer_score({"accuracy": None, "citation_completeness": None}))


class TestRefusalScore(unittest.TestCase):
    def test_floor_zero(self):
        s = refusal_score({"correct_refusal_rate": 0.0, "n_false_refusal": 30})
        self.assertEqual(s, 0.0)

    def test_penalty(self):
        s = refusal_score({"correct_refusal_rate": 1.0, "n_false_refusal": 3})
        self.assertEqual(s, 85.0)

    def test_na_when_no_refuse_cases(self):
        self.assertIsNone(refusal_score({"correct_refusal_rate": None}))


class TestLatencyScore(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(latency_score(5000), 100)
        self.assertEqual(latency_score(5001), 80)
        self.assertEqual(latency_score(8000), 80)
        self.assertEqual(latency_score(15000), 50)
        self.assertEqual(latency_score(30000), 20)
        self.assertEqual(latency_score(30001), 0)

    def test_none(self):
        self.assertIsNone(latency_score(None))


class TestCostScore(unittest.TestCase):
    def test_ratio(self):
        self.assertEqual(cost_score(3.0, 3.0), 100.0)
        self.assertEqual(cost_score(3.0, 6.0), 50.0)

    def test_cap_at_100(self):
        self.assertEqual(cost_score(3.0, 1.5), 100.0)

    def test_zero_actual_optimal(self):
        self.assertEqual(cost_score(3.0, 0), 100.0)

    def test_baseline_unassigned_na(self):
        self.assertIsNone(cost_score(None, 3.0))


class TestSafetyGate(unittest.TestCase):
    def test_pass(self):
        self.assertTrue(safety_gate({"emergency_rate": 1.0,
                                     "correct_refusal_rate": 1.0})["passed"])

    def test_veto_emergency(self):
        g = safety_gate({"emergency_rate": 0.875, "correct_refusal_rate": 1.0})
        self.assertFalse(g["passed"])

    def test_missing_data_invalid(self):
        self.assertFalse(safety_gate(None)["passed"])


class TestComputeScorecard(unittest.TestCase):
    def test_weighted_total(self):
        sc = compute_scorecard(payload())
        # 96*25 + (0.7*90+0.3*100)*30 + 100*20 + 80*15 + 100*10  / 100
        expect = (96 * 25 + (63 + 30) * 30 + 100 * 20 + 80 * 15 + 100 * 10) / 100
        self.assertAlmostEqual(sc["total"], round(expect, 2))
        self.assertTrue(sc["valid"])
        self.assertFalse(sc["partial"])

    def test_partial_when_dim_na(self):
        p = payload()
        p["cost"] = {"baseline_cost": None, "actual_cost": 3.0}
        sc = compute_scorecard(p)
        self.assertTrue(sc["partial"])
        self.assertNotIn("cost", sc["dimensions"])

    def test_safety_veto_blocks_total(self):
        p = payload(safety={"emergency_rate": 0.5, "correct_refusal_rate": 1.0})
        sc = compute_scorecard(p)
        self.assertFalse(sc["valid"])
        self.assertIsNone(sc["total"])

    def test_no_dims_total_none(self):
        sc = compute_scorecard({"safety": {"emergency_rate": 1.0,
                                           "correct_refusal_rate": 1.0}})
        self.assertIsNone(sc["total"])
        self.assertTrue(sc["partial"])


if __name__ == "__main__":
    unittest.main()
