"""trace 时延/成本聚合（acceptance.traceagg）确定性单测。

锁定口径：
- 百分位 = nearest-rank（第 ceil(p/100×n) 个，1 起）；
- 时延分档：P95 ≤5s→100 / ≤8s→80 / ≤15s→50 / ≤30s→20 / >30s→0；
- 成本估算：completed 题 3 次调用/题，token≈answer 字符/4，blocked 剔除；
- aggregate_runs：缺 records / 坏文件 → warning 不中断。
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

from acceptance.traceagg import (aggregate_runs, cost_estimate,
                                 latency_summary)


class TestLatencySummary(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(latency_summary([]))

    def test_nearest_rank_percentile(self):
        out = latency_summary([100, 200, 300, 400])
        # p50: ceil(0.5×4)=2 → 200；p95: ceil(0.95×4)=4 → 400
        self.assertEqual(out["p50_ms"], 200)
        self.assertEqual(out["p95_ms"], 400)
        self.assertEqual(out["n"], 4)

    def test_nearest_rank_n5(self):
        out = latency_summary([5, 1, 2, 3, 4])  # 排序后 1..5
        self.assertEqual(out["p50_ms"], 3)   # ceil(2.5)=3
        self.assertEqual(out["p95_ms"], 5)   # ceil(4.75)=5

    def test_tier_boundaries(self):
        # 构造 n=20：p50=100；p95 为第 19 个值（nearest-rank，index 18），
        # 恰好落在各档边界
        for limit_ms, tier in ((5000, 100), (8000, 80),
                               (15000, 50), (30000, 20)):
            out = latency_summary([100] * 18 + [limit_ms, limit_ms + 1])
            self.assertEqual(out["p95_ms"], limit_ms)
            self.assertEqual(out["tier"], tier, f"p95={limit_ms}ms")
        out = latency_summary([100] * 18 + [30001, 30002])
        self.assertEqual(out["tier"], 0)

    def test_tier_uses_p95_not_p50(self):
        # p50 很小但 p95 超档 → 按 p95 计
        out = latency_summary([100] * 18 + [15000, 16000])
        self.assertEqual(out["p50_ms"], 100)
        self.assertEqual(out["p95_ms"], 15000)
        self.assertEqual(out["tier"], 50)


class TestCostEstimate(unittest.TestCase):
    def test_completed_only_estimation(self):
        records = [
            {"outcome": "completed", "answer": "abcd" * 100},  # 400 字符
            {"outcome": "completed", "answer": "a" * 40},      # 40 字符
            {"outcome": "insufficient_evidence", "answer": "x" * 400},
            {"outcome": "completed", "answer": "z" * 44, "blocked": True},
        ]
        out = cost_estimate(records)
        self.assertEqual(out["avg_calls_per_completed"], 3.0)
        self.assertEqual(out["n_completed"], 2)
        self.assertEqual(out["est_input_tokens"], 110.0)  # 440/4
        self.assertEqual(out["note"], "估算")

    def test_no_completed(self):
        out = cost_estimate([{"outcome": "insufficient_evidence"}])
        self.assertIsNone(out["avg_calls_per_completed"])
        self.assertEqual(out["est_input_tokens"], 0)
        self.assertEqual(out["n_completed"], 0)

    def test_missing_answer_counts_zero_chars(self):
        out = cost_estimate([{"outcome": "completed"}])
        self.assertEqual(out["est_input_tokens"], 0.0)


class TestAggregateRuns(unittest.TestCase):
    def _write(self, dirpath, name, payload):
        p = Path(dirpath) / name
        p.write_text(json.dumps(payload, ensure_ascii=False),
                     encoding="utf-8")
        return str(p)

    def test_multi_file_aggregation(self):
        with tempfile.TemporaryDirectory() as td:
            good1 = self._write(td, "run_a.json", {"records": [
                {"ms": 100, "outcome": "completed", "answer": "a" * 40},
                {"ms": 300, "outcome": "completed", "answer": "b" * 40},
            ]})
            good2 = self._write(td, "run_b.json", {"records": [
                {"ms": 200, "outcome": "insufficient_evidence"},
            ]})
            agg = aggregate_runs([good1, good2])
        self.assertEqual(agg["sources"], ["run_a.json", "run_b.json"])
        self.assertEqual(agg["warnings"], [])
        self.assertEqual(agg["n_blocked"], 0)
        # ms 汇总 [100, 300, 200] → p50=200, p95=300
        self.assertEqual(agg["latency"]["p50_ms"], 200)
        self.assertEqual(agg["latency"]["p95_ms"], 300)
        self.assertEqual(agg["latency"]["n"], 3)
        self.assertEqual(agg["cost"]["n_completed"], 2)
        self.assertEqual(agg["cost"]["est_input_tokens"], 20.0)

    def test_missing_records_warns_and_continues(self):
        with tempfile.TemporaryDirectory() as td:
            no_records = self._write(td, "gate.json", {"summary": {}})
            not_json = Path(td) / "broken.json"
            not_json.write_text("{not json", encoding="utf-8")
            good = self._write(td, "ok.json",
                               {"records": [{"ms": 1000}]})
            agg = aggregate_runs([no_records, str(not_json), good,
                                  str(Path(td) / "absent.json")])
        self.assertEqual(agg["sources"], ["ok.json"])
        self.assertEqual(len(agg["warnings"]), 3)
        self.assertEqual(agg["latency"]["p50_ms"], 1000)

    def test_blocked_excluded_from_latency_and_cost(self):
        with tempfile.TemporaryDirectory() as td:
            good = self._write(td, "run.json", {"records": [
                {"ms": 100, "outcome": "completed", "answer": "a" * 40},
                {"ms": 99999, "outcome": "completed", "answer": "x" * 400,
                 "blocked": True},
            ]})
            agg = aggregate_runs([good])
        self.assertEqual(agg["n_blocked"], 1)
        self.assertEqual(agg["latency"]["n"], 1)  # 99999ms 不计入时延样本
        self.assertEqual(agg["cost"]["n_completed"], 1)

    def test_non_numeric_ms_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            good = self._write(td, "run.json", {"records": [
                {"ms": 100}, {"ms": None}, {"ms": "slow"}, {"ms": True},
            ]})
            agg = aggregate_runs([good])
        self.assertEqual(agg["latency"]["n"], 1)


class TestPercentileMath(unittest.TestCase):
    """与手算对照（验收标准：p50/p95 与手算一致）。"""

    def test_against_manual_computation(self):
        import random
        rng = random.Random(42)
        vals = [rng.uniform(500, 8000) for _ in range(72)]
        out = latency_summary(vals)
        s = sorted(vals)
        manual_p50 = s[math.ceil(0.50 * 72) - 1]
        manual_p95 = s[math.ceil(0.95 * 72) - 1]
        self.assertEqual(out["p50_ms"], manual_p50)
        self.assertEqual(out["p95_ms"], manual_p95)


if __name__ == "__main__":
    unittest.main()
