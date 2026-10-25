"""report.py（S4c 报告聚合）确定性单测。

覆盖：records 归一（coverage-all 展开）、安全红线、答案质量分母、时延/成本
blocked 纪律、组×归因交叉、失败/观察/blocked 样本分流、Markdown 渲染与 topN。
评分卡公式属 S4b，不在此重测：聚合测试注入 scorecard_fn 桩，真实
compute_scorecard 集成与 CLI 端到端由 TestScorecardIntegration 锁定。
"""

import unittest

from acceptance import report as rp


def stub_scorecard(payload):
    """只记录输入、返回固定形状——验证的是 report 的聚合契约，不是评分公式。"""
    return {"total": 88.0, "valid": True, "partial": True,
            "dimensions": {"retrieval": {"score": 96.36, "weight": 25, "capped": False}},
            "safety": {"passed": True, "reasons": []}, "note": "stub",
            "seen_payload": payload}


def scen(id="CS-01", flags=("answerable",), outcome="completed", expected="completed",
         sources=("高血压.pdf - 第1页",), ms=1000, blocked=False, mode="lock", **kw):
    rec = {"id": id, "flags": list(flags), "outcome": outcome,
           "expected_outcome": expected, "sources": list(sources), "ms": ms,
           "blocked": blocked, "mode": mode, "answer": "回答", "top1": "高血压.pdf",
           "scenario": "正常可答", "verdict": "PASS"}
    rec.update(kw)
    return rec


REFUSAL = {"id": "CS-05", "flags": ["refuse"], "outcome": "insufficient_evidence",
           "expected_outcome": "insufficient_evidence", "answer": "暂未找到足够依据",
           "sources": [], "ms": 500, "blocked": False, "mode": "lock", "verdict": "PASS"}


class TestRunRecords(unittest.TestCase):
    def test_plain_and_coverage_all(self):
        self.assertEqual(rp.run_records({"records": [{"id": 1}]}), [{"id": 1}])
        rep = {"groups": {"p0": {"records": [{"id": "a"}]},
                          "p1a": {"records": [{"id": "b"}, {"id": "c"}]}}}
        self.assertEqual([r["id"] for r in rp.run_records(rep)], ["a", "b", "c"])

    def test_missing_records_empty(self):
        self.assertEqual(rp.run_records({"summary": {}}), [])
        self.assertEqual(rp.run_records({}), [])

    def test_retrieval_metrics_overall_then_metrics(self):
        self.assertEqual(rp.retrieval_metrics({"overall": {"recall@10": 1.0}}),
                         {"recall@10": 1.0})
        self.assertEqual(rp.retrieval_metrics({"metrics": {"recall@10": 0.5}}),
                         {"recall@10": 0.5})
        self.assertIsNone(rp.retrieval_metrics(None))


class TestSafety(unittest.TestCase):
    def test_emergency_all_intercepted(self):
        records = [scen("CS-11", ("emergency",), "emergency_handled",
                        "emergency_handled", sources=())]
        out = rp.safety_from_scenarios(records)
        self.assertEqual(out["emergency_rate"], 1.0)
        self.assertEqual(out["n_emergency"], 1)

    def test_emergency_with_sources_or_wrong_outcome_fails(self):
        records = [
            scen("CS-11", ("emergency",), "emergency_handled", "emergency_handled",
                 sources=()),
            scen("CS-12", ("emergency",), "unclassified", "emergency_handled",
                 sources=()),
            scen("CS-13", ("emergency",), "emergency_handled", "emergency_handled",
                 sources=("某药.pdf",)),  # 急症带来源 = 不合规
        ]
        out = rp.safety_from_scenarios(records)
        self.assertEqual(out["emergency_rate"], round(1 / 3, 4))

    def test_no_emergency_is_na_not_perfect(self):
        out = rp.safety_from_scenarios([scen()])
        self.assertIsNone(out["emergency_rate"])
        self.assertIsNone(out["correct_refusal_rate"])

    def test_blocked_excluded(self):
        records = [scen("CS-11", ("emergency",), "emergency_handled",
                        "emergency_handled", sources=(), blocked=True)]
        self.assertIsNone(rp.safety_from_scenarios(records)["emergency_rate"])

    def test_correct_refusal_from_s3_metric(self):
        out = rp.safety_from_scenarios([dict(REFUSAL)])
        self.assertEqual(out["correct_refusal_rate"], 1.0)


class TestAnswerMetrics(unittest.TestCase):
    def test_citation_completeness_denominator_is_answered(self):
        records = [scen("A", sources=("s.pdf",)), scen("B", sources=()),
                   scen("C", outcome="insufficient_evidence", sources=())]
        out = rp.answer_metrics(records)
        self.assertEqual(out["n_answered"], 2)
        self.assertEqual(out["n_cited"], 1)
        self.assertEqual(out["citation_completeness"], 0.5)
        self.assertIsNone(out["accuracy"])
        self.assertEqual(out["accuracy_source"], "未核验（N/A）")

    def test_no_answered_is_na(self):
        out = rp.answer_metrics([scen(outcome="insufficient_evidence", sources=())])
        self.assertIsNone(out["citation_completeness"])

    def test_health72_n_sources_and_answer_marker_fallback(self):
        self.assertEqual(rp.answer_metrics(
            [{"outcome": "completed", "n_sources": 2}])["citation_completeness"], 1.0)
        self.assertEqual(rp.answer_metrics(
            [{"outcome": "completed", "answer": "……【参考来源】1. x.pdf"}]
        )["citation_completeness"], 1.0)

    def test_accuracy_manual_entry(self):
        out = rp.answer_metrics([scen()], accuracy=0.85)
        self.assertEqual(out["accuracy"], 0.85)
        self.assertEqual(out["accuracy_source"], "人工录入")


class TestLatencyCost(unittest.TestCase):
    def test_blocked_and_bad_ms(self):
        records = [scen(ms=1000), scen("B", ms=3000),
                   scen("C", ms=99999, blocked=True), scen("D", ms=None)]
        out = rp.latency_cost(records)
        self.assertEqual(out["latency"]["n"], 2)
        self.assertEqual(out["latency"]["p50_ms"], 1000)
        self.assertEqual(out["latency"]["p95_ms"], 3000)
        self.assertEqual(out["n_blocked"], 1)

    def test_empty_latency_none(self):
        out = rp.latency_cost([scen(ms=None)])
        self.assertIsNone(out["latency"])
        self.assertIn("cost", out)


class TestCrossAndFailures(unittest.TestCase):
    def test_cross_table_union_verdicts(self):
        rep = {"groups": {
            "p0": {"summary": {"RETRIEVAL_OK": 20}},
            "pa": {"summary": {"RETRIEVAL_OK": 11, "RETRIEVAL_MISS": 1}}}}
        cross = rp.cross_table(rep)
        self.assertEqual(cross["verdicts"], ["RETRIEVAL_MISS", "RETRIEVAL_OK"])
        self.assertEqual(cross["rows"]["pa"]["RETRIEVAL_MISS"], 1)
        self.assertEqual(cross["rows"]["p0"].get("RETRIEVAL_MISS", 0), 0)

    def test_failure_split(self):
        coverage = {"groups": {"pa": {"records": [
            {"id": "PA-03", "verdict": "RETRIEVAL_MISS", "question": "q",
             "hit_rank": None, "top1": "x.pdf"},
            {"id": "PA-01", "verdict": "RETRIEVAL_OK"}]}}}
        scenarios = [
            scen("CS-13", ("emergency",), "unclassified", "emergency_handled",
                 sources=(), verdict="FAIL"),
            scen("CS-09", ("answerable", "sentinel"), "insufficient_evidence",
                 sources=(), mode="watch", verdict="WATCH"),
            scen("CS-30", blocked=True, verdict="BLOCKED"),
            scen("CS-01"),
        ]
        other = {"health72": [{"id": "A01", "outcome": "insufficient_evidence",
                               "top_docs": [{"doc": "d.pdf"}]},
                              {"id": "B03", "outcome": "emergency_handled"}]}
        f = rp.failure_samples(coverage, scenarios, other)
        self.assertEqual([x["id"] for x in f["retrieval"]], ["PA-03"])
        self.assertEqual([x["id"] for x in f["chain"]], ["CS-13"])
        self.assertEqual([x["id"] for x in f["watch"]], ["CS-09"])
        self.assertEqual([x["id"] for x in f["blocked"]], ["CS-30"])
        self.assertEqual(f["other"][0]["top1"], "d.pdf")

    def test_emergency_handled_is_not_failure(self):
        # health72 含急症题：emergency_handled 是合规拦截，不得计入未实质回答
        other = {"health72": [{"id": "B03", "outcome": "emergency_handled"},
                              {"id": "B06", "outcome": "insufficient_evidence"}]}
        f = rp.failure_samples(None, [], other)
        self.assertEqual([x["id"] for x in f["emergency"]], ["B03"])
        self.assertEqual([x["id"] for x in f["other"]], ["B06"])

    def test_outcome_mismatch_without_verdict_still_listed(self):
        # 老 run 无 verdict 字段时，outcome≠expected 也要进失败清单
        rec = scen("CS-13", ("emergency",), "unclassified", "emergency_handled",
                   sources=(), verdict=None)
        f = rp.failure_samples(None, [rec], {})
        self.assertEqual([x["id"] for x in f["chain"]], ["CS-13"])


class TestBuildAndRender(unittest.TestCase):
    def _coverage(self):
        return {"kind": "coverage-all", "generated_at": "2026-10-25T10:31:52",
                "overall": {"n_valid": 2, "n_gap": 0, "recall@10": 1.0,
                            "recall@5": 0.5, "mrr": 0.75, "top1_rate": 0.5,
                            "coverage_rate": 1.0},
                "groups": {"p0": {"summary": {"RETRIEVAL_OK": 1},
                                  "metrics": {"n_valid": 1, "recall@10": 1.0},
                                  "records": [{"id": "P0-01", "verdict": "RETRIEVAL_OK"}]},
                           "pa": {"summary": {"RETRIEVAL_MISS": 1},
                                  "metrics": {"n_valid": 1, "recall@10": 0.0},
                                  "records": [{"id": "PA-03", "verdict": "RETRIEVAL_MISS",
                                               "question": "q"}]}},
                "promptmeta": {"hash_alg": "sha256[:12]"}}

    def _scenario_rep(self):
        return {"kind": "scenarios", "generated_at": "2026-10-25T11:45:59",
                "records": [scen(), dict(REFUSAL),
                            scen("CS-11", ("emergency",), "emergency_handled",
                                 "emergency_handled", sources=())]}

    def test_build_report_payload_contract(self):
        rep = rp.build_report(self._coverage(), self._scenario_rep(), {},
                              scorecard_fn=stub_scorecard)
        payload = rep["scorecard"]["seen_payload"]
        self.assertEqual(payload["safety"]["emergency_rate"], 1.0)
        self.assertEqual(payload["retrieval"]["recall@10"], 1.0)
        self.assertEqual(payload["answer"]["citation_completeness"], 1.0)
        self.assertIsNone(payload["answer"]["accuracy"])
        self.assertEqual(payload["refusal"]["correct_refusal_rate"], 1.0)
        self.assertEqual(payload["latency"]["p95_ms"], 1000)
        self.assertEqual(payload["cost"]["actual_cost"], 0.5)  # 仅 1 题 completed，"回答"2 字符/4
        self.assertIsNone(payload["cost"]["baseline_cost"])
        self.assertIn("completion_correct_rate", rep["context"])
        self.assertEqual(rep["promptmeta"]["coverage"], {"hash_alg": "sha256[:12]"})

    def test_render_markdown_sections_and_topn(self):
        rep = rp.build_report(self._coverage(), self._scenario_rep(), {},
                              scorecard_fn=stub_scorecard)
        sources = [{"suite": "scenarios", "file": "s.json", "n": 3,
                    "generated_at": "2026-10-25T11:45:59"}]
        md = rp.render_markdown(rep, sources, ["w: 示例警告"], top_n=1)
        for section in ("## 一、数据来源", "## 二、评分卡", "## 三、安全红线",
                        "## 四、指标明细", "## 五、组×归因交叉", "## 六、失败与观察样本",
                        "## 七、口径声明"):
            self.assertIn(section, md)
        self.assertIn("w: 示例警告", md)
        self.assertIn("PA-03", md)          # 失败样本真的列出来
        self.assertIn("前 1/1 条", md)       # topN 截断标注
        self.assertIn("88.0", md)            # 总分
        self.assertIn("估算", md)
        self.assertIn("已采集", md)          # promptmeta 状态表

    def test_render_na_not_zero(self):
        rep = rp.build_report(None, None, {"health72": {"records": [
            {"id": "A01", "outcome": "completed", "n_sources": 0, "ms": 100}]}},
            scorecard_fn=stub_scorecard)
        md = rp.render_markdown(rep, [], [])
        self.assertIn("未提供场景集 run", md)
        self.assertIn("未提供 coverage run", md)
        self.assertIn("N/A", md)


class TestScorecardIntegration(unittest.TestCase):
    """真实 S4b compute_scorecard 集成（abaeb33 合入后转硬性用例，不再跳过）。"""
    def test_real_scorecard_end_to_end(self):
        rep = rp.build_report(None, {"records": [scen(), dict(REFUSAL),
                                                 scen("CS-11", ("emergency",),
                                                      "emergency_handled",
                                                      "emergency_handled", sources=())]},
                              {})
        self.assertIn("total", rep["scorecard"])
        self.assertTrue(rep["scorecard"]["valid"])
        self.assertTrue(rep["scorecard"]["partial"])  # 检索/成本缺失 → 部分口径
        self.assertIn("refusal", rep["scorecard"]["dimensions"])
        self.assertIn("latency", rep["scorecard"]["dimensions"])

    def test_real_scorecard_safety_veto(self):
        bad = scen("CS-11", ("emergency",), "unclassified", "emergency_handled",
                   sources=())
        rep = rp.build_report(None, {"records": [bad, dict(REFUSAL)]}, {})
        self.assertFalse(rep["scorecard"]["valid"])
        self.assertIsNone(rep["scorecard"]["total"])

    def test_cli_end_to_end_writes_md_and_json(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "scen.json"
            run.write_text(json.dumps({"kind": "scenarios", "records": [
                scen(), dict(REFUSAL),
                scen("CS-11", ("emergency",), "emergency_handled",
                     "emergency_handled", sources=())]}, ensure_ascii=False),
                encoding="utf-8")
            prefix = str(Path(td) / "out" / "report")
            rc = rp.main(["--scenarios", str(run), "--out-prefix", prefix,
                          "--no-conditions"])
            self.assertEqual(rc, 0)
            data = json.loads(Path(prefix + ".json").read_text(encoding="utf-8"))
            md = Path(prefix + ".md").read_text(encoding="utf-8")
        self.assertTrue(data["scorecard"]["valid"])
        self.assertTrue(data["scorecard"]["partial"])  # 检索/成本 N/A
        self.assertIsNotNone(data["scorecard"]["total"])
        self.assertEqual(data["sources"][0]["n"], 3)
        self.assertIsNone(data["conditions"])
        self.assertIn("未采集", md)
        self.assertIn("## 七、口径声明", md)
        self.assertIn("补全正确率", md)

    def test_cli_conditions_attached_and_failure_degrades(self):
        """条件指纹：正常落盘；采集异常只警告，不影响报告产出。"""
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        for side_effect, expect_conditions in (
                ({"git_sha": "abc1234", "index_stats": {"docs": 149, "chunks": 1696}}, True),
                (RuntimeError("chroma 不可用"), False)):
            with tempfile.TemporaryDirectory() as td:
                run = Path(td) / "scen.json"
                run.write_text(json.dumps({"records": [
                    scen(), dict(REFUSAL),
                    scen("CS-11", ("emergency",), "emergency_handled",
                         "emergency_handled", sources=())]}, ensure_ascii=False),
                    encoding="utf-8")
                prefix = str(Path(td) / "report")
                with patch("acceptance.conditions.current_conditions") as cc:
                    if isinstance(side_effect, Exception):
                        cc.side_effect = side_effect
                    else:
                        cc.return_value = side_effect
                    rc = rp.main(["--scenarios", str(run), "--out-prefix", prefix])
                self.assertEqual(rc, 0)
                data = json.loads(Path(prefix + ".json").read_text(encoding="utf-8"))
            if expect_conditions:
                self.assertEqual(data["conditions"], side_effect)
            else:
                self.assertIsNone(data["conditions"])
                self.assertTrue(any("条件指纹采集失败" in w for w in data["warnings"]))

    def test_cli_chroma_dir_passthrough(self):
        """--chroma-dir 必须透传给 conditions（W1 6fe261d 的跨 worktree 约定）。"""
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "scen.json"
            run.write_text(json.dumps({"records": [
                scen(), dict(REFUSAL),
                scen("CS-11", ("emergency",), "emergency_handled",
                     "emergency_handled", sources=())]}, ensure_ascii=False),
                encoding="utf-8")
            prefix = str(Path(td) / "report")
            with patch("acceptance.conditions.current_conditions") as cc:
                cc.return_value = {"index_stats": {"docs": 149, "chunks": 1696}}
                rc = rp.main(["--scenarios", str(run), "--out-prefix", prefix,
                              "--chroma-dir", "F:/medical-assistant/backend/chroma_db"])
            self.assertEqual(rc, 0)
            self.assertEqual(cc.call_args.kwargs["chroma_dir"],
                             Path("F:/medical-assistant/backend/chroma_db"))
            data = json.loads(Path(prefix + ".json").read_text(encoding="utf-8"))
        self.assertEqual(data["conditions"]["index_stats"]["chunks"], 1696)

    def test_cli_requires_some_valid_input(self):
        with self.assertRaises(SystemExit):
            rp.main([])


if __name__ == "__main__":
    unittest.main()
