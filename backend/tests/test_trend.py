"""trend.py（S5 面板数据层）确定性单测。

锁定：两种产物（S4b scorecard / S4c report）的提取契约、时间回退、排序、
N/A 不补 0、skipped 不静默、可比性警告、HTML 注入转义与离线零依赖。
图表 JS 的语法由 node --check 单独验证（见开发进度），渲染效果需浏览器目检。
"""

import json
import tempfile
import unittest
from pathlib import Path

from acceptance import trend


def report_payload(total=94.73, generated="2026-10-25T12:15:05", suites=True,
                   valid=True, partial=True, cost=False, promptmeta=None):
    rep = {
        "generated_at": generated,
        "scorecard": {"total": total if valid else None, "valid": valid,
                      "partial": partial, "note": "部分口径（有维度 N/A）",
                      "dimensions": {"retrieval": {"score": 96.36, "weight": 25},
                                     "answer": {"score": 97.22, "weight": 30},
                                     "refusal": {"score": 100.0, "weight": 20},
                                     "latency": {"score": 80, "weight": 15}}},
    }
    if cost:
        rep["scorecard"]["dimensions"]["cost"] = {"score": 100.0, "weight": 10}
    if suites:
        rep["sources"] = [{"suite": "scenarios", "file": "s.json", "n": 22},
                          {"suite": "coverage", "file": "c.json", "n": 110}]
    if promptmeta is not None:
        rep["promptmeta"] = promptmeta
    return rep


class TestExtractRun(unittest.TestCase):
    def _write(self, td, rel, payload):
        p = Path(td) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return p

    def test_report_shape(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._write(td, "20261025/report_s4c.json", report_payload())
            run = trend.extract_run(p)
        self.assertEqual(run["id"], "report_s4c")
        self.assertEqual(run["total"], 94.73)
        self.assertEqual(run["generated_at"], "2026-10-25T12:15:05")
        self.assertEqual(run["time_basis"], "run")
        self.assertEqual(run["suites"], ["coverage", "scenarios"])
        self.assertEqual(run["dimensions"]["retrieval"], 96.36)
        self.assertNotIn("cost", run["dimensions"])
        self.assertTrue(run["partial"])
        self.assertEqual(run["promptmeta_status"], "未采集")

    def test_scorecard_shape_dir_date_fallback(self):
        # S4b 的 scorecard_first.json：无 generated_at / 无 sources
        with tempfile.TemporaryDirectory() as td:
            p = self._write(td, "20261025/scorecard_first.json",
                            report_payload(generated=None, suites=False))
            run = trend.extract_run(p)
        self.assertEqual(run["generated_at"], "2026-10-25")
        self.assertEqual(run["time_basis"], "dir_date")
        self.assertEqual(run["suites"], ["未记录"])

    def test_no_date_anywhere(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._write(td, "x.json", report_payload(generated=None))
            run = trend.extract_run(p)
        self.assertIsNone(run["generated_at"])
        self.assertEqual(run["time_basis"], "unknown")

    def test_unrecognized_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._write(td, "coverage.json", {"records": []})
            with self.assertRaises(ValueError):
                trend.extract_run(p)

    def test_promptmeta_status(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._write(td, "r.json", report_payload(
                promptmeta={"coverage": {"hash_alg": "sha256[:12]"}}))
            self.assertEqual(trend.extract_run(p)["promptmeta_status"], "已采集")

    def test_promptmeta_from_conditions_counts_as_collected(self):
        payload = report_payload()
        payload["conditions"] = {"promptmeta": {"prompt_hashes": {"classify": "aaa"}}}
        with tempfile.TemporaryDirectory() as td:
            p = self._write(td, "r.json", payload)
            self.assertEqual(trend.extract_run(p)["promptmeta_status"], "已采集")

    def test_invalid_run_keeps_null_total(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._write(td, "r.json", report_payload(valid=False))
            run = trend.extract_run(p)
        self.assertIsNone(run["total"])
        self.assertFalse(run["valid"])


class TestCollectAndTrends(unittest.TestCase):
    def _run(self, id, generated, suites=("scenarios",), valid=True, pm="已采集"):
        return {"id": id, "file": id + ".json", "generated_at": generated,
                "time_basis": "run", "total": None if not valid else 90.0,
                "valid": valid, "partial": True, "note": None,
                "dimensions": {"retrieval": 96.36}, "suites": list(suites),
                "sources": [], "promptmeta_status": pm}

    def test_sorting_and_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            good = Path(td) / "b.json"
            good.write_text(json.dumps(report_payload()), encoding="utf-8")
            bad = Path(td) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            other = Path(td) / "other.json"
            other.write_text(json.dumps({"summary": {}}), encoding="utf-8")
            out = trend.collect_runs([bad, good, other])
        self.assertEqual(len(out["runs"]), 1)
        self.assertEqual({s["file"] for s in out["skipped"]}, {"bad.json", "other.json"})

    def test_matrix_preserves_na(self):
        runs = [self._run("r1", "2026-10-25T10:00:00")]
        trends = trend.build_trends(runs)
        row = trends["matrix"][0]
        self.assertEqual(row["total"], 90.0)
        self.assertEqual(row["retrieval"], 96.36)
        self.assertIsNone(row["cost"])       # N/A 不补 0
        self.assertIsNone(row["answer"])
        self.assertEqual(trends["warnings"], [])

    def test_suite_mismatch_warns_without_dropping(self):
        runs = [self._run("r1", "2026-10-25T10:00:00", suites=("scenarios",)),
                self._run("r2", "2026-10-25T12:00:00",
                          suites=("coverage", "scenarios"))]
        trends = trend.build_trends(runs)
        self.assertEqual(len(trends["matrix"]), 2)   # 不擅自剔除
        self.assertTrue(any("输入套件不一致" in w for w in trends["warnings"]))

    def test_missing_promptmeta_and_invalid_warn(self):
        runs = [self._run("r1", "2026-10-25", pm="未采集", valid=False)]
        warnings = trend.build_trends(runs)["warnings"]
        self.assertTrue(any("条件指纹" in w for w in warnings))
        self.assertTrue(any("安全否决" in w for w in warnings))


class TestRenderAndCli(unittest.TestCase):
    def test_render_escapes_script_close(self):
        template = Path(__file__).resolve().parents[2] / "Evaluation" / "dashboard.template.html"
        payload = {"runs": [{"id": "x</script><b>", "dimensions": {}}],
                   "skipped": [], "series": [], "matrix": [], "warnings": []}
        html = trend.render_html(payload, template)
        self.assertNotIn("__DATA__", html)
        self.assertNotIn("x</script>", html)   # JSON 内文本不得截断 script
        self.assertIn("x<\\/script>", html)

    def test_template_is_offline_single_file(self):
        template = (Path(__file__).resolve().parents[2] / "Evaluation"
                    / "dashboard.template.html").read_text(encoding="utf-8")
        self.assertNotIn("<script src", template)
        self.assertNotIn("<link", template)
        self.assertNotIn("fetch(", template)
        self.assertNotIn("https://", template)
        self.assertNotIn("http://", template)

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "reports" / "20261025"
            root.mkdir(parents=True)
            (root / "report_a.json").write_text(json.dumps(report_payload()),
                                                encoding="utf-8")
            (root / "scorecard_b.json").write_text(
                json.dumps(report_payload(total=92.88, generated=None, suites=False)),
                encoding="utf-8")
            out = Path(td) / "dash.html"
            template = Path(__file__).resolve().parents[2] / "Evaluation" / "dashboard.template.html"
            rc = trend.main([str(Path(td) / "reports"), "--out", str(out),
                             "--template", str(template)])
            self.assertEqual(rc, 0)
            html = out.read_text(encoding="utf-8")
            data = json.loads((out.parent / "trends.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["runs"]), 2)
        self.assertEqual(len(data["matrix"]), 2)
        self.assertTrue(any("输入套件不一致" in w for w in data["warnings"]))
        self.assertIn("医疗 RAG 评估趋势面板", html)
        self.assertIn("report_a", html)

    def test_discover_excludes_own_trends_output(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "a.json").write_text("{}", encoding="utf-8")
            (Path(td) / "trends.json").write_text("{}", encoding="utf-8")
            files = trend.discover(td)
        self.assertEqual([p.name for p in files], ["a.json"])


class TestCompareRuns(unittest.TestCase):
    """「这轮改了什么」：分数差 + 条件差异，三态可比性。"""

    def _run(self, id="r", total=90.0, dims=None, conditions=None):
        return {"id": id, "file": id + ".json", "generated_at": "2026-10-25",
                "time_basis": "run", "total": total, "valid": True, "partial": False,
                "note": None, "dimensions": dims or {"retrieval": 96.36},
                "suites": ["scenarios"], "sources": [], "promptmeta_status": "已采集",
                "conditions": conditions, "is_baseline": False}

    def _cond(self, classify="aaa", git="aaa111", docs=149, model="deepseek-chat"):
        return {"git_sha": git, "model": model,
                "promptmeta": {"prompt_hashes": {"classify": classify, "judge": "jjj"},
                               "safety_wordlist_hash": "sss"},
                "index_stats": {"docs": docs, "chunks": 1696},
                "testset_versions": {"scenarios": "2026-10-25"}}

    def test_deltas_and_na(self):
        prev = self._run("a", total=90.0, dims={"retrieval": 96.36, "cost": None})
        curr = self._run("b", total=92.5, dims={"retrieval": 90.0})
        cmp_ = trend.compare_runs(prev, curr)
        self.assertEqual(cmp_["deltas"]["total"]["delta"], 2.5)
        self.assertEqual(cmp_["deltas"]["retrieval"]["delta"], -6.36)
        self.assertIsNone(cmp_["deltas"]["cost"]["delta"])   # N/A 不算差值
        self.assertIsNone(cmp_["deltas"]["answer"]["delta"])

    def test_no_conditions_means_unknown_not_unchanged(self):
        cmp_ = trend.compare_runs(self._run("a"), self._run("b"))
        self.assertIsNone(cmp_["comparable"])
        self.assertIn("无法判断这轮改了什么", cmp_["reason"])

    def test_same_conditions_git_only_recorded(self):
        cmp_ = trend.compare_runs(self._run("a", conditions=self._cond(git="aaa111")),
                                  self._run("b", conditions=self._cond(git="bbb222")))
        self.assertTrue(cmp_["comparable"])          # git 不参与否决
        self.assertEqual(cmp_["condition_diffs"], [])
        self.assertIn("aaa111", cmp_["git"])

    def test_prompt_change_not_comparable(self):
        cmp_ = trend.compare_runs(self._run("a", conditions=self._cond(classify="aaa")),
                                  self._run("b", conditions=self._cond(classify="ccc")))
        self.assertFalse(cmp_["comparable"])
        self.assertTrue(any("prompt[classify]" in d for d in cmp_["condition_diffs"]))

    def test_index_and_model_change_listed(self):
        cmp_ = trend.compare_runs(
            self._run("a", conditions=self._cond()),
            self._run("b", conditions=self._cond(docs=150, model="other-model")))
        joined = "；".join(cmp_["condition_diffs"])
        self.assertIn("index_stats", joined)
        self.assertIn("other-model", joined)
        self.assertFalse(cmp_["comparable"])

    def test_build_trends_changes_link_previous(self):
        runs = [self._run("a", conditions=self._cond()),
                self._run("b", conditions=self._cond())]
        trends = trend.build_trends(runs)
        self.assertEqual(len(trends["changes"]), 1)
        self.assertEqual(trends["changes"][0]["vs"], "a")
        self.assertEqual(trends["changes"][0]["id"], "b")
        self.assertTrue(trends["changes"][0]["comparable"])

    def test_baseline_json_extraction(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "baseline.json"
            p.write_text(json.dumps({
                "appointed_at": "2026-10-25T13:00:00",
                "conditions": self._cond(),
                "scorecard": {"total": 93.59, "valid": True, "partial": False,
                              "dimensions": {"retrieval": {"score": 96.36}}}},
                ensure_ascii=False), encoding="utf-8")
            run = trend.extract_run(p)
        self.assertTrue(run["is_baseline"])
        self.assertEqual(run["generated_at"], "2026-10-25T13:00:00")
        self.assertEqual(run["time_basis"], "appointed")
        self.assertEqual(run["conditions"], self._cond())
        self.assertEqual(run["total"], 93.59)


class TestBaselineLinking(unittest.TestCase):
    """baseline.json：不重复计入趋势，只作参照 + 条件指纹唯一匹配关联。"""

    def _write(self, td, name, payload):
        p = Path(td) / name
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return p

    def _report(self, total=93.59, retrieval=96.36):
        return {"generated_at": "2026-10-25T12:49:29",
                "scorecard": {"total": total, "valid": True, "partial": False,
                              "dimensions": {"retrieval": {"score": retrieval}}}}

    def _baseline(self, total=93.59, retrieval=96.36):
        return {"appointed_at": "2026-10-25T13:05:00",
                "conditions": {"git_sha": "8995165", "promptmeta": {"prompt_hashes": {}},
                               "index_stats": {"docs": 149, "chunks": 1696}},
                "scorecard": {"total": total, "valid": True, "partial": False,
                              "dimensions": {"retrieval": {"score": retrieval}}}}

    def test_unique_match_links_conditions(self):
        with tempfile.TemporaryDirectory() as td:
            rep = self._write(td, "report_full.json", self._report())
            base = self._write(td, "baseline.json", self._baseline())
            out = trend.collect_runs([rep, base])
        self.assertEqual(len(out["runs"]), 1)          # 基线不重复计入
        run = out["runs"][0]
        self.assertTrue(run["is_baseline"])
        self.assertEqual(run["conditions_source"], "baseline")
        self.assertEqual(run["conditions"]["git_sha"], "8995165")
        self.assertEqual(out["baseline"]["total"], 93.59)

    def test_ambiguous_match_not_linked(self):
        with tempfile.TemporaryDirectory() as td:
            a = self._write(td, "report_a.json", self._report())
            b = self._write(td, "report_b.json", self._report())
            base = self._write(td, "baseline.json", self._baseline())
            out = trend.collect_runs([a, b, base])
        self.assertEqual(len(out["runs"]), 2)
        self.assertTrue(all(not r["conditions"] for r in out["runs"]))
        self.assertTrue(any("未自动关联" in s["reason"] for s in out["skipped"]))

    def test_different_score_not_linked(self):
        with tempfile.TemporaryDirectory() as td:
            rep = self._write(td, "report_full.json", self._report(total=90.0))
            base = self._write(td, "baseline.json", self._baseline())
            out = trend.collect_runs([rep, base])
        self.assertFalse(out["runs"][0]["conditions"])


if __name__ == "__main__":
    unittest.main()
