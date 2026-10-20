"""验收 harness 自身的确定性单测（不依赖真实 LLM / 真实索引）。

覆盖任务书要求：用例选择、API 失败不误记通过、密钥脱敏、FR-01 诊断记录可保存。
真实 LLM 验收由 `python -m acceptance` 负责，不属于本文件。
"""

import json
import unittest
from unittest.mock import MagicMock

from acceptance import runner, support
from acceptance.cases import CASE_GENERATORS, fr01_cases, _t03
from acceptance.fr01 import run_fr01_sample


class _FakeClient:
    """按问题关键词返回固定响应的假 TestClient。"""

    def __init__(self, mode="ok"):
        self.mode = mode
        self.calls = 0

    def post(self, path, json=None, timeout=None):
        self.calls += 1
        resp = MagicMock()
        resp.headers = {"content-type": "application/json"}
        if self.mode == "raise":
            raise ConnectionError(" simulated network down")
        resp.status_code = 500 if self.mode == "http500" else 200
        if self.mode == "bad_shape":
            resp.json.return_value = {"unexpected": True}
        else:
            resp.json.return_value = {
                "answer": "请立即拨打120或前往急诊。",
                "sources": [], "outcome": "emergency_handled",
            }
        return resp


class TestSelectCases(unittest.TestCase):
    def test_default_selects_all_groups(self):
        sel = runner.select_cases(CASE_GENERATORS, only=None, fr01=False)
        self.assertEqual([g for g, _, _ in sel],
                         ["T-03", "T-02b", "T-06③", "T-05⑥", "T-12", "T-13"])

    def test_group_only(self):
        sel = runner.select_cases(CASE_GENERATORS, only="T-03", fr01=False)
        self.assertEqual([g for g, _, _ in sel], ["T-03"])

    def test_fr01_appended(self):
        sel = runner.select_cases(CASE_GENERATORS, only="T-03", fr01=True)
        ids = [c["id"] if isinstance(c, dict) else g for g, c, _ in sel]
        self.assertIn("FR-01-1", ids)
        self.assertEqual(len([i for i in ids if str(i).startswith("FR-01")]), 8)


class TestFailNotPass(unittest.TestCase):
    def test_api_down_is_blocked_not_pass(self):
        report = runner.run_report(only="T-03", client=_FakeClient(mode="raise"),
                                   out_path=None)
        pf = [r for r in report["results"] if r.get("kind") != "diagnostic"]
        self.assertTrue(pf)
        for r in pf:
            self.assertFalse(r.get("ok"))
            self.assertEqual(r["status"], "blocked")
            self.assertIn("error", r)
        self.assertTrue(report["summary"]["api_blocked"])

    def test_wrong_shape_is_fail_not_pass(self):
        report = runner.run_report(only="T-03", client=_FakeClient(mode="bad_shape"),
                                   out_path=None)
        for r in report["results"]:
            self.assertFalse(r.get("ok"))
            self.assertEqual(r["status"], "fail")

    def test_success_shape_passes(self):
        report = runner.run_report(only="T-03", client=_FakeClient(mode="ok"),
                                   out_path=None)
        for r in report["results"]:
            self.assertTrue(r["ok"])
            self.assertEqual(r["status"], "pass")


class TestRedact(unittest.TestCase):
    def test_sk_key_masked(self):
        s = support.redact("error auth sk-abcdef1234567890XYZ end")
        self.assertNotIn("sk-abcdef", s)
        self.assertIn("REDACTED", s)

    def test_key_value_masked(self):
        s = support.redact('{"api_key": "deadbeefcafe1234"}')
        self.assertNotIn("deadbeef", s)

    def test_report_payload_has_no_secret(self):
        import os
        os.environ["DEEPSEEK_API_KEY"] = "sk-testsecret000111222"
        try:
            report = runner.run_report(only="T-03", client=_FakeClient(mode="raise"),
                                       out_path=None)
            blob = json.dumps(report, ensure_ascii=False)
            self.assertNotIn("sk-testsecret", blob)
        finally:
            del os.environ["DEEPSEEK_API_KEY"]


class TestFr01Diagnostic(unittest.TestCase):
    def _ctx(self):
        ctx = runner.ClientCtx(client=_FakeClient(mode="ok"))
        ctx.retrieve = lambda q: {
            "outcome": None,
            "top": [{"doc": "失眠_默沙东诊疗手册.pdf", "distance": 0.31,
                     "content": "改善睡眠卫生：保持规律作息，睡前避免咖啡因。"}],
        }
        return ctx

    def test_record_saved_with_chain_facts(self):
        rec = run_fr01_sample(self._ctx(), "失眠怎么办？", "失眠")
        self.assertIn("retrieval", rec)
        self.assertEqual(rec["retrieval"]["expect_doc_hit_rank"], 1)
        self.assertIn("answer", rec)
        self.assertIn("auto_support_t16lite", rec)

    def test_fr01_cases_are_diagnostic_kind(self):
        for c in fr01_cases():
            self.assertEqual(c["kind"], "diagnostic")

    def test_retrieve_error_still_records(self):
        ctx = self._ctx()
        def boom(q):
            raise RuntimeError("vector store missing")
        ctx.retrieve = boom
        rec = run_fr01_sample(ctx, "失眠怎么办？", "失眠")
        self.assertIn("error", rec["retrieval"])
        self.assertIn("answer", rec)   # 链路部分仍记录

    def test_support_metric_math(self):
        r = support.char_ngram_support(
            "睡前避免咖啡因，保持规律作息。这句话完全是凭空编造的结论内容。",
            ["改善睡眠卫生：睡前避免咖啡因，保持规律作息，避免熬夜。"])
        self.assertEqual(r["sentences"], 2)
        self.assertEqual(r["supported"], 1)
        self.assertEqual(r["ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
