"""覆盖测试集分类逻辑（acceptance.coverage.classify）的确定性单测。

锁定失败原因分类口径（任务清单第五节）：
- 检索级：命中=RETRIEVAL_OK；未命中且期望文档非空=RETRIEVAL_MISS；
  期望文档为空（已知无来源）=KNOWLEDGE_MISSING，不得冒充实覆盖。
- 链路级：completed=OK；insufficient_evidence 按检索命中细分
  EVIDENCE_INSUFFICIENT / RETRIEVAL_MISS / KNOWLEDGE_MISSING；其余 outcome 原样记录。
"""

import unittest

from acceptance.coverage import classify


def case(expect):
    return {"id": "T", "question": "q", "topic": "t", "expect_docs": expect}


class TestCoverageClassify(unittest.TestCase):
    def test_retrieval_hit(self):
        self.assertEqual(classify(case(["肥胖"]), 1, None), "RETRIEVAL_OK")
        self.assertEqual(classify(case(["肥胖"]), 5, None), "RETRIEVAL_OK")

    def test_retrieval_miss_with_expected_docs(self):
        self.assertEqual(classify(case(["肥胖"]), None, None), "RETRIEVAL_MISS")

    def test_retrieval_missing_no_expected_docs(self):
        # 已知无来源：记 KNOWLEDGE_MISSING（待补充），不得记成 RETRIEVAL_MISS
        self.assertEqual(classify(case([]), None, None), "KNOWLEDGE_MISSING")

    def test_chat_completed(self):
        self.assertEqual(classify(case(["肥胖"]), 1, "completed"), "OK")
        self.assertEqual(classify(case(["肥胖"]), None, "completed"), "OK")

    def test_chat_insufficient_with_hit(self):
        self.assertEqual(classify(case(["肥胖"]), 2, "insufficient_evidence"),
                         "EVIDENCE_INSUFFICIENT")

    def test_chat_insufficient_without_hit(self):
        self.assertEqual(classify(case(["肥胖"]), None, "insufficient_evidence"),
                         "RETRIEVAL_MISS")
        self.assertEqual(classify(case([]), None, "insufficient_evidence"),
                         "KNOWLEDGE_MISSING")

    def test_other_outcomes_recorded_verbatim(self):
        self.assertEqual(classify(case(["肥胖"]), 1, "unclassified"),
                         "OUTCOME:unclassified")
        self.assertEqual(classify(case(["肥胖"]), 1, "retrieval_failed"),
                         "OUTCOME:retrieval_failed")


class TestCoverageMetricsIntegration(unittest.TestCase):
    """阶段一接线：run_coverage / run_all_groups 输出必须带 metrics 块（打桩 client）。"""

    class _FakeClient:
        """按题目查各自期望文档的假检索（恒命中，零 IO）。"""

        def __init__(self, cases_by_question):
            self._map = cases_by_question

        def retrieve(self, question):
            expects = self._map.get(question, ["未知"])
            # top1 文档名串联该题全部期望子串——重复题（跨组同题不同期望）也恒命中
            return {"outcome": None,
                    "top": [{"doc": " ".join(expects), "distance": 0.3, "content": "c"}]}

        def chat(self, sid, q):
            return 200, {"outcome": "completed"}, 1

    @classmethod
    def setUpClass(cls):
        from acceptance.coverage import CASE_GROUPS
        merged = {}
        for _name, cases in CASE_GROUPS.values():
            for c in cases:
                # 跨组重复题（如 P2B-04/EXT-07）：期望取并集，保证打桩恒命中
                merged.setdefault(c["question"], [])
                for e in c["expect_docs"]:
                    if e not in merged[c["question"]]:
                        merged[c["question"]].append(e)
        cls._map = merged

    def _factory(self):
        m = self._map
        return type("Ctx", (), {"retrieve": lambda self, q: self._c.retrieve(q),
                                "chat": lambda self, s, q: (200, {"outcome": "completed"}, 1),
                                "_c": self._FakeClient(m)})()

    def test_run_coverage_report_has_metrics(self):
        from acceptance.coverage import run_coverage
        rep = run_coverage(do_chat=False, group="p0", client_factory=self._factory)
        self.assertIn("metrics", rep)
        m = rep["metrics"]
        self.assertEqual(m["n_valid"], 20)
        self.assertEqual(m["recall@10"], 1.0)
        self.assertEqual(m["recall@5"], 1.0)
        self.assertEqual(m["mrr"], 1.0)

    def test_run_all_groups_aggregates(self):
        from acceptance.coverage import run_all_groups, CASE_GROUPS
        rep = run_all_groups(do_chat=False, client_factory=self._factory)
        self.assertIn("overall", rep)
        self.assertIn("group_metrics", rep)
        total = sum(len(cases) for _name, cases in CASE_GROUPS.values())
        self.assertEqual(rep["overall"]["n_total"], total)
        # 打桩恒命中 → 全组 Recall@10 = 1.0
        self.assertEqual(rep["overall"]["recall@10"], 1.0)


if __name__ == "__main__":
    unittest.main()
