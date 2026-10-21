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


if __name__ == "__main__":
    unittest.main()
