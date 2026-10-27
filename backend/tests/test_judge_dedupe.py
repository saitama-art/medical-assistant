"""judge 上下文按文档去重（O2 修复）的单测：去重语义 + 接线 + 回归边界。"""

import unittest

from app.rag import service
from app.medical_agent.state import NodeStatus, RetrievedChunk, RetrievalResult


def _c(doc, text, dist=0.3):
    return RetrievedChunk(content=text, document=doc, page=1, distance=dist)


class TestDedupeForJudge(unittest.TestCase):
    def test_per_doc_cap_and_order(self):
        chunks = [_c("A.pdf", f"a{i}") for i in range(5)] + \
                 [_c("B.pdf", f"b{i}") for i in range(3)] + \
                 [_c("C.pdf", "c0")]
        out = service._dedupe_for_judge(chunks)
        # 每文档至多 2 段：A 2 + B 2 + C 1 = 5；保持原排名顺序
        self.assertEqual([c.content for c in out], ["a0", "a1", "b0", "b1", "c0"])

    def test_top1_preserved(self):
        chunks = [_c("A.pdf", "a0"), _c("A.pdf", "a1"), _c("B.pdf", "b0")]
        out = service._dedupe_for_judge(chunks)
        self.assertIs(out[0], chunks[0])

    def test_no_duplication_when_already_diverse(self):
        chunks = [_c("A.pdf", "a"), _c("B.pdf", "b")]
        self.assertEqual(len(service._dedupe_for_judge(chunks)), 2)


class TestJudgeUsesDedupedContext(unittest.TestCase):
    def test_judge_receives_deduped_chunks(self):
        # 模拟 CS-09 形态：A 文档 5 段、B 文档 3 段、C 文档 3 段（答案段在 rank 7）
        docs = [("孕期用药.pdf", 5), ("孕期自我保健.pdf", 3), ("孕期母体变化.pdf", 3)]
        chunks = []
        for doc, n in docs:
            for i in range(n):
                chunks.append(_c(doc, f"{doc}段{i}"))
        # 答案段（C 文档第 2 段，原 rank 7）带可逐字校验的引文
        answer_chunk = _c("孕期母体变化.pdf", "胃灼热和胃痛：处理建议正文")
        chunks[5 + 3 + 1] = answer_chunk
        retrieval = RetrievalResult(outcome=None, chunks=chunks)
        captured = {}

        def fake_judge(question, ch, llm=None, extra_instruction=""):
            captured["chunks"] = ch
            return True, "胃灼热和胃痛"

        original = service._judge_by_llm
        service._judge_by_llm = fake_judge
        try:
            result = service.judge_evidence("孕期常见不适有哪些一般处理原则？", retrieval)
        finally:
            service._judge_by_llm = original
        self.assertIs(result.status, NodeStatus.OK)
        j = result.value
        got = captured["chunks"]
        self.assertEqual(len(got), 2 + 2 + 2)  # 去重后每文档至多 2 段
        # 原 rank 7（C 文档第 2 段，答案段）必须仍在判定上下文中——去重不能把答案段挤掉
        self.assertTrue(any(c is answer_chunk for c in got))
        self.assertTrue(j.sufficient)

    def test_insufficient_still_insufficient(self):
        # 回归边界：没有可答片段时去重不会凭空变可答
        retrieval = RetrievalResult(outcome=None, chunks=[_c("A.pdf", "无关内容")])
        original = service._judge_by_llm
        service._judge_by_llm = lambda q, ch, llm=None, extra_instruction="": (False, "")
        try:
            result = service.judge_evidence("完全无关的问题", retrieval)
        finally:
            service._judge_by_llm = original
        self.assertIs(result.status, NodeStatus.OK)
        self.assertFalse(result.value.sufficient)


class TestFullFallbackOnFocusedFailure(unittest.TestCase):
    """聚焦子集失败时回退全量：防"答案段被去重挤掉"的回归（CS-15/17 实证）。"""

    def test_fallback_to_full_chunks(self):
        docs = [("A.pdf", 5), ("B.pdf", 3), ("C.pdf", 3)]
        chunks = []
        for doc, n in docs:
            for i in range(n):
                chunks.append(_c(doc, f"{doc}段{i}"))
        answer_chunk = _c("C.pdf", "胃灼热和胃痛：处理建议正文")
        chunks[5 + 3 + 2] = answer_chunk  # C 文档第 3 段（去重 cap=2 会挤掉）
        retrieval = RetrievalResult(outcome=None, chunks=chunks)
        calls = []

        def fake_judge(question, ch, llm=None, extra_instruction=""):
            calls.append(len(ch))
            # 聚焦子集（不含答案段）→ 不可答；全量（含答案段）→ 可答
            if any(c is answer_chunk for c in ch):
                return True, "胃灼热和胃痛"
            return False, ""

        original = service._judge_by_llm
        service._judge_by_llm = fake_judge
        try:
            result = service.judge_evidence("孕期常见不适有哪些一般处理原则？", retrieval)
        finally:
            service._judge_by_llm = original
        self.assertIs(result.status, NodeStatus.OK)
        self.assertEqual(len(calls), 2)  # 聚焦一次 + 全量回退一次
        self.assertLess(calls[0], calls[1])
        self.assertTrue(result.value.sufficient)


class TestJudgeExecutionFailures(unittest.TestCase):
    def test_timeout_is_not_business_insufficient(self):
        class _Timeout:
            def invoke(self, messages):
                raise TimeoutError("judge timeout")

        retrieval = RetrievalResult(outcome=None, chunks=[_c("A.pdf", "依据")])
        result = service.judge_evidence("问题", retrieval, llm=_Timeout())
        self.assertIs(result.status, NodeStatus.TIMEOUT)
        self.assertFalse(result.value.sufficient)
        self.assertEqual(result.detail, "TimeoutError")

    def test_invalid_json_is_invalid_output(self):
        class _Message:
            content = "not-json"

        class _Invalid:
            def invoke(self, messages):
                return _Message()

        retrieval = RetrievalResult(outcome=None, chunks=[_c("A.pdf", "依据")])
        result = service.judge_evidence("问题", retrieval, llm=_Invalid())
        self.assertIs(result.status, NodeStatus.INVALID_OUTPUT)
        self.assertFalse(result.value.sufficient)

    def test_unverifiable_quote_remains_conservative_business_result(self):
        retrieval = RetrievalResult(outcome=None, chunks=[_c("A.pdf", "真实依据")])
        original = service._judge_by_llm
        service._judge_by_llm = (
            lambda q, ch, llm=None, extra_instruction="": (True, "编造引文")
        )
        try:
            result = service.judge_evidence("问题", retrieval)
        finally:
            service._judge_by_llm = original
        self.assertIs(result.status, NodeStatus.OK)
        self.assertFalse(result.value.sufficient)
        self.assertEqual(result.value.reason, "stage2_quote_not_found")


if __name__ == "__main__":
    unittest.main()
