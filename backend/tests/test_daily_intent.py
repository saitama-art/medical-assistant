"""daily 意图类别与直接回应路径（2026-10-28 意图层优化）。

语义：日常表达（饿了/最近好累/医生让我减肥/今天我说了什么）由模型判为 daily，
走 0 检索、1 次轻量 LLM 的直接回应；不触发模板澄清，不追加医学免责声明。
分类/RAG 边界、安全门顺序不受影响。
"""

import json
import unittest

from app.medical_agent import respond, router, session
from app.medical_agent.state import (
    NodeStatus,
    QuestionCategory,
    Route,
    TerminationReason,
)
from tests.test_medical_agent_step4 import (
    FakeLLM,
    _FakeMessage,
    _fresh_session,
    _run,
)


class TestDailyClassification(unittest.TestCase):
    def test_daily_enum_accepted_by_classifier(self):
        llm = FakeLLM(classify={"category": "daily", "confidence": 0.9})
        result = router.classify("饿了", llm=llm)
        self.assertIs(result.status, NodeStatus.OK)
        self.assertIs(result.value.category, QuestionCategory.DAILY)

    def test_daily_maps_to_daily_route(self):
        from app.medical_agent.state import ContextStatus, EmergencyLevel
        route = router.decide_route(
            QuestionCategory.DAILY, EmergencyLevel.NONE,
            ContextStatus.OK, "饿了",
        )
        self.assertIs(route, Route.DAILY)


class TestDailyRoute(unittest.TestCase):
    def test_daily_gets_direct_answer_no_retrieval(self):
        llm = FakeLLM(
            classify={"category": "daily", "confidence": 0.9},
            answer="先吃点容易消化的东西垫一下，比如粥或面包。",
        )
        state, calls = _run(_fresh_session(), "饿了", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)
        self.assertIs(state.route, Route.DAILY)
        self.assertEqual(calls["retrieval"], 0)          # 不进 RAG
        self.assertIn("容易消化", state.answer_text)
        self.assertNotIn(respond.DISCLAIMER, state.answer_text)  # 非医学结论

    def test_daily_low_confidence_still_answers(self):
        """低置信度不阻断 daily（任务书补充要求 2/3）。"""
        llm = FakeLLM(
            classify={"category": "daily", "confidence": 0.35},
            answer="减肥可以从每天多走 20 分钟开始。",
        )
        state, calls = _run(_fresh_session(), "医生让我减肥", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)
        self.assertNotIn("请补充", state.answer_text)

    def test_daily_history_review_uses_session_history(self):
        """「今天我说了什么」：daily 提示词携带真实历史，由模型据实总结。"""
        sid = _fresh_session()
        session.complete(sid, "我最近睡眠不太好", "答：注意作息。", got_lock=True)
        captured = {}

        class _SpyLLM(FakeLLM):
            def invoke(self, messages):
                for m in messages:
                    role = getattr(m, "type", "")
                    if role == "system" and "日常对话模块" in (
                            getattr(m, "content", "") or ""):
                        captured["user_msg"] = str(messages[-1].content)
                return super().invoke(messages)

        llm = _SpyLLM(
            classify={"category": "daily", "confidence": 0.9},
            resolve=json.dumps({"question": "我们今天聊了什么内容？"},
                               ensure_ascii=False),
            answer="你今天提到最近睡眠不太好。",
        )
        state, _ = _run(sid, "今天我说了什么？", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)
        self.assertIn("睡眠不太好", state.answer_text)
        self.assertIn("睡眠不太好", captured["user_msg"])  # 历史确实进入提示词

    def test_daily_llm_failure_falls_back_not_retrieval(self):
        class _DownLLM:
            def invoke(self, messages):
                raise RuntimeError("network down")

        state, calls = _run(_fresh_session(), "饿了", _DownLLM(), None)
        # 分类失败（模型不可用）→ processing_failed，不回落到检索编造
        self.assertEqual(calls["retrieval"], 0)
        self.assertIs(state.termination_reason, TerminationReason.PROCESSING_FAILED)


class TestFollowupNoOpRetry(unittest.TestCase):
    def test_noop_resolution_repairs_once(self):
        """N02 原样返回视为可修复失败：repair 后补全实体（2026-10-28 F2）。"""
        calls = {"resolve": 0}

        class _NoOpThenGood(FakeLLM):
            def invoke(self, messages):
                system = ""
                for m in messages:
                    if getattr(m, "type", "") == "system":
                        system = getattr(m, "content", "") or ""
                if "对话上下文整理器" in system:
                    calls["resolve"] += 1
                    if calls["resolve"] == 1:
                        import json as _json
                        return _FakeMessage(
                            _json.dumps({"question": "需要注意什么？"},
                                        ensure_ascii=False))
                    import json as _json
                    return _FakeMessage(
                        _json.dumps({"question": "被狗咬了需要注意什么？"},
                                    ensure_ascii=False))
                return super().invoke(messages)

        from app.rag import service
        ok = service.RetrievalResult(
            outcome=None, chunks=[service.RetrievedChunk(
                content="依据片段", document="d.pdf", page=1, distance=0.3)])
        llm = _NoOpThenGood(
            classify={"category": "knowledge", "confidence": 0.9},
            judge={"answerable": True, "supporting_quote": "依据片段"},
        )
        sid = _fresh_session()
        session.complete(sid, "被狗咬了", "答：尽快处理伤口并就医。", got_lock=True)
        state, _ = _run(sid, "需要注意什么？", llm, ok)
        self.assertEqual(calls["resolve"], 2)            # no-op → 一次修复
        self.assertIn("被狗咬了", state.resolved_question)
        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)


if __name__ == "__main__":
    unittest.main()
