"""主 Agent Step 4（router / session / orchestrator / /agent/chat）的测试。

用标准库 unittest，不引入 pytest 等额外依赖。全部用例不依赖网络：
LLM 用假实现注入，检索 monkeypatch 掉 service.retrieve_with_scores。
在 backend/ 目录下运行：

    python -m unittest discover -s tests -p "test_*.py" -v
"""

from __future__ import annotations

import uuid
import unittest
from unittest.mock import patch

from app.medical_agent import orchestrator, respond, router, safety, session
from app.medical_agent.state import (
    AgentState,
    ContextStatus,
    EmergencyLevel,
    EvidenceJudgement,
    QuestionCategory,
    RetrievedChunk,
    RetrievalOutcome,
    RetrievalResult,
    Route,
    Source,
    SubagentResult,
    SubagentStatus,
    TerminationReason,
    ToolCallRecord,
    ToolStatus,
    TraceRecord,
)
from app.medical_agent.subagents import knowledge
from app.rag import service


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """按 Prompt 角色分发的假 LLM：分类 / 判定 / 生成各返回固定内容。

    用法：FakeLLM(classify={...}, judge={...}, answer="...")，
    或 raise_on_invoke=True 用于断言"该路径根本不应调用 LLM"。
    """

    def __init__(self, *, classify=None, judge=None, answer="生成回答正文。", raise_on_invoke=False):
        self.classify = classify
        self.judge = judge
        self.answer = answer
        self.raise_on_invoke = raise_on_invoke
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.raise_on_invoke:
            raise AssertionError("该路径不应调用 LLM")
        system = ""
        for m in messages:
            role = getattr(m, "type", "")
            if role == "system":
                system = getattr(m, "content", "") or ""
        import json

        if "意图分类器" in system:
            return _FakeMessage(json.dumps(self.classify, ensure_ascii=False))
        if "证据充分性判定器" in system:
            return _FakeMessage(json.dumps(self.judge, ensure_ascii=False))
        return _FakeMessage(self.answer)


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        content=text,
        document=r"F:\project_my\backend\data\国药准字H20184130_盐酸二甲双胍片.pdf",
        page=1,
        distance=0.5,
    )


def _fresh_session() -> str:
    return f"test-{uuid.uuid4().hex[:12]}"


# ==================== router ====================


class TestRouter(unittest.TestCase):
    def test_greeting_short_circuit(self):
        for text in ["你好", "您好！", "在吗", "hello", "早上好"]:
            self.assertTrue(router.is_greeting(text), text)
        self.assertFalse(router.is_greeting("你好，二甲双胍怎么用"))
        self.assertFalse(router.is_greeting("二甲双胍的适应症"))

    def test_route_table_priorities(self):
        none = EmergencyLevel.NONE
        # 1. 强紧急优先于一切
        self.assertIs(
            router.decide_route(QuestionCategory.KNOWLEDGE, EmergencyLevel.STRONG,
                                ContextStatus.OK, "x"), Route.EMERGENCY)
        # 2. 上下文缺失 + 指代 → 要求重述
        self.assertIs(
            router.decide_route(QuestionCategory.KNOWLEDGE, none,
                                ContextStatus.MISSING, "那它的副作用呢"), Route.CONTEXT_MISSING)
        # 2b. 上下文缺失但无指代 → 正常按类别走
        self.assertIs(
            router.decide_route(QuestionCategory.KNOWLEDGE, none,
                                ContextStatus.MISSING, "二甲双胍的适应症"), Route.KNOWLEDGE)
        # 3~6. 类别映射
        self.assertIs(router.decide_route(QuestionCategory.CHITCHAT, none, ContextStatus.OK, "x"), Route.CHITCHAT)
        self.assertIs(router.decide_route(QuestionCategory.UNCLASSIFIABLE, none, ContextStatus.OK, "x"), Route.UNCLASSIFIABLE)
        self.assertIs(router.decide_route(None, none, ContextStatus.OK, "x"), Route.UNCLASSIFIABLE)
        self.assertIs(router.decide_route(QuestionCategory.TRIAGE, none, ContextStatus.OK, "x"), Route.TRIAGE)
        self.assertIs(router.decide_route(QuestionCategory.KNOWLEDGE, none, ContextStatus.OK, "x"), Route.KNOWLEDGE)

    def test_classify_failure_goes_unclassifiable(self):
        class _Boom:
            def invoke(self, messages):
                raise RuntimeError("network down")

        cat, conf = router.classify("whatever", llm=_Boom())
        self.assertIs(cat, QuestionCategory.UNCLASSIFIABLE)
        self.assertEqual(conf, 0.0)


class TestT05IntentFailureModes(unittest.TestCase):
    """T-05（意图识别异常，设计文档第六部分）：分类器异常 / 超时 / 非法枚举 /
    置信不足 / 环节预算不足未启动 → 全部落 unclassifiable，**无一进入检索**，
    回答为固定"无法处理 + 请补充说明"文案。②~⑤为确定性用例（①与②同走
    classify 异常吞噬路径，router 层已有用例）；⑥语义模糊输入需真实 LLM，
    在验收脚本中覆盖。

    期望（设计文档）：六种情形全部落 unclassifiable、无一进入检索、
    模糊输入不被强行归入某一类。
    """

    def _assert_unclassified_no_retrieval(self, state, calls):
        self.assertIs(state.termination_reason, TerminationReason.UNCLASSIFIED)
        self.assertEqual(calls["retrieval"], 0)
        # 固定文案，不含来源段 / 免责声明 / 风险提示（无 weak 信号时）
        self.assertEqual(state.answer_text.strip(), respond.UNCLASSIFIABLE_REPLY.strip())
        # trace 里有 N04 决策记录（T-15 追踪完整性）
        self.assertIn("N04", [t.node for t in state.trace])

    def test_timeout_goes_unclassifiable(self):
        """② 分类器超时（TimeoutError 与连接错误同走异常吞噬路径）。"""

        class _TimeoutLLM:
            def invoke(self, messages):
                raise TimeoutError("classify llm timeout")

        state, calls = _run(_fresh_session(), "二甲双胍怎么用", _TimeoutLLM(),
                            RetrievalResult(outcome=RetrievalOutcome.SUFFICIENT, chunks=[_chunk("x")]))
        self._assert_unclassified_no_retrieval(state, calls)

    def test_illegal_enum_goes_unclassifiable(self):
        """③ 模型输出非法枚举值：不得猜最接近的一类。"""
        llm = FakeLLM(classify={"category": "banana", "confidence": 0.95})
        state, calls = _run(_fresh_session(), "二甲双胍怎么用", llm,
                            RetrievalResult(outcome=RetrievalOutcome.SUFFICIENT, chunks=[_chunk("x")]))
        self._assert_unclassified_no_retrieval(state, calls)

    def test_low_confidence_goes_unclassifiable(self):
        """④ confidence < CATEGORY_CONFIDENCE_MIN(0.6)：即使类别看似明确也归入 unclassifiable。"""
        llm = FakeLLM(classify={"category": "knowledge", "confidence": 0.35})
        state, calls = _run(_fresh_session(), "二甲双胍怎么用", llm,
                            RetrievalResult(outcome=RetrievalOutcome.SUFFICIENT, chunks=[_chunk("x")]))
        self._assert_unclassified_no_retrieval(state, calls)

    def test_non_numeric_confidence_goes_unclassifiable(self):
        """④b confidence 缺失/非数值（如 "high"）：按 0.0 处理，同归 unclassifiable。"""
        llm = FakeLLM(classify={"category": "knowledge", "confidence": "high"})
        state, calls = _run(_fresh_session(), "二甲双胍怎么用", llm,
                            RetrievalResult(outcome=RetrievalOutcome.SUFFICIENT, chunks=[_chunk("x")]))
        self._assert_unclassified_no_retrieval(state, calls)

    def test_classify_skipped_when_node_budget_unusable(self):
        """⑤ 环节预算不足未启动 N04：不调用 LLM（防假动作），直接 unclassifiable。"""
        calls = {"retrieval": 0}

        def fake_retrieve(question, k=service.DEFAULT_TOP_K):
            calls["retrieval"] += 1
            return RetrievalResult(outcome=RetrievalOutcome.SUFFICIENT, chunks=[_chunk("x")])

        real_node_budget = orchestrator._node_budget

        def fake_node_budget(state, node_ms):
            # 只掐掉 N04 分类环节预算，其余环节保持真实分配
            if node_ms == orchestrator.BUDGET_CLASSIFY_MS:
                return None
            return real_node_budget(state, node_ms)

        llm = FakeLLM(raise_on_invoke=True)  # 走到 LLM 即失败
        with patch.object(orchestrator, "_node_budget", fake_node_budget), \
             patch.object(orchestrator, "get_llm", lambda timeout=35: llm), \
             patch.object(knowledge, "get_llm", lambda timeout=35: llm), \
             patch.object(service, "retrieve_with_scores", fake_retrieve):
            state = orchestrator.run_agent(_fresh_session(), "二甲双胍怎么用")

        self.assertEqual(llm.calls, 0)
        self.assertEqual(calls["retrieval"], 0)
        self.assertIs(state.termination_reason, TerminationReason.UNCLASSIFIED)
        self.assertEqual(state.category, QuestionCategory.UNCLASSIFIABLE)
        # N04 留有预算跳过记录（T-15 追踪完整性）
        n04_budget_traces = [t for t in state.trace
                             if t.node == "N04" and t.event == "budget"
                             and t.detail.get("skipped")]
        self.assertEqual(len(n04_budget_traces), 1)


class TestChestPainScenarios(unittest.TestCase):
    """「胸痛」系列场景的路由正确性（回归：router.decide_route 曾有多余的
    category==emergency → emergency 行，覆盖了安全规则对裸症状的 weak 判断）。

    约定（设计文档 2.2）：完整紧急处理只由 emergency.level==strong 触发；
    分类器判 emergency 而规则未判 strong 时，落入兜底 unclassifiable。
    全部确定性，无网络、无 LLM。
    """

    def test_emergency_category_with_weak_level_falls_through(self):
        # 裸症状：规则 weak，分类器判 emergency，不得升级为紧急
        self.assertIs(
            router.decide_route(QuestionCategory.EMERGENCY, EmergencyLevel.WEAK,
                                ContextStatus.OK, "胸痛"), Route.UNCLASSIFIABLE)

    def test_emergency_category_with_none_level_falls_through(self):
        # 即使规则判 none，只要分类器判 emergency，也不得升级
        self.assertIs(
            router.decide_route(QuestionCategory.EMERGENCY, EmergencyLevel.NONE,
                                ContextStatus.OK, "胸痛"), Route.UNCLASSIFIABLE)

    def test_knowledge_with_weak_level_routes_knowledge(self):
        # weak 不改变路由：知识类仍走 knowledge，风险提示由编排器另加
        self.assertIs(
            router.decide_route(QuestionCategory.KNOWLEDGE, EmergencyLevel.WEAK,
                                ContextStatus.OK, "胸痛是什么原因引起的？"), Route.KNOWLEDGE)

    def test_bare_symptom_chest_pain_is_weak(self):
        hit = safety.check_emergency("胸痛")
        self.assertIs(hit.level, EmergencyLevel.WEAK)
        self.assertEqual(hit.rule_id, "bare_symptom")

    def test_explicit_help_seeking_is_strong_emergency(self):
        hit = safety.check_emergency("我现在胸痛、呼吸困难，怎么办？")
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertIs(
            router.decide_route(QuestionCategory.EMERGENCY, hit.level,
                                ContextStatus.OK, "我现在胸痛、呼吸困难，怎么办？"),
            Route.EMERGENCY)

    def test_third_person_help_seeking_is_strong(self):
        hit = safety.check_emergency("我朋友突然胸痛，还呼吸困难")
        self.assertIs(hit.level, EmergencyLevel.STRONG)

    def test_knowledge_query_chest_pain_cause_is_weak(self):
        hit = safety.check_emergency("胸痛是什么原因引起的？")
        self.assertIs(hit.level, EmergencyLevel.WEAK)
        self.assertIs(
            router.decide_route(QuestionCategory.KNOWLEDGE, hit.level,
                                ContextStatus.OK, "胸痛是什么原因引起的？"),
            Route.KNOWLEDGE)


# ==================== session ====================


class TestSession(unittest.TestCase):
    def test_roundtrip_and_turn_count(self):
        sid = _fresh_session()
        got, history, status = session.acquire(sid)
        self.assertTrue(got)
        self.assertEqual(history, [])
        self.assertIs(status, ContextStatus.NOT_NEEDED)
        session.complete(sid, "q1", "a1", got_lock=got)
        got, history, status = session.acquire(sid)
        self.assertEqual(history, [("user", "q1"), ("assistant", "a1")])
        self.assertIs(status, ContextStatus.OK)
        session.complete(sid, "q2", "a2", got_lock=got)

    def test_truncation_marks_missing(self):
        sid = _fresh_session()
        got, _, _ = session.acquire(sid)
        # 写满 10 条（5 轮）后再写一轮 ⇒ 最早一轮被挤出 ⇒ truncated
        for i in range(session.MAX_CONTEXT_TURNS + 1):
            session.complete(sid, f"q{i}", f"a{i}", got_lock=got)
            got, _, status = session.acquire(sid)
        self.assertIs(status, ContextStatus.MISSING)

    def test_degraded_without_lock_keeps_history_intact(self):
        sid = _fresh_session()
        session.complete(sid, "q1", "a1", got_lock=True)
        # got_lock=False（锁超时降级）不写历史、不释放别人的锁
        session.complete(sid, "qX", "aX", got_lock=False)
        got, history, _ = session.acquire(sid)
        self.assertEqual(history, [("user", "q1"), ("assistant", "a1")])
        session.complete(sid, "q2", "a2", got_lock=got)


# ==================== orchestrator ====================


def _run(state_holder, question, fake_llm, retrieval_result, side_effect=None):
    """跑一遍 run_agent，LLM 与检索全部替换（父与子 Agent 两侧的 LLM 出口都要 patch）。"""
    calls = {"retrieval": 0}

    def fake_retrieve(question, k=service.DEFAULT_TOP_K):
        calls["retrieval"] += 1
        if side_effect:
            return side_effect(question, k)
        return retrieval_result

    with patch.object(orchestrator, "get_llm", lambda timeout=35: fake_llm), \
         patch.object(knowledge, "get_llm", lambda timeout=35: fake_llm), \
         patch.object(service, "retrieve_with_scores", fake_retrieve):
        state = orchestrator.run_agent(state_holder, question)
    return state, calls


class TestOrchestratorN00(unittest.TestCase):
    def test_missing_params_rejected_before_orch(self):
        with self.assertRaises(orchestrator.AgentRequestError):
            orchestrator.run_agent("", "你好")
        with self.assertRaises(orchestrator.AgentRequestError):
            orchestrator.run_agent("s1", "   ")


class TestOrchestratorEmergency(unittest.TestCase):
    def test_strong_emergency_terminates_without_llm(self):
        llm = FakeLLM(raise_on_invoke=True)
        state, calls = _run(_fresh_session(), "我突然胸痛，呼吸困难怎么办", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.EMERGENCY_HANDLED)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(calls["retrieval"], 0)
        self.assertEqual(state.tool_calls, [])
        # 文案 = 紧急应答 + 免责声明；不含病因判断/用药建议
        self.assertIn(respond.EMERGENCY_ANSWER, state.answer_text)
        self.assertIn(respond.DISCLAIMER, state.answer_text)

    def test_emergency_across_turns_second_pass(self):
        # 原文不含急症词，N02 补全后才命中 → N03 第二趟拦截
        sid = _fresh_session()
        session.complete(sid, "我胸口疼该怎么办", "……", got_lock=True)

        def fake_retrieve(question, k=service.DEFAULT_TOP_K):
            raise AssertionError("紧急路径不得调用检索")

        class _ResolveLLM(FakeLLM):
            def invoke(self, messages):
                system = "".join(
                    getattr(m, "content", "") or "" for m in messages
                    if getattr(m, "type", "") == "system"
                )
                if "对话上下文整理器" in system:
                    return _FakeMessage('{"question": "胸口疼要挂什么科"}')
                return super().invoke(messages)

        with patch.object(orchestrator, "get_llm", lambda timeout=35: _ResolveLLM(
                classify={"category": "knowledge", "confidence": 0.9},
                answer="无关回答")), \
             patch.object(service, "retrieve_with_scores", fake_retrieve):
            state = orchestrator.run_agent(sid, "它要挂什么科")

        self.assertIs(state.termination_reason, TerminationReason.EMERGENCY_HANDLED)
        self.assertEqual(state.emergency.level, EmergencyLevel.STRONG)

    def test_bare_symptom_not_escalated_to_emergency(self):
        """端到端：裸症状「胸痛」分类为 emergency 也不得升级为完整紧急；WEAK 风险提示保留。"""
        llm = FakeLLM(classify={"category": "emergency", "confidence": 0.85})
        state, calls = _run(_fresh_session(), "胸痛", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.UNCLASSIFIED)
        self.assertIn(respond.RISK_NOTICE, state.answer_text)
        self.assertNotIn(respond.EMERGENCY_ANSWER, state.answer_text)
        self.assertEqual(calls["retrieval"], 0)
        self.assertEqual(llm.calls, 1)  # 仅一次分类，未进入检索/判定/生成


class TestOrchestratorChitchat(unittest.TestCase):
    def test_greeting_zero_llm_invocations(self):
        llm = FakeLLM(raise_on_invoke=True)
        state, calls = _run(_fresh_session(), "你好", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)
        self.assertEqual(state.answer_text, respond.CHITCHAT_REPLY)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(calls["retrieval"], 0)

    def test_unclassifiable_never_retrieves(self):
        llm = FakeLLM(classify={"category": "unclassifiable", "confidence": 0.2})
        state, calls = _run(_fresh_session(), "*&^%$#@", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.UNCLASSIFIED)
        self.assertEqual(calls["retrieval"], 0)


class TestOrchestratorKnowledge(unittest.TestCase):
    def test_completed_path_assembles_sources_without_drive_path(self):
        chunks = [_chunk("盐酸二甲双胍片适用于2型糖尿病患者。")]
        retrieval = RetrievalResult(outcome=None, chunks=chunks, distance_metric="l2")
        llm = FakeLLM(
            classify={"category": "knowledge", "confidence": 0.95},
            judge={"answerable": True, "supporting_quote": "适用于2型糖尿病患者"},
            answer="二甲双胍用于2型糖尿病。",
        )
        state, calls = _run(_fresh_session(), "二甲双胍的适应症是什么", llm, retrieval)

        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)
        self.assertEqual(state.tool_rounds, 1)
        self.assertEqual(len(state.tool_calls), 1)
        self.assertEqual(calls["retrieval"], 1)
        # 来源只含文档名，不含盘符与目录（设计文档 4.1 / T-14⑥）
        self.assertEqual(len(state.sources), 1)
        displayed = orchestrator.display_sources(state)
        self.assertEqual(len(displayed), 1)
        self.assertNotIn("\\", displayed[0]["document"])
        self.assertNotIn(":", displayed[0]["document"])
        self.assertIn("盐酸二甲双胍片.pdf", displayed[0]["document"])
        self.assertIn("第1页", displayed[0]["document"])  # 状态层页码已是 1 基
        # 正文 + 来源 + 免责声明
        self.assertIn("二甲双胍用于2型糖尿病。", state.answer_text)
        self.assertIn("【参考来源】", state.answer_text)
        self.assertIn(respond.DISCLAIMER, state.answer_text)

    def test_insufficient_evidence_retries_once_then_gives_up(self):
        chunks = [_chunk(" unrelated 片段，不含所问内容。")]
        retrieval = RetrievalResult(outcome=None, chunks=chunks, distance_metric="l2")
        llm = FakeLLM(
            classify={"category": "knowledge", "confidence": 0.95},
            judge={"answerable": False, "supporting_quote": ""},
        )
        state, calls = _run(_fresh_session(), "血压的正常范围是多少", llm, retrieval)

        self.assertIs(state.termination_reason, TerminationReason.INSUFFICIENT_EVIDENCE)
        # 回边恰好 1 次：检索执行 2 轮，之后终止，不进入生成
        self.assertEqual(state.tool_rounds, 2)
        self.assertEqual(calls["retrieval"], 2)
        self.assertEqual(len(state.tool_calls), 2)
        self.assertEqual(llm.calls, 3)  # 分类 1 + 判定 2，无生成调用
        self.assertIn(respond.EVIDENCE_INSUFFICIENT, state.answer_text)
        # 证据不足文案不得拼出来源（T-14① 精神：非完成路径不展示来源）
        self.assertNotIn("【参考来源】", state.answer_text)

    def test_empty_result_distinct_from_error(self):
        retrieval = RetrievalResult(outcome=RetrievalOutcome.EMPTY_RESULT)
        llm = FakeLLM(classify={"category": "knowledge", "confidence": 0.9})
        state, _ = _run(_fresh_session(), "随便问一句", llm, retrieval)
        self.assertIs(state.termination_reason, TerminationReason.EMPTY_KNOWLEDGE_BASE)
        self.assertEqual(state.answer_text, respond.EMPTY_RESULT)
        self.assertNotEqual(respond.EMPTY_RESULT, respond.RETRIEVAL_ERROR)

    def test_retrieval_error_terminates_without_generation(self):
        error = RetrievalResult(outcome=RetrievalOutcome.RETRIEVAL_ERROR, error="TimeoutError")
        llm = FakeLLM(
            classify={"category": "knowledge", "confidence": 0.9},
            answer="不应被生成",
        )
        state, calls = _run(_fresh_session(), "二甲双胍怎么吃", llm, None, side_effect=lambda q, k: error)
        self.assertIs(state.termination_reason, TerminationReason.RETRIEVAL_FAILED)
        self.assertEqual(state.answer_text, respond.RETRIEVAL_ERROR)
        self.assertNotIn("不应被生成", state.answer_text)


class TestOrchestratorContext(unittest.TestCase):
    def test_context_missing_with_reference_asks_restatement(self):
        llm = FakeLLM(raise_on_invoke=True)
        sid = _fresh_session()
        # 制造 truncated 状态
        got, _, _ = session.acquire(sid)
        for i in range(session.MAX_CONTEXT_TURNS + 1):
            session.complete(sid, f"q{i}", f"a{i}", got_lock=got)
            got, _, status = session.acquire(sid)
        self.assertIs(status, ContextStatus.MISSING)
        session.complete(sid, "q_end", "a_end", got_lock=got)  # 释放会话锁

        state, calls = _run(sid, "那它的副作用呢", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.CONTEXT_MISSING)
        self.assertEqual(calls["retrieval"], 0)
        # N04 仍会尝试一次分类（失败归入 unclassifiable），但 N05 优先级 2 覆盖为要求重述
        self.assertLessEqual(llm.calls, 1)

    def test_triage_route_returns_guidance_without_tool(self):
        llm = FakeLLM(classify={"category": "triage", "confidence": 0.9})
        state, calls = _run(_fresh_session(), "我这种情况要不要去医院", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)
        self.assertEqual(state.answer_text, respond.TRIAGE_GUIDANCE)
        self.assertEqual(calls["retrieval"], 0)


class TestDelegation(unittest.TestCase):
    """父 Agent ↔ 知识子 Agent 的委派契约（是否委派、传什么、怎么校验、怎么合并）。"""

    def test_parent_validates_status_and_merges_records(self):
        record = ToolCallRecord(
            seq=1, tool="search_medical_kb", args_digest="len=10;sha1=abc",
            status=ToolStatus.OK, latency_ms=5, budget_ms=5000, round=0,
        )
        trace = [TraceRecord(seq=1, node="N07", event="tool_call", detail={"status": "ok"})]
        result = SubagentResult(
            status=SubagentStatus.INSUFFICIENT,
            evidence=EvidenceJudgement(reason="stage3_not_answerable"),
            retrieval=RetrievalResult(),
            tool_rounds=1, tool_calls=[record], trace=trace,
        )
        llm = FakeLLM(classify={"category": "knowledge", "confidence": 0.9})
        with patch.object(orchestrator, "get_llm", lambda timeout=35: llm), \
             patch.object(knowledge, "run") as mock_run:
            mock_run.return_value = result
            state = orchestrator.run_agent(_fresh_session(), "高血压的正常范围")

        # 父校验：status → 终止原因 + 固定文案
        self.assertIs(state.termination_reason, TerminationReason.INSUFFICIENT_EVIDENCE)
        self.assertIn(respond.EVIDENCE_INSUFFICIENT, state.answer_text)
        # 合并：轮数累计、工具记录与轨迹并入父状态（trace 连续性不断）
        self.assertEqual(state.tool_rounds, 1)
        self.assertEqual(len(state.tool_calls), 1)
        self.assertEqual(state.tool_calls[0].seq, 1)
        self.assertIn("N07", [t.node for t in state.trace])

    def test_subagent_receives_only_question_and_deadline(self):
        """上下文隔离：子 Agent 只拿到问题与 deadline，不拿会话历史。"""
        llm = FakeLLM(classify={"category": "knowledge", "confidence": 0.9})
        with patch.object(orchestrator, "get_llm", lambda timeout=35: llm), \
             patch.object(knowledge, "run") as mock_run:
            mock_run.return_value = SubagentResult(
                status=SubagentStatus.OK, answer="答",
                sources=[Source(document="d.pdf", page=1)], tool_rounds=1,
            )
            orchestrator.run_agent(_fresh_session(), "二甲双胍怎么吃")

        mock_run.assert_called_once()
        call = mock_run.call_args
        self.assertEqual(call.args[0], "二甲双胍怎么吃")
        self.assertIsInstance(call.kwargs["deadline"], float)

    def test_launch_budget_blocks_delegation(self):
        """启动预算不足：不调用子 Agent（防"假动作"），直接预算终止。

        注：经完整 run_agent 走不到这里（N04 预算检查会先归 unclassifiable），
        该守卫是防御纵深，故直接构造低剩余预算的状态测 _delegate_knowledge。
        """
        import time as _time

        state = AgentState(
            session_id=_fresh_session(), raw_question="q", resolved_question="q",
        )
        state.budget.deadline = _time.monotonic() + 1.0  # 剩余 ~1000ms < 3000
        with patch.object(knowledge, "run") as mock_run:
            out = orchestrator._delegate_knowledge(state, got_lock=False)

        mock_run.assert_not_called()
        self.assertIs(out.termination_reason, TerminationReason.BUDGET_EXHAUSTED)


class TestTraceIntegrity(unittest.TestCase):
    def test_every_termination_lands_on_trace(self):
        llm = FakeLLM(classify={"category": "chitchat", "confidence": 0.9})
        state, _ = _run(_fresh_session(), "今天天气不错", llm, None)
        self.assertIsNotNone(state.termination_reason)
        nodes = [t.node for t in state.trace]
        self.assertIn("N00", nodes)
        self.assertEqual(state.trace[-1].node, "N12")
        self.assertEqual(state.trace[-1].detail["termination"], "completed")


if __name__ == "__main__":
    unittest.main()
