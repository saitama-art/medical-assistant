"""T-16c 多轮上下文电池：隐性回指触发与约束保留（S1，测试先行）。

基线实测（2026-10-24）：
- D1 缺口：「用量呢？」has_reference=False、has_followup_lead=False → 不触发 N02
  → 有历史时残缺问句直送分类器 → unclassified（CS-17b 端到端实锤）。
- D1b 既有误触发（本测试新发现）：自包含句「那要么明天再说。」「那种保健品…」
  命中 FOLLOWUP_LEAD 分支 → 历史非空时白付一次改写调用、失败走重述。
- 分层事实：REFERENCE_WORDS 不含「那种」，句首「那种」由 followup 层「种」分支兜底。
- 约束保留（D2）：改写产物无任何程序校验（stage-1 用例如实钉住）。
"""

import json as _json
import unittest

from app.medical_agent.router import has_followup_lead, has_reference, classify
from app.medical_agent.state import NodeStatus, QuestionCategory, ResolutionStatus
from tests.test_medical_agent_step4 import FakeLLM


class TestFollowupTrigger(unittest.TestCase):
    """隐性回指触发器（has_followup_lead）正反例。

    前置语义（orchestrator N02 既有）：历史非空才调用本函数，单轮误触发防护
    由 N02 前置条件兜住，本测试只锁文本级判定本身。
    """

    # ---- 既有锚点与合法追问（修复不得回退，必须保持 True）----
    def test_known_trigger_cs15(self):
        self.assertTrue(has_followup_lead("那什么情况要去医院？"))

    def test_known_trigger_na_hai_neng(self):
        self.assertTrue(has_followup_lead("那还能继续吃吗？"))

    def test_legit_followup_na_hai_yao(self):
        # 「还要复查吗」依赖上文（复查什么），合法触发
        self.assertTrue(has_followup_lead("那还要复查吗？"))

    # ---- D1：零代词省略追问（基线 False=缺口；S1 修复后翻转 True）----
    def test_zero_anaphora_yongliang(self):
        self.assertTrue(has_followup_lead("用量呢？"))

    def test_zero_anaphora_jidian(self):
        self.assertTrue(has_followup_lead("几点吃？"))

    def test_zero_anaphora_haobuhao(self):
        self.assertTrue(has_followup_lead("空腹吃好不好？"))

    # ---- 2026-10-28 实测缺口补：「什么/办法」结尾的省略追问 ----
    def test_zero_anaphora_xuyaozhuyi(self):
        self.assertTrue(has_followup_lead("需要注意什么？"))

    def test_zero_anaphora_chishenme(self):
        self.assertTrue(has_followup_lead("吃什么？"))

    def test_zero_anaphora_banfa(self):
        self.assertTrue(has_followup_lead("有什么办法？"))

    def test_selfcontained_weishenme_still_triggered(self):
        # 「为什么？」单独出现时依赖上文（解释什么），合法触发
        self.assertTrue(has_followup_lead("为什么？"))

    # ---- D1b：自包含句误触发（基线 True=缺口；S1 修复后翻转 False）----
    def test_selfcontained_na_yao(self):
        self.assertFalse(has_followup_lead("那要么明天再说。"))

    def test_selfcontained_na_zhong(self):
        self.assertFalse(has_followup_lead("那种保健品不用吃也行吧？"))

    # ---- 负例（修复前后都必须 False）----
    def test_greeting_not_triggered(self):
        self.assertFalse(has_followup_lead("你好"))

    def test_empty_and_none(self):
        self.assertFalse(has_followup_lead(""))
        self.assertFalse(has_followup_lead(None))  # 类型防御

    # ---- 分层事实备注（非本阶段修复对象）----
    def test_layering_na_zhong_not_in_reference_words(self):
        # 「那种」不在 REFERENCE_WORDS：句首「那种」由 followup 层兜底，
        # 句中「那种」两层都不触发——记录分层事实，扩面时知悉边界。
        self.assertFalse(has_reference("那种保健品不用吃也行吧？"))


class TestConstraintRetentionStage1(unittest.TestCase):
    """D2 约束保留——S1 先钉 FakeLLM 基线（改写产物未经程序校验的现状）。

    stage-2 修复点在 N02 的程序校验（丢「没有」→ 降级 missing，不猜），
    届时 test_negation_dropped_* 的期望改为 assertIsNone。
    """

    def _run_resolve(self, resolved: str, raw: str = "它没有胃肠道反应吗？"):
        from app.medical_agent import orchestrator
        from app.medical_agent.state import AgentState

        # FakeLLM 的 resolve 分发要求 JSON 文本（与真实 LLM 契约一致）
        llm = FakeLLM(resolve=_json.dumps({"question": resolved}, ensure_ascii=False))
        original = orchestrator.get_llm
        orchestrator.get_llm = lambda timeout=35: llm
        try:
            state = AgentState(request_id="t16c", session_id="s", raw_question=raw)
            # 与 N02 真实流程一致：进入改写前 resolved_question=原始问题
            state.resolved_question = raw
            state.history = [("user", "盐酸二甲双胍片有什么常见不良反应？")]
            out = orchestrator._resolve_reference(state, 6000)
        finally:
            orchestrator.get_llm = original
        return out

    def test_resolve_normal_returns_text(self):
        # 无否定词的正常改写：raw 用普通指代追问，产物应原样采用。
        out = self._run_resolve("盐酸二甲双胍片的用法用量是什么？",
                                raw="它的用法用量呢？")
        self.assertIs(out.status, NodeStatus.OK)
        self.assertEqual(out.value.question, "盐酸二甲双胍片的用法用量是什么？")

    def test_negation_dropped_rejected(self):
        # D2 修复：改写产物丢了「没有」（语义反转）→ 拒绝采用，返回 None，
        # 调用方（N02）据此降级 missing 走"请重述"，不猜。
        out = self._run_resolve("盐酸二甲双胍片有胃肠道反应吗？")
        self.assertIs(out.status, NodeStatus.INVALID_OUTPUT)

    def test_negation_kept_accepted(self):
        # 否定词原样保留的改写产物正常采用。
        out = self._run_resolve("盐酸二甲双胍片没有胃肠道反应吗？")
        self.assertIs(out.status, NodeStatus.OK)
        self.assertIs(out.value.status, ResolutionStatus.RESOLVED)
        self.assertEqual(out.value.question, "盐酸二甲双胍片没有胃肠道反应吗？")


class TestClassifyUnchanged(unittest.TestCase):
    """触发器扩面不得波及分类器：寒暄短路与失败语义原样。"""

    def test_greeting_shortcut(self):
        result = classify("你好", llm=None)
        self.assertIs(result.status, NodeStatus.OK)
        self.assertIs(result.value.category, QuestionCategory.CHITCHAT)
        self.assertEqual(result.value.confidence, 1.0)


class TestResolvedQuestionField(unittest.TestCase):
    """S1-D4：/agent/chat 响应的 resolved_question 调试字段（评估方案条件 7）。

    - 恒存在（多轮改写产物 / 单轮原样 / None 透传）；
    - 仅断言契约（字段存在与透传），语义（什么值该出现）由 N02 层测试覆盖；
    - 不触真实 LLM / 数据库：编排器打桩，conversations 打桩，身份走依赖覆盖。
    """

    def _client(self, state):
        from fastapi.testclient import TestClient
        from app.rag.main import app
        from app.auth import deps
        from app import conversations
        from app.medical_agent import orchestrator

        app.dependency_overrides[deps.get_current_user] = lambda: {"id": 1}
        self._orig = (orchestrator.run_agent, conversations.claim_or_verify,
                      conversations.append_turn)
        orchestrator.run_agent = lambda sid, q: state
        conversations.claim_or_verify = lambda sid, owner: sid
        conversations.append_turn = lambda *a, **k: None
        self._app = app
        return TestClient(app, base_url="https://testserver")

    def _cleanup(self):
        from app.auth import deps
        from app import conversations
        from app.medical_agent import orchestrator
        orchestrator.run_agent, conversations.claim_or_verify, \
            conversations.append_turn = self._orig
        self._app.dependency_overrides.clear()

    def _state(self, resolved):
        from app.medical_agent.state import TerminationReason
        return type("S", (), dict(
            request_id="t", answer_text="答", sources=[],
            category=None, termination_reason=TerminationReason.COMPLETED,
            risk_notice=None, resolved_question=resolved))()

    def _post(self, client):
        return client.post("/agent/chat",
                           json={"session_id": "s1", "question": "用量呢？"},
                           headers={"X-Requested-With": "XMLHttpRequest"})

    def test_field_passthrough_resolved(self):
        client = self._client(self._state("瑞舒伐他汀的用法用量是什么？"))
        try:
            r = self._post(client)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["resolved_question"],
                             "瑞舒伐他汀的用法用量是什么？")
        finally:
            self._cleanup()

    def test_field_passthrough_none(self):
        # 急症等 N02 之前终止的路径：resolved_question=None 原样透传
        client = self._client(self._state(None))
        try:
            r = self._post(client)
            self.assertEqual(r.status_code, 200)
            self.assertIsNone(r.json()["resolved_question"])
        finally:
            self._cleanup()

    def test_field_always_present(self):
        client = self._client(self._state("它有什么副作用？"))
        try:
            body = self._post(client).json()
            self.assertIn("resolved_question", body)
            self.assertIn("error_code", body)
            self.assertIn("error_node", body)
        finally:
            self._cleanup()

    def test_structured_error_fields_are_safe_machine_values(self):
        from app.medical_agent.state import AgentError, AgentErrorKind

        state = self._state("它有什么副作用？")
        state.error = AgentError("N02", AgentErrorKind.MODEL_TIMEOUT)
        client = self._client(state)
        try:
            body = self._post(client).json()
            self.assertEqual(body["error_code"], "model_timeout")
            self.assertEqual(body["error_node"], "N02")
            self.assertNotIn("resolver timeout", str(body))
        finally:
            self._cleanup()


if __name__ == "__main__":
    unittest.main()
