"""健康计算工具（calculator.py）的确定性单测 + 主 Agent 集成测试。

覆盖任务书验收点：
- 只问 BMI / 标准体重 → 计算工具直接返回（0 LLM、0 检索）
- 缺参 → 固定追问文案；异常参数 → 不返回虚假结果
- 普通知识问题/药品问题 → 不分发到计算工具
- 急症 + 计算词 → 安全规则优先
- 组合问法（计算+解释）→ 先计算后知识检索
"""

import json
import unittest

from app.medical_agent import calculator, orchestrator, respond, session
from app.medical_agent.state import TerminationReason

from tests.test_medical_agent_step4 import FakeLLM, _fresh_session, _run
from app.rag import service
from app.medical_agent.subagents import knowledge


class TestCalcFunctions(unittest.TestCase):
    def test_bmi_normal(self):
        r = calculator.calc_bmi(170, 65)
        self.assertEqual(r.value, 22.5)
        self.assertEqual(r.extra["label"], "正常范围")

    def test_bmi_china_classes(self):
        self.assertEqual(calculator.calc_bmi(170, 50).extra["label"], "偏瘦")
        self.assertEqual(calculator.calc_bmi(170, 70).extra["label"], "超重")
        self.assertEqual(calculator.calc_bmi(160, 80).extra["label"], "肥胖")

    def test_bmi_invalid_params(self):
        self.assertEqual(calculator.calc_bmi(300, 65).error, "height_out_of_range")
        self.assertEqual(calculator.calc_bmi(170, 500).error, "weight_out_of_range")

    def test_standard_weight(self):
        r = calculator.calc_standard_weight(175)
        self.assertEqual(r.value, 70.0)
        self.assertEqual(r.extra["range"], (63.0, 77.0))

    def test_convert_units(self):
        self.assertEqual(calculator.convert_unit(65, "kg", "斤").value, 130.0)
        self.assertEqual(calculator.convert_unit(170, "cm", "m").value, 1.7)
        r = calculator.convert_unit(65, "m", "斤")
        self.assertEqual(r.error, "cross_category_convert")
        r = calculator.convert_unit(65, "kg", "盎司")
        self.assertTrue(r.error.startswith("unsupported_to_unit"))


class TestCalcDetect(unittest.TestCase):
    def test_bmi_full(self):
        p = calculator.detect("我身高170，体重65公斤，BMI是多少？")
        self.assertEqual((p.status, p.kind), ("ready", "bmi"))
        self.assertEqual(p.args, {"height_cm": 170.0, "weight_kg": 65.0})

    def test_bmi_jin(self):
        p = calculator.detect("身高1米75，体重130斤，BMI多少")
        self.assertEqual(p.args["height_cm"], 175.0)
        self.assertEqual(p.args["weight_kg"], 65.0)

    def test_bmi_missing(self):
        p = calculator.detect("BMI")
        self.assertEqual(p.status, "missing")
        self.assertEqual(p.missing, ("身高", "体重"))

    def test_bmi_missing_weight(self):
        p = calculator.detect("我身高170，BMI是多少")
        self.assertEqual(p.status, "missing")
        self.assertEqual(p.missing, ("体重",))

    def test_standard_weight_detect(self):
        p = calculator.detect("身高175标准体重多少")
        self.assertEqual((p.status, p.kind), ("ready", "standard_weight"))

    def test_convert_detect(self):
        for q in ("65kg等于多少斤", "170厘米等于多少米", "130斤折合多少公斤"):
            p = calculator.detect(q)
            self.assertEqual((p.status, p.kind), ("ready", "convert"), q)

    def test_combined_splits_remaining(self):
        p = calculator.detect("我身高170，体重65公斤，BMI是多少？有什么危害")
        self.assertEqual(p.remaining, "有什么危害")

    def test_bare_numbers_fallback(self):
        # 口语裸数字：第一个数=身高、带「斤」的数=体重（用户实测 badcase）
        p = calculator.detect("帮我计算我的bmi 178 170斤")
        self.assertEqual((p.status, p.kind), ("ready", "bmi"))
        self.assertEqual(p.args, {"height_cm": 178.0, "weight_kg": 85.0})

    def test_bare_numbers_out_of_order(self):
        p = calculator.detect("bmi 170斤 178")
        self.assertEqual(p.args, {"height_cm": 178.0, "weight_kg": 85.0})

    def test_bare_numbers_too_many_not_guessed(self):
        # 数字过多有歧义：不猜，交回追问
        p = calculator.detect("bmi 178 170 25")
        self.assertEqual(p.status, "missing")

    def test_bare_numbers_not_confused_with_bmi_value(self):
        # 「BMI 24」的 24 不在身高区间，不得被当成身高
        p = calculator.detect("我BMI 24正常吗")
        self.assertEqual(p.status, "missing")

    def test_non_calc_returns_none(self):
        for q in ("二甲双胍怎么吃", "高血压注意什么", "我有点头晕", "晚上睡不着怎么办"):
            self.assertIsNone(calculator.detect(q), q)


class TestCalcIntegration(unittest.TestCase):
    def test_bmi_no_llm_no_retrieval(self):
        llm = FakeLLM(raise_on_invoke=True)  # 任何 LLM 调用都抛 → 证明 0 LLM
        state, calls = _run(_fresh_session(), "我身高170，体重65公斤，BMI是多少？", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)
        self.assertEqual(calls["retrieval"], 0)
        self.assertIn("BMI = **22.5**", state.answer_text)
        self.assertIn("正常范围", state.answer_text)
        self.assertIn(respond.DISCLAIMER, state.answer_text)

    def test_standard_weight_no_llm(self):
        llm = FakeLLM(raise_on_invoke=True)
        state, calls = _run(_fresh_session(), "我身高175，标准体重是多少？", llm, None)
        self.assertIn("70.0kg", state.answer_text)
        self.assertEqual(calls["retrieval"], 0)

    def test_convert_no_llm(self):
        llm = FakeLLM(raise_on_invoke=True)
        state, _ = _run(_fresh_session(), "65kg等于多少斤", llm, None)
        self.assertIn("65.0 kg = 130.0 斤", state.answer_text)

    def test_missing_params_ask(self):
        llm = FakeLLM(raise_on_invoke=True)
        state, calls = _run(_fresh_session(), "BMI", llm, None)
        self.assertEqual(state.answer_text, respond.CALC_NEED_HEIGHT_WEIGHT)
        self.assertEqual(calls["retrieval"], 0)

    def test_invalid_params_no_fake_result(self):
        llm = FakeLLM(raise_on_invoke=True)
        state, _ = _run(_fresh_session(), "我身高300，体重65公斤，BMI是多少？", llm, None)
        self.assertEqual(state.answer_text, respond.CALC_NEED_HEIGHT_WEIGHT)
        self.assertNotIn("BMI = **", state.answer_text)

    def test_emergency_overrides_calc(self):
        # 「胸痛要吃什么药BMI怎么算」类混合输入：急症永远优先于计算
        llm = FakeLLM(classify={"category": "knowledge", "confidence": 0.9})
        state, calls = _run(_fresh_session(), "我胸口压榨样疼痛大汗淋漓，顺便算下BMI", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.EMERGENCY_HANDLED)
        self.assertEqual(calls["retrieval"], 0)

    def test_knowledge_not_misrouted_to_calc(self):
        # 知识问题带数字也不进计算（detect 不命中即回原链路）
        llm = FakeLLM(classify={"category": "knowledge", "confidence": 0.9},
                      judge={"answerable": True, "supporting_quote": "依据片段"}, answer="知识回答。")
        ok = service.RetrievalResult(
            outcome=None, chunks=[service.RetrievedChunk(
                content="依据片段", document="d.pdf", page=1, distance=0.3)],
        )
        state, calls = _run(_fresh_session(), "血压140要注意什么", llm, ok)
        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)
        self.assertEqual(calls["retrieval"], 1)
        self.assertIn("知识回答", state.answer_text)

    def test_combined_calc_then_knowledge(self):
        llm = FakeLLM(classify={"category": "knowledge", "confidence": 0.9},
                      judge={"answerable": True, "supporting_quote": "依据片段"}, answer="超重危害的解释。")
        ok = service.RetrievalResult(
            outcome=None, chunks=[service.RetrievedChunk(
                content="依据片段", document="d.pdf", page=1, distance=0.3)],
        )
        state, calls = _run(_fresh_session(),
                            "我身高170，体重80公斤，BMI是多少？有什么危害", llm, ok)
        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)
        self.assertEqual(calls["retrieval"], 1)  # 剩余问题走了知识检索
        self.assertIn("BMI = **27.7**", state.answer_text)
        self.assertIn("超重危害的解释", state.answer_text)


class TestCalcParamFill(unittest.TestCase):
    """隐性回指补全（N04a 参数回填）：「我的BMI」类追问从会话历史提取参数。

    约束：改写只由 LLM 做"补全问法"，数字始终由 calculator 确定性函数产出；
    任何补不出来（无历史/改写失败/仍缺参/丢意图）都降级为原追问文案，不猜。
    """

    def _seed(self, sid, turns):
        """把若干轮问答写入会话历史。"""
        for q, a in turns:
            session.complete(sid, q, a, got_lock=True)

    def test_followup_bmi_from_history(self):
        # 用户实测 badcase：上一轮算过 BMI，追问「我的BMI」应复用历史参数
        sid = _fresh_session()
        self._seed(sid, [("帮我计算我的bmi 178 170斤", "BMI = **26.8** 超重")])
        llm = FakeLLM(answer=json.dumps(
            {"question": "我身高178，体重170斤，BMI是多少？"}, ensure_ascii=False))
        state, calls = _run(sid, "我的BMI是多少", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)
        self.assertIn("BMI = **26.8**", state.answer_text)
        self.assertIn("超重", state.answer_text)
        self.assertIn(respond.CALC_PARAM_FROM_HISTORY, state.answer_text)
        self.assertIn(respond.DISCLAIMER, state.answer_text)
        self.assertEqual(calls["retrieval"], 0)      # 仍是 0 检索
        self.assertEqual(llm.calls, 1)               # 仅 1 次改写，无分类/生成

    def test_followup_standard_weight_from_history(self):
        sid = _fresh_session()
        self._seed(sid, [("我身高175，标准体重是多少？", "70.0kg")])
        llm = FakeLLM(answer=json.dumps(
            {"question": "我身高175，标准体重是多少？"}, ensure_ascii=False))
        state, calls = _run(sid, "我的标准体重呢", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)
        self.assertIn("70.0kg", state.answer_text)
        self.assertIn(respond.CALC_PARAM_FROM_HISTORY, state.answer_text)
        self.assertEqual(calls["retrieval"], 0)

    def test_no_history_still_asks(self):
        # 首轮就问「我的BMI」：没有历史可回填，维持追问且 0 LLM
        llm = FakeLLM(raise_on_invoke=True)
        state, _ = _run(_fresh_session(), "我的BMI是多少", llm, None)
        self.assertEqual(state.answer_text, respond.CALC_NEED_HEIGHT_WEIGHT)

    def test_params_present_no_rewrite(self):
        # 参数齐全时不触发改写：有历史也 0 LLM（不浪费调用）
        sid = _fresh_session()
        self._seed(sid, [("我身高170，体重65公斤，BMI是多少？", "22.5")])
        llm = FakeLLM(raise_on_invoke=True)
        state, calls = _run(sid, "我身高180，体重70公斤，BMI是多少？", llm, None)
        self.assertIn("BMI = **21.6**", state.answer_text)
        self.assertNotIn(respond.CALC_PARAM_FROM_HISTORY, state.answer_text)
        self.assertEqual(calls["retrieval"], 0)

    def test_rewrite_failed_still_asks(self):
        # LLM 改写失败（空问题）→ 降级原追问，不猜参数
        sid = _fresh_session()
        self._seed(sid, [("帮我计算我的bmi 178 170斤", "26.8")])
        llm = FakeLLM(answer=json.dumps({"question": ""}, ensure_ascii=False))
        state, _ = _run(sid, "我的BMI是多少", llm, None)
        self.assertEqual(state.answer_text, respond.CALC_NEED_HEIGHT_WEIGHT)

    def test_rewrite_losing_intent_still_asks(self):
        # 改写后不再是计算意图 → 不采纳，维持追问
        sid = _fresh_session()
        self._seed(sid, [("帮我计算我的bmi 178 170斤", "26.8")])
        llm = FakeLLM(answer=json.dumps(
            {"question": "今天天气怎么样"}, ensure_ascii=False))
        state, _ = _run(sid, "我的BMI是多少", llm, None)
        self.assertEqual(state.answer_text, respond.CALC_NEED_HEIGHT_WEIGHT)

    def test_truncated_history_not_used(self):
        # 历史已截断（最早的对话被丢弃）：上下文不完整，不用于回填健康参数
        sid = _fresh_session()
        self._seed(sid, [(f"q{i}", f"a{i}") for i in range(6)])  # 12 条 > maxlen 10
        llm = FakeLLM(raise_on_invoke=True)  # 任何 LLM 调用都抛 → 证明未改写
        state, _ = _run(sid, "我的BMI是多少", llm, None)
        self.assertEqual(state.answer_text, respond.CALC_NEED_HEIGHT_WEIGHT)

    def test_combined_fill_then_knowledge(self):
        # 回填 + 组合问法：先算 BMI，剩余「有什么危害」交知识子 Agent
        sid = _fresh_session()
        self._seed(sid, [("帮我计算我的bmi 178 170斤", "26.8")])
        llm = FakeLLM(resolve=json.dumps(
            {"question": "我身高178，体重170斤，BMI是多少？有什么危害"},
            ensure_ascii=False),
            classify={"category": "knowledge", "confidence": 0.9},
            judge={"answerable": True, "supporting_quote": "依据片段"},
            answer="超重危害的解释。")
        ok = service.RetrievalResult(
            outcome=None, chunks=[service.RetrievedChunk(
                content="依据片段", document="d.pdf", page=1, distance=0.3)],
        )
        state, calls = _run(sid, "我的BMI？有什么危害", llm, ok)
        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)
        self.assertIn("BMI = **26.8**", state.answer_text)
        self.assertIn(respond.CALC_PARAM_FROM_HISTORY, state.answer_text)
        self.assertIn("超重危害的解释", state.answer_text)
        self.assertEqual(calls["retrieval"], 1)


if __name__ == "__main__":
    unittest.main()