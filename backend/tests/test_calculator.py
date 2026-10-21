"""健康计算工具（calculator.py）的确定性单测 + 主 Agent 集成测试。

覆盖任务书验收点：
- 只问 BMI / 标准体重 → 计算工具直接返回（0 LLM、0 检索）
- 缺参 → 固定追问文案；异常参数 → 不返回虚假结果
- 普通知识问题/药品问题 → 不分发到计算工具
- 急症 + 计算词 → 安全规则优先
- 组合问法（计算+解释）→ 先计算后知识检索
"""

import unittest

from app.medical_agent import calculator, orchestrator, respond
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


if __name__ == "__main__":
    unittest.main()
