"""主 Agent 的测试。

用标准库 unittest，不引入 pytest 等额外依赖。
在 backend/ 目录下运行：

    python -m unittest discover -s tests -p "test_*.py" -v

第一阶段覆盖 state / respond / safety 三个基础模块。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.medical_agent import respond, safety
from app.medical_agent.safety import EmergencyRuleError, check_emergency
from app.medical_agent.state import (
    AgentState,
    EmergencyHit,
    EmergencyLevel,
    Source,
    TerminationReason,
    ToolCallRecord,
    ToolStatus,
    TraceRecord,
)


class TestEmergencyStrong(unittest.TestCase):
    """必须拦截的情形。"""

    def test_first_person_acute_symptom(self):
        hit = check_emergency("我胸口疼，喘不上气")
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertEqual(hit.rule_id, "help_subject")
        self.assertIn("胸口疼", hit.terms)

    def test_first_person_alone_is_enough(self):
        self.assertIs(check_emergency("我胸痛").level, EmergencyLevel.STRONG)

    def test_third_person_help_seeking(self):
        hit = check_emergency("我妈突然胸痛怎么办")
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertEqual(hit.rule_id, "help_subject")

    def test_third_person_minor(self):
        self.assertIs(check_emergency("孩子高热惊厥").level, EmergencyLevel.STRONG)

    def test_acute_context_without_subject(self):
        hit = check_emergency("突然胸口疼得厉害")
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertEqual(hit.rule_id, "acute_context")

    def test_help_seeking_without_subject(self):
        self.assertIs(check_emergency("胸痛挂哪个科").level, EmergencyLevel.STRONG)

    def test_whitespace_is_ignored(self):
        self.assertIs(check_emergency("我 胸 痛").level, EmergencyLevel.STRONG)

    def test_all_acute_term_groups_trigger(self):
        """8 组急症词每组至少一条必须触发 STRONG。"""
        cases = [
            "我现在胸痛",
            "我喘不上气",
            "我家人昏迷了",
            "我爸呕血了怎么办",
            "我剧烈头痛",
            "我妈一侧无力",
            "我误服了药",
            "孩子高热惊厥",
        ]
        self.assertEqual(len(safety.ACUTE_TERMS), 8)
        for text in cases:
            with self.subTest(text=text):
                self.assertIs(check_emergency(text).level, EmergencyLevel.STRONG)


class TestEmergencyMetaQuestion(unittest.TestCase):
    """元问题语境不得压过真实急症信号。"""

    def test_drug_leaflet_question_is_not_intercepted(self):
        hit = check_emergency("二甲双胍说明书是否提到胸痛？")
        self.assertIs(hit.level, EmergencyLevel.WEAK)
        self.assertEqual(hit.rule_id, "meta_question")

    def test_meta_word_does_not_downgrade_real_symptom(self):
        """既有"这个药"又有第一人称急症时，仍须拦截。"""
        hit = check_emergency("我吃这个药后突然胸痛")
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertEqual(hit.rule_id, "help_subject")

    def test_bare_ta_is_not_a_subject_marker(self):
        """"阿托伐他汀"含"他"，不得被当作第三人称求助。"""
        hit = check_emergency("阿托伐他汀的说明书里提到胸痛吗")
        self.assertIs(hit.level, EmergencyLevel.WEAK)
        self.assertEqual(hit.rule_id, "meta_question")

    def test_bare_symptom_word_is_weak(self):
        hit = check_emergency("胸痛")
        self.assertIs(hit.level, EmergencyLevel.WEAK)
        self.assertEqual(hit.rule_id, "bare_symptom")

    def test_mild_symptom_not_in_table(self):
        self.assertIs(check_emergency("我有点胸闷").level, EmergencyLevel.NONE)


class TestEmergencyNegation(unittest.TestCase):
    """否定语境。"""

    def test_all_negated_returns_none(self):
        hit = check_emergency("我没有胸痛")
        self.assertIs(hit.level, EmergencyLevel.NONE)
        self.assertEqual(hit.rule_id, "all_negated")
        self.assertTrue(hit.hit_negated)

    def test_partial_negation_keeps_other_symptom(self):
        """"没有胸痛"被否定，但"喘不上气"仍然有效——不得整句降级。"""
        hit = check_emergency("我没有胸痛但现在突然喘不上气")
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertIn("喘不上气", hit.terms)
        self.assertTrue(hit.hit_negated)

    def test_negation_exception_buzhipai(self):
        """"不排除"不是否定。"""
        hit = check_emergency("不排除胸痛可能")
        self.assertFalse(hit.hit_negated)
        self.assertIsNot(hit.level, EmergencyLevel.NONE)

    def test_negation_exception_buzhi(self):
        """"不止"不是否定。"""
        hit = check_emergency("不止胸痛，还有别的不舒服")
        self.assertFalse(hit.hit_negated)
        self.assertIsNot(hit.level, EmergencyLevel.NONE)

    def test_no_acute_term(self):
        hit = check_emergency("二甲双胍怎么吃")
        self.assertIs(hit.level, EmergencyLevel.NONE)
        self.assertEqual(hit.rule_id, "no_acute_term")


class TestEmergencyRuleFault(unittest.TestCase):
    """规则自身异常必须抛出，交由调用方保守降级。"""

    def test_empty_table_raises(self):
        with patch.object(safety, "ACUTE_TERMS", ()):
            with self.assertRaises(EmergencyRuleError):
                check_emergency("我胸痛")

    def test_non_string_raises(self):
        with self.assertRaises(EmergencyRuleError):
            check_emergency(123)

    def test_none_is_empty_input(self):
        hit = check_emergency(None)
        self.assertIs(hit.level, EmergencyLevel.NONE)
        self.assertEqual(hit.rule_id, "empty_input")

    def test_blank_string_is_empty_input(self):
        self.assertEqual(check_emergency("   ").rule_id, "empty_input")


class TestRespondTexts(unittest.TestCase):
    """固定文案的硬约束。"""

    def test_emergency_text_has_no_diagnosis_or_medication_advice(self):
        forbidden = ("建议服用", "可以吃", "剂量", "毫克", "mg", "诊断为", "患了", "考虑为")
        for word in forbidden:
            with self.subTest(word=word):
                self.assertNotIn(word, respond.EMERGENCY_ANSWER)

    def test_emergency_text_gives_action_guidance(self):
        self.assertIn("120", respond.EMERGENCY_ANSWER)
        self.assertIn("急诊", respond.EMERGENCY_ANSWER)

    def test_insufficient_text_has_no_conclusion(self):
        forbidden = ("适应症", "禁忌", "用法用量", "可以治疗", "适用于")
        for word in forbidden:
            with self.subTest(word=word):
                self.assertNotIn(word, respond.EVIDENCE_INSUFFICIENT)

    def test_failure_texts_are_distinguishable(self):
        """空结果 / 检索失败 / 生成失败 / 证据不足 必须两两不同。"""
        texts = [
            respond.EMPTY_RESULT,
            respond.RETRIEVAL_ERROR,
            respond.GENERATION_ERROR,
            respond.EVIDENCE_INSUFFICIENT,
        ]
        self.assertEqual(len(texts), len(set(texts)))

    def test_termination_texts_cover_all_reasons_except_completed(self):
        expected = {r for r in TerminationReason if r is not TerminationReason.COMPLETED}
        self.assertEqual(set(respond.TERMINATION_TEXTS), expected)

    def test_no_internal_information_in_texts(self):
        """文案里不得泄漏堆栈、密钥、本机路径。"""
        texts = {
            name: value
            for name, value in vars(respond).items()
            if not name.startswith("_") and isinstance(value, str)
        }
        texts.update({f"TERMINATION_TEXTS[{k}]": v for k, v in respond.TERMINATION_TEXTS.items()})
        self.assertGreater(len(texts), 10)
        forbidden = ("Traceback", "sk-", ".py", "C:\\", "F:\\", 'File "')
        for name, text in texts.items():
            for word in forbidden:
                with self.subTest(constant=name, word=word):
                    self.assertNotIn(word, text)


class TestAgentState(unittest.TestCase):
    """状态对象的基本契约。"""

    def test_defaults_are_independent_per_instance(self):
        """两个请求各自新建状态，列表字段不得共享同一个对象。"""
        a, b = AgentState(), AgentState()
        a.add_trace("N01", "enter")
        a.history.append(("user", "x"))
        a.tool_calls.append(
            ToolCallRecord(seq=1, tool="kb_search", args_digest="d", status=ToolStatus.OK)
        )
        a.sources.append(Source(document="doc"))
        self.assertEqual(b.trace, [])
        self.assertEqual(b.history, [])
        self.assertEqual(b.tool_calls, [])
        self.assertEqual(b.sources, [])

    def test_records_are_pure_data(self):
        record = TraceRecord(seq=1, node="N01", event="enter")
        self.assertEqual(record.detail, {})

    def test_add_trace_increments_seq(self):
        st = AgentState()
        st.add_trace("N01", "decision", {"level": "strong"})
        st.add_trace("N02", "decision", {})
        self.assertEqual([r.seq for r in st.trace], [1, 2])
        self.assertEqual(st.trace[0].node, "N01")

    def test_default_emergency_is_none(self):
        self.assertEqual(AgentState().emergency, EmergencyHit())
        self.assertIs(AgentState().emergency.level, EmergencyLevel.NONE)


if __name__ == "__main__":
    unittest.main()
