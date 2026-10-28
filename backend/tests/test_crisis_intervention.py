"""心理危机干预：safety.check_crisis 规则与编排器固定文案（2026-10-28）。

铁律与急症同一套：确定性纯规则、否定同窗口、不检索不生成、
宁可误拦不可漏拦。危机文案由用户逐字审核（respond.CRISIS_ANSWER）。
"""

import unittest

from app.medical_agent import respond, safety
from app.medical_agent.orchestrator import run_agent
from app.medical_agent.state import EmergencyLevel, TerminationReason


class TestCrisisDetection(unittest.TestCase):
    def test_self_harm_ideation_strong(self):
        hit = safety.check_crisis("我有伤害自己的念头，控制不住")
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertEqual(hit.rule_id, "crisis_subject")

    def test_common_ideation_phrases(self):
        for text in ("我不想活了", "我想死", "我想结束生命",
                     "孩子说要自杀怎么办", "朋友有轻生念头，我该怎么办"):
            with self.subTest(text=text):
                self.assertIs(safety.check_crisis(text).level, EmergencyLevel.STRONG)

    def test_negated_not_flagged(self):
        for text in ("我不想伤害自己", "没有自杀的念头", "我从未想过轻生"):
            with self.subTest(text=text):
                hit = safety.check_crisis(text)
                self.assertIs(hit.level, EmergencyLevel.NONE)

    def test_bare_or_statistical_question_not_flagged(self):
        for text in ("自杀率是多少", "抑郁症和自杀的关系是什么"):
            with self.subTest(text=text):
                hit = safety.check_crisis(text)
                self.assertIs(hit.level, EmergencyLevel.NONE)

    def test_emergency_rules_unchanged(self):
        """危机词表不影响既有急症判定。"""
        hit = safety.check_emergency("我突然胸口压榨样疼痛，喘不上气")
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertIs(safety.check_crisis("我突然胸口压榨样疼痛").level,
                      EmergencyLevel.NONE)


class TestCrisisEndToEnd(unittest.TestCase):
    def test_crisis_returns_crisis_text_no_retrieval(self):
        state = run_agent("crisis-s1", "我有伤害自己的念头，控制不住")
        self.assertIs(state.termination_reason, TerminationReason.EMERGENCY_HANDLED)
        self.assertIn("听起来，你现在正承受着很大的压力", state.answer_text)
        self.assertIn("是否正处于可能马上伤害自己的危险中", state.answer_text)
        self.assertEqual(state.sources, [])            # 不检索
        self.assertEqual(state.tool_rounds, 0)         # 不调用工具
        self.assertEqual(state.emergency.rule_id, "crisis_subject")

    def test_crisis_text_is_not_medical_emergency_text(self):
        state = run_agent("crisis-s2", "我不想活了")
        self.assertIn("急救服务", state.answer_text)
        self.assertNotIn("120", state.answer_text)     # 危机文案与 120 急症文案不同

    def test_medical_emergency_still_uses_120_text(self):
        state = run_agent("crisis-s3", "我突然胸口压榨样疼痛，喘不上气")
        self.assertIs(state.termination_reason, TerminationReason.EMERGENCY_HANDLED)
        self.assertIn("120", state.answer_text)
        self.assertNotIn("听起来，你现在正承受着很大的压力", state.answer_text)

    def test_followup_crisis_after_context_resolution(self):
        """N03 路径：先建立上下文，再用指代表达危机也要拦下。"""
        first = run_agent("crisis-s4", "我最近压力很大")
        state = run_agent("crisis-s4", "那我不想活了怎么办")
        self.assertIs(state.termination_reason, TerminationReason.EMERGENCY_HANDLED)
        self.assertIn("听起来，你现在正承受着很大的压力", state.answer_text)

    def test_crisis_answer_has_no_disclaimer_conflict(self):
        """免责声明追加契约：危机路径与急症一致（EMERGENCY_HANDLED 追加）。"""
        state = run_agent("crisis-s5", "我想死")
        self.assertIn(respond.DISCLAIMER.split("，")[0], state.answer_text)


if __name__ == "__main__":
    unittest.main()
