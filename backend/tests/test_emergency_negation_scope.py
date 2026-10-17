"""紧急规则的**否定范围**与**偏侧症状词覆盖**回归测试。

背景（2026-10-16 发现的漏拦）：

    check_emergency("我的一侧手脚无力，口角歪斜") → level=none, rule=all_negated
    decide_route(EMERGENCY, NONE, OK, ...)        → unclassifiable

「我的一侧手脚无力，口角歪斜」是典型的急性卒中征象，却整条漏拦。根因是两个缺陷叠加：

    ① 否定窗口跨过分句标点。`_split_by_negation` 取急症词起点前 3 个字符当窗口，
       对「口角歪斜」而言窗口是 `无力，`，其中的「无」属于**上一个分句**的「无力」，
       并不是对「口角歪斜」的否定。结果唯一命中的词被误判为否定，active 为空。
    ② 词表条目 `一侧无力` 不是 `一侧手脚无力` 的子串（部位词插在中间），
       所以该组只命中 `口角歪斜` 一个词，正好被 ① 吃掉。

修复取最小组合：① 让否定窗口在分句标点处截断（+ 同一词多处出现时只要有一处未被
否定就算有效）；② 在既有卒中征象组（第 6 组）内补齐「偏侧 + 部位 + 无力」的表述。
两项都**不改变任何档位**：6 级优先级与 `BARE_SYMPTOM_LEVEL` 逐字未动，C-4 的评审
对象（裸症状词档位）不受影响。

全部用例确定性，无网络、无 LLM、无 IO。在 backend/ 目录下运行：

    python -m unittest tests.test_emergency_negation_scope -v
"""

from __future__ import annotations

import unittest

from app.medical_agent import router, safety
from app.medical_agent.state import ContextStatus, EmergencyLevel, QuestionCategory, Route


def _route(text: str, category: QuestionCategory = QuestionCategory.EMERGENCY) -> Route:
    """按真实调用方式取路由：先跑规则，再把规则等级交给 decide_route。"""
    return router.decide_route(category, safety.check_emergency(text).level,
                               ContextStatus.OK, text)


class TestNegationScope(unittest.TestCase):
    """否定只在同一分句内生效（缺陷 ①）。"""

    def test_reported_case_enters_emergency_path(self):
        """复现案例：「我的一侧手脚无力，口角歪斜」必须进入紧急路径。"""
        text = "我的一侧手脚无力，口角歪斜"
        hit = safety.check_emergency(text)
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertEqual(hit.rule_id, "help_subject")
        self.assertIs(_route(text), Route.EMERGENCY)

    def test_laterality_phrase_without_context_also_hits_a_term(self):
        """去掉「我」和第二个症状后，仍须命中急症词（缺陷 ② 的直接断言）。

        修改前 `一侧无力` 不是子串 → rule_id 落到 `no_acute_term`。
        """
        hit = safety.check_emergency("一侧手脚无力")
        self.assertIn("一侧手脚无力", hit.terms)
        self.assertIs(hit.level, EmergencyLevel.WEAK)

    def test_punctuation_does_not_carry_negation_forward(self):
        """前半句的否定不得流到后半句：阳性症状必须被判为有效。"""
        text = "没有口角歪斜，但突然一侧手脚无力"
        hit = safety.check_emergency(text)
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertIn("一侧手脚无力", hit.terms)
        self.assertTrue(hit.hit_negated)          # 「口角歪斜」确实被否定
        self.assertNotIn("口角歪斜", hit.terms)    # 但它不该是触发词
        self.assertIs(_route(text), Route.EMERGENCY)

    def test_negation_after_punctuation_does_not_leak_backwards(self):
        """同为标点分隔，方向反过来也必须互不影响。"""
        text = "突然一侧手脚无力，没有口角歪斜"
        hit = safety.check_emergency(text)
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertIn("一侧手脚无力", hit.terms)
        self.assertNotIn("口角歪斜", hit.terms)

    def test_same_term_negated_once_and_positive_once(self):
        """同一个词前后出现两次：只要有一处未被否定就算有效命中。"""
        text = "没有口角歪斜，但现在突然口角歪斜"
        hit = safety.check_emergency(text)
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertIn("口角歪斜", hit.terms)
        # hit_negated 的语义是"**回报的命中词**里有没有被否定的"：本句中该词已被
        # 判定为阳性，negated 为空，故为 False（回退成只查首次出现则会变 True 且漏拦）。
        self.assertFalse(hit.hit_negated)

    def test_negated_symptom_alone_does_not_trigger(self):
        """「没有口角歪斜」不得仅凭该症状触发紧急规则。"""
        text = "没有口角歪斜"
        hit = safety.check_emergency(text)
        self.assertIs(hit.level, EmergencyLevel.NONE)
        self.assertEqual(hit.rule_id, "all_negated")
        self.assertTrue(hit.hit_negated)
        self.assertIsNot(_route(text), Route.EMERGENCY)

    def test_clinical_denial_does_not_trigger(self):
        """「否认肢体无力」不得触发紧急规则。

        说明：`肢体无力` 刻意**不在**词表内（不收裸的「肢体无力」，避免把双侧慢性
        乏力误拦为急症），所以这里走的是 `no_acute_term`。否定本身由「否认」承担，
        而「否认」不在 NEGATION 里——两重保险都指向"不触发"。
        """
        text = "否认肢体无力"
        hit = safety.check_emergency(text)
        self.assertIs(hit.level, EmergencyLevel.NONE)
        self.assertIsNot(_route(text), Route.EMERGENCY)

    def test_window_still_crosses_within_one_clause(self):
        """截断只发生在标点处：同一分句内的否定照旧生效。"""
        for text in ("我没有胸痛", "无明显胸痛", "没有口角歪斜"):
            with self.subTest(text=text):
                self.assertTrue(safety.check_emergency(text).hit_negated)


class TestLateralityTerms(unittest.TestCase):
    """偏侧症状词覆盖（缺陷 ②）。"""

    def test_sudden_unilateral_limb_weakness(self):
        """「突然出现单侧肢体无力」应进入紧急路径。"""
        text = "突然出现单侧肢体无力"
        hit = safety.check_emergency(text)
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertEqual(hit.rule_id, "acute_context")
        self.assertIs(_route(text), Route.EMERGENCY)

    def test_laterality_variants_are_all_covered(self):
        """同义表述必须同组命中，任一写法都不能漏。"""
        variants = ("一侧无力", "单侧无力", "一侧肢体无力", "单侧肢体无力",
                    "一侧手脚无力", "单侧手脚无力")
        for term in variants:
            with self.subTest(term=term):
                hit = safety.check_emergency(f"我突然{term}")
                self.assertIn(term, hit.terms)
                self.assertIs(hit.level, EmergencyLevel.STRONG)

    def test_new_terms_stay_in_the_existing_group(self):
        """新增条目只能是既有第 6 组的补充，不得新增分组（2.3 表共 8 组）。"""
        self.assertEqual(len(safety.ACUTE_TERMS), 8)
        stroke_group = safety.ACUTE_TERMS[5]
        for term in ("一侧手脚无力", "单侧肢体无力"):
            with self.subTest(term=term):
                self.assertIn(term, stroke_group)

    def test_bilateral_weakness_words_are_not_in_table(self):
        """刻意不收裸的「无力」「肢体无力」：双侧慢性乏力不是急症征象。"""
        self.assertNotIn("无力", safety._iter_terms())
        self.assertNotIn("肢体无力", safety._iter_terms())
        self.assertIs(safety.check_emergency("我最近总觉得四肢无力").level,
                      EmergencyLevel.NONE)


class TestNoRegressionOfExistingRules(unittest.TestCase):
    """原有紧急规则不得回归（否定 / 例外 / 元问题 / 裸症状 / 路由约定）。"""

    def test_rule_ladder_unchanged(self):
        cases = [
            # (文本, 期望等级, 期望 rule_id)
            ("我胸口疼，喘不上气", EmergencyLevel.STRONG, "help_subject"),
            ("我胸痛", EmergencyLevel.STRONG, "help_subject"),
            ("我妈突然胸痛怎么办", EmergencyLevel.STRONG, "help_subject"),
            ("突然胸口疼得厉害", EmergencyLevel.STRONG, "acute_context"),
            ("胸痛挂哪个科", EmergencyLevel.STRONG, "acute_context"),
            ("我没有胸痛", EmergencyLevel.NONE, "all_negated"),
            ("二甲双胍怎么吃", EmergencyLevel.NONE, "no_acute_term"),
            ("二甲双胍说明书是否提到胸痛？", EmergencyLevel.WEAK, "meta_question"),
            ("阿托伐他汀的说明书里提到胸痛吗", EmergencyLevel.WEAK, "meta_question"),
            ("胸痛", EmergencyLevel.WEAK, "bare_symptom"),
            ("我有点胸闷", EmergencyLevel.NONE, "no_acute_term"),
        ]
        for text, level, rule_id in cases:
            with self.subTest(text=text):
                hit = safety.check_emergency(text)
                self.assertIs(hit.level, level)
                self.assertEqual(hit.rule_id, rule_id)

    def test_negation_mixed_with_positive_symptom(self):
        """部分否定不得整句降级（既有断言，窗口改动后必须保持）。"""
        hit = safety.check_emergency("我没有胸痛但现在突然喘不上气")
        self.assertIs(hit.level, EmergencyLevel.STRONG)
        self.assertIn("喘不上气", hit.terms)
        self.assertTrue(hit.hit_negated)

    def test_negation_exceptions_still_exceptions(self):
        """「不排除」「不止」不是否定——窗口截断不得把例外一起截掉。"""
        for text in ("不排除胸痛可能", "不止胸痛，还有别的不舒服"):
            with self.subTest(text=text):
                hit = safety.check_emergency(text)
                self.assertFalse(hit.hit_negated)
                self.assertIsNot(hit.level, EmergencyLevel.NONE)

    def test_all_acute_term_groups_still_strong(self):
        cases = [
            "我现在胸痛", "我喘不上气", "我家人昏迷了", "我爸呕血了怎么办",
            "我剧烈头痛", "我妈一侧无力", "我误服了药", "孩子高热惊厥",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertIs(safety.check_emergency(text).level, EmergencyLevel.STRONG)

    def test_bare_symptom_level_untouched(self):
        """C-4 评审对象（裸症状档位）必须逐字未动。"""
        self.assertIs(safety.BARE_SYMPTOM_LEVEL, EmergencyLevel.WEAK)

    def test_route_convention_2d7c9e6_not_reverted(self):
        """裸症状/否定落 unclassifiable 的约定不得为了"进紧急路径"而回退。"""
        self.assertIs(
            router.decide_route(QuestionCategory.EMERGENCY, EmergencyLevel.WEAK,
                                ContextStatus.OK, "胸痛"),
            Route.UNCLASSIFIABLE)
        self.assertIs(
            router.decide_route(QuestionCategory.EMERGENCY, EmergencyLevel.NONE,
                                ContextStatus.OK, "没有口角歪斜"),
            Route.UNCLASSIFIABLE)

    def test_strong_always_routes_to_emergency(self):
        self.assertIs(
            router.decide_route(QuestionCategory.EMERGENCY, EmergencyLevel.STRONG,
                                ContextStatus.OK, "我胸痛"),
            Route.EMERGENCY)


if __name__ == "__main__":
    unittest.main()
