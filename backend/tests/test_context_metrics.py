"""context_metrics 五指标的确定性单测（S2，合成 records 锁公式与 N/A 规则）。"""

import unittest

from acceptance.context_metrics import (
    backfill_correct_rate,
    completion_correct_rate,
    constraint_retention_rate,
    degrade_correct_rate,
    interference_rate,
    sentinel_counts,
    summarize,
)


def rec(id="T", **kw):
    base = dict(id=id, mode="lock", flags=[], blocked=False, outcome="completed",
                sources=["某文档 - 第1页"], top1="某文档", answer="回答",
                resolved_question=None, control_top1=None)
    base.update(kw)
    return base


class TestCompletionCorrectRate(unittest.TestCase):
    def test_multi_turn_match(self):
        rs = [rec("A", flags=["multi_turn"], outcome="completed",
                  top1="二甲双胍", control_outcome="completed", control_top1="二甲双胍"),
              rec("B", flags=["multi_turn"], outcome="completed",
                  top1="二甲双胍", control_outcome="completed", control_top1="二甲双胍")]
        m = completion_correct_rate(rs)
        self.assertEqual(m["value"], 1.0)
        self.assertEqual(m["n_denom"], 2)

    def test_mismatch_top1(self):
        rs = [rec("A", flags=["multi_turn"], outcome="completed",
                  top1="阿莫西林", control_outcome="completed", control_top1="二甲双胍")]
        self.assertEqual(completion_correct_rate(rs)["value"], 0.0)

    def test_control_blocked_is_na(self):
        rs = [rec("A", flags=["multi_turn"], outcome="completed",
                  top1="X", control_outcome="completed", control_top1="X",
                  control_blocked=True)]
        self.assertIsNone(completion_correct_rate(rs))

    def test_single_turn_not_in_denom(self):
        rs = [rec("A", flags=["answerable"])]
        self.assertIsNone(completion_correct_rate(rs))


class TestConstraintRetention(unittest.TestCase):
    def test_words_kept(self):
        rs = [rec(flags=["constraint"], constraint={"words": ["没有"]},
                  resolved_question="盐酸二甲双胍片没有胃肠道反应吗？")]
        self.assertEqual(constraint_retention_rate(rs)["value"], 1.0)

    def test_word_dropped(self):
        rs = [rec(flags=["constraint"], constraint={"words": ["没有"]},
                  resolved_question="盐酸二甲双胍片有胃肠道反应吗？")]
        self.assertEqual(constraint_retention_rate(rs)["value"], 0.0)

    def test_watch_excluded(self):
        rs = [rec(mode="watch", flags=["constraint"], constraint={"words": ["最近一周"]},
                  resolved_question="失眠加重？")]
        self.assertIsNone(constraint_retention_rate(rs))


class TestDegradeCorrect(unittest.TestCase):
    def test_unclassified_with_clue(self):
        rs = [rec(flags=["degrade"], outcome="unclassified", sources=[],
                  expected_outcome_any=["unclassified", "context_missing"],
                  answer="抱歉，我暂时无法判断……请补充说明您想了解的内容")]
        self.assertEqual(degrade_correct_rate(rs)["value"], 1.0)

    def test_wrong_outcome(self):
        rs = [rec(flags=["degrade"], outcome="completed", sources=["s"],
                  expected_outcome_any=["unclassified", "context_missing"])]
        self.assertEqual(degrade_correct_rate(rs)["value"], 0.0)

    def test_no_clue_text(self):
        rs = [rec(flags=["degrade"], outcome="unclassified", sources=[],
                  expected_outcome_any=["unclassified"], answer="别的什么话")]
        self.assertEqual(degrade_correct_rate(rs)["value"], 0.0)


class TestBackfillCorrect(unittest.TestCase):
    def test_value_and_note(self):
        rs = [rec(flags=["backfill"], expect_answer_all=["26.8", "此前在对话中提供的信息"],
                  answer="BMI=26.8（本次计算参数取自你此前在对话中提供的信息）")]
        self.assertEqual(backfill_correct_rate(rs)["value"], 1.0)

    def test_missing_note(self):
        rs = [rec(flags=["backfill"], expect_answer_all=["26.8", "此前在对话中提供的信息"],
                  answer="BMI=26.8")]
        self.assertEqual(backfill_correct_rate(rs)["value"], 0.0)


class TestInterferenceAndSentinel(unittest.TestCase):
    def test_interference_none_when_no_cases(self):
        self.assertIsNone(interference_rate([rec()]))

    def test_interference_hit_domain(self):
        rs = [rec(flags=["interference"], interference_domain="高血压", top1="高血压_默沙东")]
        self.assertEqual(interference_rate(rs)["value"], 1.0)

    def test_sentinel_counted_separately(self):
        rs = [rec(flags=["answerable", "sentinel"], outcome="insufficient_evidence"),
              rec(flags=["answerable", "sentinel"], outcome="completed"),
              rec(mode="watch", flags=["sentinel"], outcome="insufficient_evidence")]
        c = sentinel_counts(rs)
        self.assertEqual(c, dict(n_sentinel=3, n_sentinel_refused=2))


class TestBlockedNa(unittest.TestCase):
    def test_blocked_excluded_everywhere(self):
        rs = [rec(flags=["degrade"], blocked=True, outcome=None),
              rec(flags=["degrade"], outcome="unclassified", sources=[],
                  expected_outcome_any=["unclassified"], answer="请补充说明")]
        self.assertEqual(degrade_correct_rate(rs)["value"], 1.0)
        self.assertEqual(degrade_correct_rate(rs)["n_denom"], 1)

    def test_all_blocked_returns_none(self):
        rs = [rec(flags=["degrade"], blocked=True)]
        self.assertIsNone(degrade_correct_rate(rs))
        self.assertIsNone(summarize(rs)["completion_correct_rate"])


if __name__ == "__main__":
    unittest.main()
