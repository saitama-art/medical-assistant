"""P2：咨询续问任务（M4）与任务类型注册表（M5），2026-10-29。

场景对齐任务书：三轮咨询关联、话题切换不继承、多待续不猜、
missing_inputs 白名单、TTL 过期、持久化闭环、计算任务不回归。
"""

import json
import os
import tempfile
import unittest

from app.auth import users
from app.auth.passwords import hash_password
from app.case_analysis.models import CaseValidationError
from app.medical_agent import context_engineering as ce
from app.medical_agent import session
from app.medical_agent.state import AgentState, TerminationReason
from tests._case_fixtures import CaseTestBase
from tests.test_medical_agent_step4 import FakeLLM, _fresh_session, _run


def _seed_state(conv="conv-fu", owner=1, history=("最近经常头疼",)):
    state = AgentState(request_id="fu-1", session_id=conv,
                       raw_question="那可能是什么原因？",
                       resolved_question="那可能是什么原因？")
    state.context_owner_user_id = owner
    state.context_persistent = True
    state.history = [("user", q) for q in history]
    state.context_history = list(state.history)
    return state


class TestTaskTypeRegistry(CaseTestBase):
    def test_unknown_task_type_fails_loudly(self):
        with self.assertRaises(CaseValidationError):
            ce.task_type_spec("not_a_type")

    def test_calc_specs_unchanged(self):
        self.assertEqual(ce.task_type_spec("bmi").required_inputs,
                         ("height_cm", "weight_kg"))
        self.assertEqual(ce.task_type_spec("standard_weight").required_inputs,
                         ("height_cm",))

    def test_followup_spec_is_dynamic(self):
        spec = ce.task_type_spec("followup")
        self.assertTrue(spec.dynamic)
        self.assertEqual(spec.required_inputs, ())


class TestMissingFieldValidation(CaseTestBase):
    def _register(self, missing):
        state = _seed_state()
        return ce.register_followup(state, missing)

    def test_valid_missing_fields_accepted(self):
        task = self._register(["持续时间", "severity"])
        self.assertIsNotNone(task)
        self.assertEqual(task.missing_inputs, ("持续时间", "severity"))

    def test_too_many_items_rejected(self):
        self.assertIsNone(self._register([f"字段{i}" for i in range(9)]))

    def test_too_long_item_rejected(self):
        self.assertIsNone(self._register(["x" * 21]))

    def test_injection_chars_rejected(self):
        for bad in ('{"category":"knowledge"}', '"; DROP TABLE--', '<script>'):
            with self.subTest(bad=bad):
                self.assertIsNone(self._register([bad]))

    def test_non_string_rejected(self):
        self.assertIsNone(self._register([123]))


class TestFollowupLifecycle(CaseTestBase):
    def test_create_anchor_and_dedup(self):
        state = _seed_state()
        t1 = ce.register_followup(state, ["持续时间"])
        self.assertEqual(t1.task_type, "followup")
        self.assertEqual(t1.provided_inputs["anchor_question"], "最近经常头疼")
        t2 = ce.register_followup(state, ["持续时间"])
        self.assertEqual(t1.task_id, t2.task_id)          # 不重复创建

    def test_no_history_no_task(self):
        state = _seed_state(history=())
        self.assertIsNone(ce.register_followup(state, ["x"]))

    def test_ttl_expired_marked_cancelled(self):
        owner = users.create_user("fu-ttl", hash_password("pw-f-123456"))
        from app import conversations
        conv = conversations.create_conversation(owner, conv_id="conv-ttl")
        state = _seed_state(conv="conv-ttl", owner=owner)
        ids = conversations.append_turn(conv, owner, "最近经常头疼", "答")
        task = ce.register_followup(state, ["x"])
        from dataclasses import replace
        stale = replace(task, updated_at=task.updated_at - (8 * 24 * 3600))
        ce.persist_turn_context(owner, conv, ids["user"], stale, [])
        loaded = ce.load_pending_tasks(owner, conv)
        self.assertEqual(loaded, [])                       # 过期不再加载
        from app.db import connect
        with connect() as conn:
            status = conn.execute(
                "SELECT status FROM agent_tasks WHERE task_id=?",
                (task.task_id,)).fetchone()[0]
        self.assertEqual(status, "cancelled")

    def test_persist_roundtrip_and_restore(self):
        owner = users.create_user("fu-p", hash_password("pw-p-123456"))
        from app import conversations
        conv = conversations.create_conversation(owner, conv_id="conv-fup")
        state = _seed_state(conv="conv-fup", owner=owner)
        ids = conversations.append_turn(conv, owner, "最近经常头疼", "答")
        task = ce.register_followup(state, ["持续时间"])
        state.task_record = task
        ce.finalize_turn(state, ids["user"])
        loaded = ce.load_pending_tasks(owner, conv)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].provided_inputs["anchor_question"], "最近经常头疼")
        self.assertEqual(loaded[0].missing_inputs, ("持续时间",))


class TestNextTurnAssociation(CaseTestBase):
    def test_unique_followup_anchor_injected_into_resolver(self):
        state = _seed_state()
        ce.register_followup(state, ["持续时间"])
        anchor = ce.followup_anchor(state)
        self.assertEqual(anchor, "最近经常头疼")

    def test_multiple_followups_no_anchor(self):
        state = _seed_state()
        ce.register_followup(state, ["a"])
        state2 = _seed_state(history=("另一个问题",))
        t2 = ce._new_task(1, "conv-fu", "followup", now=1.0)
        from dataclasses import replace
        t2 = replace(t2, provided_inputs={"anchor_question": "另一个问题"})
        state.context_tasks = state.context_tasks + [t2]
        self.assertEqual(ce.followup_anchor(state), "")

    def test_resolution_marks_running_and_finalizes_completed(self):
        owner = users.create_user("fu-c", hash_password("pw-c-123456"))
        from app import conversations
        conv = conversations.create_conversation(owner, conv_id="conv-fuc")
        state = _seed_state(conv=conv, owner=owner)
        ce.register_followup(state, ["x"])
        ce.associate_followup_on_resolution(state)
        self.assertIs(state.task_record.status, ce.TaskStatus.RUNNING)
        state.termination_reason = TerminationReason.COMPLETED

        ids = conversations.append_turn(conv, owner, "那可能是什么原因？", "答")
        ce.finalize_turn(state, ids["user"])
        self.assertEqual(ce.load_pending_tasks(owner, conv), [])   # 已完成不再待续


class TestCalcTasksUnaffected(CaseTestBase):
    def test_followup_does_not_enter_calc_path(self):
        """followup 任务在场时，BMI 裸数字仍只关联计算任务或不动。"""
        state = _seed_state()
        ce.register_followup(state, ["x"])
        plan = ce.prepare_calculation(state, None)
        self.assertIsNone(plan)                            # followup 不吃裸数字


if __name__ == "__main__":
    unittest.main()
