"""上下文工程 V1 确定性测试：临时 SQLite，无模型、网络或 Chroma。"""

import os
import tempfile
import unittest
from dataclasses import replace

from app import conversations
from app.auth import users
from app.db import connect
from app.medical_agent import calculator, context_engineering as ce
from app.medical_agent.state import AgentState, ContextStatus, TerminationReason


def make_task(task_id="task-1", owner=1, conversation="conv-1", **kwargs):
    values = dict(
        task_id=task_id,
        owner_user_id=owner,
        conversation_id=conversation,
        task_type="bmi",
        status=ce.TaskStatus.WAITING_FOR_INPUT,
        required_inputs=("height_cm", "weight_kg"),
        missing_inputs=("height_cm", "weight_kg"),
        created_at=10.0,
        updated_at=10.0,
    )
    values.update(kwargs)
    return ce.TaskRecord(**values)


def make_fact(fact_id="fact-1", fact_type="height_cm", value=170.0, **kwargs):
    values = dict(
        fact_id=fact_id,
        owner_user_id=1,
        conversation_id="conv-1",
        source_message_id=11,
        fact_type=fact_type,
        value=value,
        unit="cm" if fact_type == "height_cm" else "kg",
        event_time=10.0,
        recorded_at=10.0,
    )
    values.update(kwargs)
    return ce.StructuredFact(**values)


def make_state(question="我的 BMI 是多少"):
    return AgentState(
        request_id="request-1",
        session_id="conv-1",
        raw_question=question,
        resolved_question=question,
        context_owner_user_id=1,
    )


class TestTaskLayer(unittest.TestCase):
    def test_missing_inputs_create_waiting_task(self):
        state = make_state()
        plan = ce.prepare_calculation(state, calculator.detect(state.raw_question))
        self.assertEqual(plan.status, "missing")
        self.assertIs(state.task_record.status, ce.TaskStatus.WAITING_FOR_INPUT)
        self.assertEqual(state.task_record.missing_inputs, ("height_cm", "weight_kg"))

    def test_partial_input_is_retained(self):
        state = make_state("我的 BMI，身高 170")
        ce.prepare_calculation(state, calculator.detect(state.raw_question))
        self.assertEqual(state.task_record.provided_inputs, {"height_cm": 170.0})
        self.assertEqual(state.task_record.missing_inputs, ("weight_kg",))

    def test_unique_waiting_task_resumes_with_semantic_input(self):
        waiting = make_task(provided_inputs={"height_cm": 170.0},
                            missing_inputs=("weight_kg",))
        state = make_state("体重 65 公斤")
        state.context_tasks = [waiting]
        plan = ce.prepare_calculation(state, calculator.detect(state.raw_question))
        self.assertEqual(plan.status, "ready")
        self.assertEqual(plan.args, {"height_cm": 170.0, "weight_kg": 65.0})
        self.assertIs(state.task_record.status, ce.TaskStatus.RUNNING)
        self.assertEqual(state.task_record.attempt, 1)

    def test_bare_numbers_resume_unique_waiting_task(self):
        """系统刚澄清「请提供身高体重」时，裸数字就是该澄清的直接回答。"""
        waiting = make_task(provided_inputs={"height_cm": 170.0},
                            missing_inputs=("weight_kg",))
        state = make_state("65")
        state.context_tasks = [waiting]
        plan = ce.prepare_calculation(state, None)
        self.assertEqual(plan.status, "ready")
        self.assertEqual(plan.args, {"height_cm": 170.0, "weight_kg": 65.0})

    def test_bare_numbers_without_waiting_task_do_nothing(self):
        state = make_state("170 65")
        plan = ce.prepare_calculation(state, None)
        self.assertIsNone(plan)
        self.assertIsNone(state.task_record)

    def test_bare_numbers_with_multiple_tasks_require_disambiguation(self):
        state = make_state("170 65")
        state.context_tasks = [make_task("task-1"), make_task("task-2")]
        plan = ce.prepare_calculation(state, None)
        self.assertIsNone(plan)
        self.assertIsNone(state.task_record)

    def test_labeled_height_bare_weight_resumes(self):
        waiting = make_task(provided_inputs={"weight_kg": 65.0},
                            missing_inputs=("height_cm",))
        state = make_state("身高170")
        state.context_tasks = [waiting]
        plan = ce.prepare_calculation(state, None)
        self.assertEqual(plan.status, "ready")
        self.assertEqual(plan.args["height_cm"], 170.0)

    def test_unrelated_request_with_measurement_does_not_resume(self):
        state = make_state("我身高 170，二甲双胍怎么吃？")
        state.context_tasks = [make_task()]
        self.assertIsNone(ce.prepare_calculation(state, None))

    def test_multiple_candidates_require_disambiguation(self):
        state = make_state("体重 65 公斤")
        state.context_tasks = [make_task("task-1"), make_task("task-2")]
        plan = ce.prepare_calculation(state, None)
        self.assertEqual(plan.status, "ambiguous")
        self.assertIsNone(state.task_record)

    def test_conflicting_task_input_is_not_overwritten(self):
        waiting = make_task(provided_inputs={"weight_kg": 60.0},
                            missing_inputs=("height_cm",))
        state = make_state("体重 65 公斤")
        state.context_tasks = [waiting]
        plan = ce.prepare_calculation(state, None)
        self.assertEqual(plan.status, "conflict")
        self.assertEqual(waiting.provided_inputs["weight_kg"], 60.0)

    def test_ready_requires_all_inputs_and_running_is_not_completed(self):
        task = make_task().with_inputs({"height_cm": 170, "weight_kg": 65}, now=20)
        self.assertIs(task.status, ce.TaskStatus.READY)
        running = task.mark_running(now=21)
        self.assertIs(running.status, ce.TaskStatus.RUNNING)
        self.assertIsNot(running.status, ce.TaskStatus.COMPLETED)
        with self.assertRaises(ValueError):
            make_task().mark_running(now=22)

    def test_repeated_missing_request_reuses_task(self):
        waiting = make_task()
        state = make_state()
        state.context_tasks = [waiting]
        ce.prepare_calculation(state, calculator.detect(state.raw_question))
        self.assertEqual(state.task_record.task_id, waiting.task_id)

    def test_cancelled_is_distinct_lifecycle_state(self):
        task = replace(make_task(), status=ce.TaskStatus.CANCELLED)
        self.assertEqual(task.status.value, "cancelled")
        self.assertIsNot(task.status, ce.TaskStatus.COMPLETED)

    def test_terminal_task_cannot_be_reopened_by_duplicate_input(self):
        task = replace(make_task(), status=ce.TaskStatus.COMPLETED)
        with self.assertRaises(ValueError):
            task.with_inputs({"weight_kg": 65}, now=30)


class TestMemoryAndProcessing(unittest.TestCase):
    def test_current_question_task_and_facts_enter_package(self):
        task = make_task(provided_inputs={"height_cm": 170.0},
                         missing_inputs=("weight_kg",))
        package = ce.build_context(
            current_question="我的 BMI 是多少",
            tasks=[task], facts=[make_fact()], token_budget=1000,
        )
        self.assertEqual(package.current_question, "我的 BMI 是多少")
        self.assertEqual(package.missing_information, ("weight_kg",))
        self.assertEqual(package.relevant_facts[0].source_message_id, 11)

    def test_unrelated_old_history_is_excluded(self):
        history = [
            {"id": 1, "role": "user", "content": "北京天气怎么样", "turn": 1},
            {"id": 2, "role": "assistant", "content": "无法回答天气", "turn": 1},
            {"id": 3, "role": "user", "content": "二甲双胍有什么副作用", "turn": 2},
            {"id": 4, "role": "assistant", "content": "可能有胃肠道反应", "turn": 2},
        ]
        package = ce.build_context(current_question="它呢", history=history)
        self.assertEqual([m.message_id for m in package.recent_history], [3, 4])

    def test_constraints_survive_without_summary_or_message_rewrite(self):
        text = "最近一周没有服用二甲双胍，剂量 500mg，不要建议停药"
        history = [
            {"id": 1, "role": "user", "content": text, "turn": 1},
            {"id": 2, "role": "assistant", "content": "已记录二甲双胍信息", "turn": 1},
            {"id": 3, "role": "user", "content": "二甲双胍怎么吃", "turn": 2},
            {"id": 4, "role": "assistant", "content": "需要看说明书", "turn": 2},
        ]
        package = ce.build_context(current_question="二甲双胍注意事项", history=history)
        contents = [m.content for m in package.recent_history]
        self.assertIn(text, contents)
        self.assertFalse(hasattr(package, "summary"))

    def test_old_fact_does_not_automatically_fill_new_task(self):
        state = make_state()
        state.context_facts = [make_fact(value=170), make_fact(
            "fact-old-weight", "weight_kg", 65, unit="kg",
        )]
        ce.prepare_calculation(state, calculator.detect(state.raw_question))
        self.assertEqual(state.task_record.provided_inputs, {})
        self.assertTrue(all(fact.freshness == "historical_unconfirmed"
                            for fact in state.context_facts))

    def test_conflicting_facts_are_explicit(self):
        package = ce.build_context(
            current_question="我的 BMI 是多少",
            facts=[make_fact(value=170), make_fact("fact-2", value=175)],
        )
        self.assertEqual(len(package.conflicts), 1)
        self.assertEqual(package.conflicts[0].kind, "height_cm")
        self.assertEqual(len(package.conflicts[0].source_refs), 2)

    def test_evidence_has_external_source_ref(self):
        package = ce.build_context(
            current_question="二甲双胍适应症",
            retrieval_evidence=[ce.EvidenceContext("原文", "药品说明书.pdf", 3)],
        )
        self.assertIn("药品说明书.pdf#page=3", package.source_refs)
        self.assertEqual(package.relevant_facts, ())

    def test_budget_overflow_does_not_drop_mandatory_context(self):
        question = "没有停药，最近一周" * 100
        package = ce.build_context(current_question=question, token_budget=20)
        self.assertIs(package.assembly_status, ce.ContextAssemblyStatus.BUDGET_EXCEEDED)
        self.assertEqual(package.current_question, question)
        self.assertEqual(package.recent_history, ())

    def test_history_budget_omits_whole_turns_explicitly(self):
        history = [
            {"id": 1, "role": "user", "content": "二甲双胍" * 50, "turn": 1},
            {"id": 2, "role": "assistant", "content": "旧回答" * 50, "turn": 1},
            {"id": 3, "role": "user", "content": "最新问题", "turn": 2},
        ]
        package = ce.build_context(current_question="它呢", history=history,
                                   token_budget=60)
        self.assertEqual(package.current_question, "它呢")
        self.assertGreaterEqual(package.omitted_history_turns, 1)
        self.assertLessEqual(package.token_estimate, package.token_budget)

    def test_assembly_failure_is_not_business_success(self):
        package = ce.build_context(current_question="")
        self.assertIs(package.assembly_status, ce.ContextAssemblyStatus.INVALID_INPUT)
        self.assertEqual(ce.format_history(package), "")

    def test_history_degradation_is_preserved(self):
        package = ce.build_context(current_question="它呢",
                                   context_status=ContextStatus.MISSING)
        self.assertIs(package.context_status, ContextStatus.MISSING)


class TestWorkerEvidenceBudget(unittest.TestCase):
    def test_oversized_evidence_blocks_judge_model_call(self):
        import time
        from unittest.mock import patch
        from app.medical_agent import task_spec
        from app.medical_agent.subagents import knowledge
        from app.medical_agent.state import (
            QuestionCategory, RetrievalOutcome, RetrievalResult, RetrievedChunk,
            Route, SubagentStatus, ToolCallRecord, ToolStatus,
        )

        state = AgentState(
            request_id="context-budget", raw_question="医学问题",
            resolved_question="医学问题", category=QuestionCategory.KNOWLEDGE,
            route=Route.KNOWLEDGE,
        )
        state.budget.deadline = time.monotonic() + 30
        spec = task_spec.build_task_spec(state)
        permit = task_spec.authorize_task_spec(spec, state).permit
        retrieval = RetrievalResult(
            outcome=RetrievalOutcome.SUFFICIENT,
            chunks=[RetrievedChunk("必要证据" * 100, "来源.pdf", 1)],
        )
        invocation = knowledge.tools.ToolInvocation(
            ToolCallRecord(1, knowledge.RETRIEVAL_TOOL, "digest", ToolStatus.OK),
            retrieval,
        )
        with patch.object(knowledge.tools, "call_tool", return_value=invocation), \
                patch.object(knowledge, "_judge_with_repair") as judge, \
                patch.object(knowledge._cfg, "EVIDENCE_CONTEXT_TOKEN_BUDGET", 20):
            result = knowledge.run(spec, permit)
        self.assertIs(result.status, SubagentStatus.BUDGET_EXHAUSTED)
        self.assertEqual(result.detail, "evidence_context_budget")
        judge.assert_not_called()


class TestSQLiteContextStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ["APP_DB_PATH"] = os.path.join(self.tmp.name, "context.db")
        self.addCleanup(os.environ.pop, "APP_DB_PATH", None)
        self.owner = users.create_user("context-a", "hash-a")
        self.other = users.create_user("context-b", "hash-b")
        self.conv = conversations.create_conversation(self.owner, conv_id="conv-1")
        self.other_conv = conversations.create_conversation(
            self.other, conv_id="conv-2",
        )

    def append(self, conv=None, owner=None, question="原始问题"):
        return conversations.append_turn(
            conv or self.conv, owner or self.owner, question, "原始回答",
        )

    def test_raw_message_unchanged_and_fact_traceable(self):
        ids = self.append(question="身高 170，不要改写")
        fact = replace(make_fact(), owner_user_id=self.owner,
                       source_message_id=ids["user"])
        ce.persist_turn_context(self.owner, self.conv, ids["user"], None, [fact])
        messages = conversations.list_messages(self.conv, self.owner)
        self.assertEqual(messages[0]["content"], "身高 170，不要改写")
        stored = ce.load_facts(self.owner, self.conv)[0]
        self.assertEqual(stored.source_message_id, ids["user"])
        self.assertEqual(stored.unit, "cm")

    def test_waiting_task_survives_memory_loss_and_repeat_is_idempotent(self):
        ids = self.append()
        task = make_task(owner=self.owner)
        ce.persist_turn_context(self.owner, self.conv, ids["user"], task, [])
        ce.persist_turn_context(self.owner, self.conv, ids["user"], task, [])
        loaded = ce.load_pending_tasks(self.owner, self.conv)
        self.assertEqual(len(loaded), 1)
        self.assertIs(loaded[0].status, ce.TaskStatus.WAITING_FOR_INPUT)

    def test_cross_user_source_and_task_reads_are_rejected(self):
        other_ids = self.append(self.other_conv, self.other)
        task = make_task(owner=self.owner)
        with self.assertRaises(PermissionError):
            ce.persist_turn_context(self.owner, self.conv, other_ids["user"], task, [])
        self.assertEqual(ce.load_pending_tasks(self.other, self.conv), [])
        self.assertEqual(ce.load_facts(self.other, self.conv), [])

    def test_finalize_marks_completed_only_after_successful_turn(self):
        ids = self.append()
        state = make_state()
        state.context_persistent = True
        state.context_owner_user_id = self.owner
        state.task_record = make_task(owner=self.owner).with_inputs(
            {"height_cm": 170, "weight_kg": 65}, now=20,
        ).mark_running(now=21)
        state.termination_reason = TerminationReason.COMPLETED
        self.assertTrue(ce.finalize_turn(state, ids["user"]))
        with connect() as conn:
            status = conn.execute("SELECT status FROM agent_tasks").fetchone()[0]
        self.assertEqual(status, "completed")

    def test_finalize_does_not_complete_clarification_task(self):
        ids = self.append()
        state = make_state()
        state.context_persistent = True
        state.context_owner_user_id = self.owner
        state.task_record = make_task(owner=self.owner)
        state.termination_reason = TerminationReason.CLARIFICATION_REQUIRED
        ce.finalize_turn(state, ids["user"])
        self.assertIs(ce.load_pending_tasks(self.owner, self.conv)[0].status,
                      ce.TaskStatus.WAITING_FOR_INPUT)


class TestAgentContextIntegration(TestSQLiteContextStore):
    def setUp(self):
        super().setUp()
        from app.auth import deps
        from app.medical_agent import session
        self.deps = deps
        self.session = session
        self.session._SESSIONS.clear()
        self.addCleanup(self.session._SESSIONS.clear)
        self.owner_token = self.deps.current_owner.set(self.owner)
        self.addCleanup(self.deps.current_owner.reset, self.owner_token)

    def finish(self, state, question):
        ids = self.append(question=question)
        ce.finalize_turn(state, ids["user"])

    def test_bmi_waiting_task_resumes_after_memory_loss_without_llm(self):
        from app.medical_agent import orchestrator

        first = orchestrator.run_agent(self.conv, "我的 BMI，身高 170")
        self.assertIs(first.termination_reason, TerminationReason.CLARIFICATION_REQUIRED)
        self.assertIs(first.task_record.status, ce.TaskStatus.WAITING_FOR_INPUT)
        self.finish(first, "我的 BMI，身高 170")

        self.session._SESSIONS.clear()  # 模拟进程重启；SQLite 任务与事实仍在
        second = orchestrator.run_agent(self.conv, "体重 65 公斤")
        self.assertIs(second.termination_reason, TerminationReason.COMPLETED)
        self.assertIn("BMI = **22.5**", second.answer_text)
        self.finish(second, "体重 65 公斤")
        self.assertEqual(ce.load_pending_tasks(self.owner, self.conv), [])
        facts = ce.load_facts(self.owner, self.conv)
        self.assertEqual({fact.fact_type for fact in facts}, {"height_cm", "weight_kg"})
        self.assertTrue(all(fact.source_message_id for fact in facts))


if __name__ == "__main__":
    unittest.main()
