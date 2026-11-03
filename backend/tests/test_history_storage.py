"""历史存储层 P1/P2 修复的边界与兼容测试（2026-10-29，阶段 1）。

P1（历史写入剥离）：SQLite 存完整回答供前端展示；进上下文前（begin_turn）
对 assistant 消息幂等剥离【参考来源】与免责声明——展示路径内容不变，
新旧格式混合会话读取一致。
P2（历史查询顺序）：list_messages 取最新 N 条（DESC LIMIT 再反转升序），
覆盖 499/500/501/600 边界。
"""

import os
import tempfile
import unittest

from app import conversations, db
from app.auth import deps, users
from app.auth.passwords import hash_password
from app.medical_agent import context_engineering, respond
from app.medical_agent.state import AgentState

FULL_ANSWER = (
    "这是回答正文，含医学内容。\n\n"
    "【参考来源】：二甲双胍_说明书.pdf - 第1、2页\n\n"
    + respond.DISCLAIMER
)
PLAIN_ANSWER = "这是只有正文的回答。"


class HistoryStorageBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ["APP_DB_PATH"] = os.path.join(self.tmp.name, "test.db")
        self.addCleanup(os.environ.pop, "APP_DB_PATH", None)
        self.uid = users.create_user("hist-user", hash_password("pw-hist-12345"))
        self.conv = conversations.create_conversation(self.uid)

    def _insert_messages(self, n):
        """批量插入 n 条消息（user/assistant 交替），内容 msg-1..msg-n。"""
        with db.connect() as conn:
            for i in range(1, n + 1):
                conn.execute(
                    "INSERT INTO messages(conversation_id, role, content,"
                    " created_at, turn) VALUES(?,?,?,?,?)",
                    (self.conv, "user" if i % 2 == 1 else "assistant",
                     f"msg-{i}", float(i), (i + 1) // 2),
                )
            conn.commit()


class TestListMessagesLatestN(HistoryStorageBase):
    """P2 边界：>500 返回最新 500 条且时间升序；≤500 全量返回。"""

    def _assert_latest(self, n, expected_count):
        self._insert_messages(n)
        rows = conversations.list_messages(self.conv, self.uid)
        self.assertEqual(len(rows), expected_count)
        ids = [r["id"] for r in rows]
        self.assertEqual(ids, sorted(ids))                # 时间升序
        self.assertEqual(rows[-1]["content"], f"msg-{n}")  # 最新消息必在
        return [r["content"] for r in rows]

    def test_499_below_limit_returns_all(self):
        contents = self._assert_latest(499, 499)
        self.assertEqual(contents[0], "msg-1")

    def test_500_exact_limit_returns_all(self):
        contents = self._assert_latest(500, 500)
        self.assertEqual(contents[0], "msg-1")

    def test_501_over_limit_drops_oldest_one(self):
        contents = self._assert_latest(501, 500)
        self.assertEqual(contents, [f"msg-{i}" for i in range(2, 502)])

    def test_600_over_limit_returns_latest_500_in_order(self):
        contents = self._assert_latest(600, 500)
        self.assertEqual(contents, [f"msg-{i}" for i in range(101, 601)])


class TestContextHistoryStripping(HistoryStorageBase):
    """P1：展示路径内容不变；上下文路径剥离组装件；新旧混合读取一致。"""

    def _load_context_history(self):
        token = deps.current_owner.set(self.uid)
        self.addCleanup(deps.current_owner.reset, token)
        state = AgentState(session_id=self.conv, raw_question="追问")
        context_engineering.begin_turn(state)
        return state.context_history

    def test_display_path_keeps_full_answer(self):
        # 前端历史消息接口（GET /conversations/{id}/messages）同一直读路径：
        # 完整回答（正文+来源+免责声明）必须原样可见
        conversations.append_turn(self.conv, self.uid, "问题一", FULL_ANSWER)
        rows = conversations.list_messages(self.conv, self.uid)
        assistant = [r for r in rows if r["role"] == "assistant"]
        self.assertEqual(len(assistant), 1)
        self.assertIn("回答正文", assistant[0]["content"])
        self.assertIn("【参考来源】", assistant[0]["content"])
        self.assertIn(respond.DISCLAIMER, assistant[0]["content"])

    def test_context_path_strips_assembly_parts(self):
        conversations.append_turn(self.conv, self.uid, "问题一", FULL_ANSWER)
        history = self._load_context_history()
        assistant = [m for m in history if m.role == "assistant"]
        self.assertEqual(len(assistant), 1)
        self.assertIn("回答正文", assistant[0].content)
        self.assertNotIn("【参考来源】", assistant[0].content)
        self.assertNotIn(respond.DISCLAIMER, assistant[0].content)

    def test_mixed_old_and_new_records_read_consistently(self):
        # 旧格式（修复前落库、含组装件）与新格式（本就不含）混合的会话
        conversations.append_turn(self.conv, self.uid, "旧问题", FULL_ANSWER)
        conversations.append_turn(self.conv, self.uid, "新问题", PLAIN_ANSWER)
        history = self._load_context_history()
        assistant = [m for m in history if m.role == "assistant"]
        self.assertEqual(len(assistant), 2)
        for m in assistant:
            self.assertNotIn("【参考来源】", m.content)
            self.assertNotIn(respond.DISCLAIMER, m.content)
        self.assertEqual(assistant[0].content, "这是回答正文，含医学内容。")
        self.assertEqual(assistant[1].content, PLAIN_ANSWER)

    def test_strip_is_idempotent_and_preserves_safety_text(self):
        strip = respond.strip_assembly_for_context
        self.assertEqual(strip(PLAIN_ANSWER), PLAIN_ANSWER)
        self.assertEqual(strip(strip(FULL_ANSWER)), strip(FULL_ANSWER))
        # 风险提示等非固定部件必须保留（S1-D3 保留纪律不回归）
        risk = "注意：症状持续请尽快就医。"
        text = FULL_ANSWER.replace("这是回答正文，含医学内容。", risk)
        self.assertIn(risk, strip(text))


if __name__ == "__main__":
    unittest.main()
