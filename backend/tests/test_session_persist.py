"""session.py C-6 文件持久化的确定性单测（临时文件目录，无网络/模型）。"""

import json
import os
import tempfile
import unittest

from app.medical_agent import session
from app.medical_agent.state import ContextStatus


class TestSessionPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        os.environ[session._PERSIST_ENV] = self._tmp.name
        session._SESSIONS.clear()

    def tearDown(self):
        os.environ.pop(session._PERSIST_ENV, None)
        session._SESSIONS.clear()
        self._tmp.cleanup()

    def test_roundtrip_across_memory_loss(self):
        # 写入 → 清空内存（模拟进程重启）→ 同 session_id 恢复历史
        got, _, status = session.acquire("sess-a")
        self.assertTrue(got)
        self.assertIs(status, ContextStatus.NOT_NEEDED)
        session.complete("sess-a", "问一", "答一", got_lock=True)

        session._SESSIONS.clear()  # 模拟重启

        got, history, status = session.acquire("sess-a")
        self.assertTrue(got)
        self.assertEqual(history, [("user", "问一"), ("assistant", "答一")])
        self.assertIs(status, ContextStatus.OK)
        session.complete("sess-a", "问二", "答二", got_lock=True)

        # 文件内容可读且为两条消息
        path = os.path.join(self._tmp.name, "sess-a.json")
        payload = json.load(open(path, encoding="utf-8"))
        self.assertEqual(len(payload["messages"]), 4)
        self.assertFalse(payload["truncated"])

    def test_persist_disabled_by_default(self):
        os.environ.pop(session._PERSIST_ENV, None)
        got, _, _ = session.acquire("sess-b")
        session.complete("sess-b", "问", "答", got_lock=True)
        self.assertEqual(os.listdir(self._tmp.name), [])

    def test_unsafe_session_id_skipped(self):
        got, _, _ = session.acquire("../evil")
        session.complete("../evil", "问", "答", got_lock=True)
        self.assertEqual(os.listdir(self._tmp.name), [])

    def test_truncated_flag_survives(self):
        got, _, _ = session.acquire("sess-c")
        self.assertTrue(got)
        for i in range(session.MAX_CONTEXT_TURNS + 1):
            session.complete("sess-c", f"问{i}", f"答{i}", got_lock=True)
            got, _, _ = session.acquire("sess-c")
            self.assertTrue(got)
        session._SESSIONS.clear()
        got, _, status = session.acquire("sess-c")
        self.assertIs(status, ContextStatus.MISSING)


class TestHistoryAnswerStripping(unittest.TestCase):
    """S1-D3：写回历史时剥离来源段/免责声明，保留医疗安全信息。"""

    def setUp(self):
        session._SESSIONS.clear()

    def tearDown(self):
        session._SESSIONS.clear()

    def _stored_answer(self, answer: str) -> str:
        got, _, _ = session.acquire("s-strip")
        self.assertTrue(got)
        session.complete("s-strip", "问", answer, got_lock=True)
        _, history, _ = session.acquire("s-strip")
        return history[-1][1]

    def test_knowledge_answer_stripped(self):
        answer = "正文段落\n\n【参考来源】：高血压_默沙东诊疗手册.pdf - 第1页\n\n" + \
            session.respond.DISCLAIMER
        self.assertEqual(self._stored_answer(answer), "正文段落")

    def test_emergency_answer_kept(self):
        # 紧急固定文案是医疗安全信息：保留；只剥免责声明
        answer = session.respond.EMERGENCY_ANSWER + "\n\n" + session.respond.DISCLAIMER
        stored = self._stored_answer(answer)
        self.assertIn("120", stored)
        self.assertNotIn(session.respond.DISCLAIMER, stored)

    def test_risk_notice_kept(self):
        answer = session.respond.RISK_NOTICE + "\n\n正文\n\n" + session.respond.DISCLAIMER
        stored = self._stored_answer(answer)
        self.assertIn("120", stored)
        self.assertIn("正文", stored)
        self.assertNotIn(session.respond.DISCLAIMER, stored)

    def test_plain_answer_untouched(self):
        self.assertEqual(self._stored_answer("闲聊话术"), "闲聊话术")


if __name__ == "__main__":
    unittest.main()
