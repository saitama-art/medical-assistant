"""登录与用户隔离第一阶段测试：密码哈希、登录态、限流、业务会话归属。

全部确定性：临时 SQLite（APP_DB_PATH 环境变量切换）+ 注入时钟，
不触网、不加载模型。现状检查报告要求的 test_session_persist.py
保持独立、本文件不改它。
"""

import os
import tempfile
import unittest

from fastapi import HTTPException


def _fresh_db(testcase):
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    os.environ["APP_DB_PATH"] = os.path.join(tmp.name, "test.db")
    testcase.addCleanup(os.environ.pop, "APP_DB_PATH", None)


# ==================== 密码哈希 ====================


class TestPasswords(unittest.TestCase):
    def setUp(self):
        from app.auth import passwords
        self.pw = passwords

    def test_roundtrip(self):
        stored = self.pw.hash_password("correct-horse-9")
        self.assertTrue(self.pw.verify_password("correct-horse-9", stored))
        self.assertFalse(self.pw.verify_password("wrong-password", stored))

    def test_format_self_describing(self):
        stored = self.pw.hash_password("x" * 12)
        prefix, n, r, p, salt, dk = stored.split("$")
        self.assertEqual(prefix, "scrypt")
        self.assertEqual(int(n), 32768)
        self.assertEqual(len(bytes.fromhex(dk)), 64)

    def test_salt_randomized(self):
        self.assertNotEqual(self.pw.hash_password("same-password-1"),
                            self.pw.hash_password("same-password-1"))

    def test_verify_never_raises_on_garbage(self):
        for garbage in ("", "not-a-hash", "scrypt$bad$$$", None, 123):
            self.assertFalse(self.pw.verify_password("pw", garbage))

    def test_empty_password_rejected(self):
        with self.assertRaises(ValueError):
            self.pw.hash_password("")


# ==================== 登录态 ====================


class TestAuthSessions(unittest.TestCase):
    def setUp(self):
        _fresh_db(self)
        from app.auth import sessions, users
        from app.auth.passwords import hash_password
        self.sessions = sessions
        self.users = users
        self.uid = users.create_user("alice", hash_password("pw-alice-123"))

    def test_create_validate_revoke(self):
        token = self.sessions.create_session(self.uid, now=1_000.0)
        self.assertTrue(token)
        self.assertEqual(self.sessions.validate_token(token, now=2_000.0), self.uid)
        self.sessions.revoke_session(token)
        self.assertIsNone(self.sessions.validate_token(token, now=2_100.0))

    def test_expiry_sliding_renewal(self):
        t0 = 100_000.0
        token = self.sessions.create_session(self.uid, now=t0)
        # 过了 60% 空闲 TTL：触发滑动续期，过期点推到 now + 7d
        later = t0 + self.sessions.SESSION_IDLE_TTL * 0.6
        self.assertEqual(self.sessions.validate_token(token, now=later), self.uid)
        # 续期后可再活满一个空闲 TTL
        still = later + self.sessions.SESSION_IDLE_TTL * 0.9
        self.assertEqual(self.sessions.validate_token(token, now=still), self.uid)
        # 超过空闲 TTL 未活动 → 失效并删除
        dead = still + self.sessions.SESSION_IDLE_TTL
        self.assertIsNone(self.sessions.validate_token(token, now=dead))

    def test_absolute_cap(self):
        t0 = 0.0
        token = self.sessions.create_session(self.uid, now=t0)
        # 每次都在活动，但超过绝对 30 天必须重新登录
        beyond = t0 + self.sessions.SESSION_ABSOLUTE_TTL + 1
        self.assertIsNone(self.sessions.validate_token(token, now=beyond))

    def test_unknown_and_empty_token(self):
        self.assertIsNone(self.sessions.validate_token("no-such-token", now=1.0))
        self.assertIsNone(self.sessions.validate_token("", now=1.0))

    def test_cleanup_expired(self):
        self.sessions.create_session(self.uid, now=0.0)
        removed = self.sessions.cleanup_expired(now=40 * 24 * 3600.0)
        self.assertGreaterEqual(removed, 1)


# ==================== 限流 ====================


class TestLoginRateLimiter(unittest.TestCase):
    def _make(self):
        from app.auth.deps import LoginRateLimiter
        clock = {"t": 0.0}
        limiter = LoginRateLimiter(clock=lambda: clock["t"])
        return limiter, clock

    def test_lock_after_max_fails_and_release(self):
        limiter, clock = self._make()
        for _ in range(5):
            limiter.record_fail("Alice", "10.0.0.1")  # 大小写不敏感键
        self.assertTrue(limiter.is_locked("alice", "10.0.0.1"))
        clock["t"] = 16 * 60  # 超过 15 分钟锁定期
        self.assertFalse(limiter.is_locked("alice", "10.0.0.1"))

    def test_success_clears_fails(self):
        limiter, _ = self._make()
        limiter.record_fail("bob", "10.0.0.2")
        limiter.record_fail("bob", "10.0.0.2")
        limiter.record_success("bob", "10.0.0.2")
        self.assertFalse(limiter.is_locked("bob", "10.0.0.2"))

    def test_ip_minute_budget(self):
        limiter, clock = self._make()
        for _ in range(30):
            self.assertTrue(limiter.check_ip_budget("10.0.0.9"))
            limiter.record_fail("user%d" % _, "10.0.0.9")
        self.assertFalse(limiter.check_ip_budget("10.0.0.9"))
        clock["t"] = 61  # 过一分钟窗口
        self.assertTrue(limiter.check_ip_budget("10.0.0.9"))

    def test_different_ip_not_locked(self):
        limiter, _ = self._make()
        for _ in range(5):
            limiter.record_fail("carol", "10.0.0.1")
        self.assertFalse(limiter.is_locked("carol", "10.0.0.2"))


# ==================== 业务会话归属 ====================


class TestConversations(unittest.TestCase):
    def setUp(self):
        _fresh_db(self)
        from app.auth.passwords import hash_password
        from app.auth import users
        from app import conversations
        self.conv = conversations
        self.user_a = users.create_user("user-a", hash_password("pw-a-123456"))
        self.user_b = users.create_user("user-b", hash_password("pw-b-123456"))

    def test_create_and_list_isolated(self):
        a1 = self.conv.create_conversation(self.user_a, "会话A1", now=1.0)
        a2 = self.conv.create_conversation(self.user_a, "会话A2", now=2.0)
        self.conv.create_conversation(self.user_b, "会话B1", now=3.0)
        a_list = self.conv.list_conversations(self.user_a)
        b_list = self.conv.list_conversations(self.user_b)
        self.assertEqual([c["id"] for c in a_list], [a2, a1])  # updated_at 倒序
        self.assertEqual(len(b_list), 1)

    def test_history_cross_user_forbidden(self):
        cid = self.conv.create_conversation(self.user_a, now=1.0)
        with self.assertRaises(HTTPException) as ctx:
            self.conv.list_messages(cid, self.user_b)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_nonexistent_same_403(self):
        # 不存在与他人会话返回同一状态码，不泄露存在性
        with self.assertRaises(HTTPException) as ctx:
            self.conv.list_messages("no-such-id", self.user_b)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_append_and_read_turns(self):
        cid = self.conv.create_conversation(self.user_a, now=1.0)
        self.conv.append_turn(cid, self.user_a, "问一", "答一", now=2.0)
        self.conv.append_turn(cid, self.user_a, "问二", "答二", now=3.0)
        msgs = self.conv.list_messages(cid, self.user_a)
        self.assertEqual([(m["role"], m["content"], m["turn"]) for m in msgs],
                         [("user", "问一", 1), ("assistant", "答一", 1),
                          ("user", "问二", 2), ("assistant", "答二", 2)])

    def test_append_turn_forbidden_for_other(self):
        cid = self.conv.create_conversation(self.user_a, now=1.0)
        with self.assertRaises(HTTPException) as ctx:
            self.conv.append_turn(cid, self.user_b, "问", "答", now=2.0)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_delete_soft_and_forbidden_for_other(self):
        cid = self.conv.create_conversation(self.user_a, now=1.0)
        self.conv.append_turn(cid, self.user_a, "问", "答", now=2.0)
        with self.assertRaises(HTTPException):
            self.conv.delete_conversation(cid, self.user_b)
        self.conv.delete_conversation(cid, self.user_a)
        # 软删后：列表消失、历史 403、他人仍 403
        self.assertEqual(self.conv.list_conversations(self.user_a), [])
        with self.assertRaises(HTTPException):
            self.conv.list_messages(cid, self.user_a)

    def test_claim_or_verify_matrix(self):
        cid = self.conv.create_conversation(self.user_a, now=1.0)
        # 本人：原样返回
        self.assertEqual(self.conv.claim_or_verify(cid, self.user_a), cid)
        # 他人：403
        with self.assertRaises(HTTPException) as ctx:
            self.conv.claim_or_verify(cid, self.user_b)
        self.assertEqual(ctx.exception.status_code, 403)
        # 未知 id：按提交 id 隐式创建并归属本人（客户端 id 原样保留，
        # 否则 /agent/chat 后续用同一 session_id 会查无此会话、归属失效）
        legacy_id = "f" * 32
        new_id = self.conv.claim_or_verify(legacy_id, self.user_b)
        self.assertEqual(new_id, legacy_id)
        self.assertEqual(self.conv.claim_or_verify(new_id, self.user_b), new_id)
        with self.assertRaises(HTTPException):
            self.conv.claim_or_verify(new_id, self.user_a)


# ==================== 用户存储 ====================


class TestUsers(unittest.TestCase):
    def setUp(self):
        _fresh_db(self)
        from app.auth import users
        from app.auth.passwords import hash_password
        self.users = users
        self.hash = lambda p: hash_password(p)

    def test_username_case_insensitive_unique(self):
        self.users.create_user("Dave", self.hash("pw-dave-1234"))
        with self.assertRaises(self.users.UserStoreError):
            self.users.create_user("dave", self.hash("pw-dave-5678"))

    def test_lookup_roundtrip(self):
        uid = self.users.create_user("erin", self.hash("pw-erin-1234"),
                                     display_name="艾琳", is_admin=True)
        row = self.users.get_by_username("ERIN")
        self.assertEqual(int(row["id"]), uid)
        self.assertEqual(row["display_name"], "艾琳")
        self.assertEqual(int(row["is_admin"]), 1)
        self.assertEqual(self.users.get_by_id(uid)["username"], row["username"])
        self.assertIsNone(self.users.get_by_username("nobody"))

class TestConversationAutoTitle(unittest.TestCase):
    """首轮问答自动命名：默认标题 → 首轮问题（截断 20 字），不覆盖自定义标题。"""

    def setUp(self):
        _fresh_db(self)
        from app.auth.passwords import hash_password
        from app.auth import users
        from app import conversations
        self.conv = conversations
        self.user = users.create_user("user-t", hash_password("pw-t-123456"))

    def _title(self, cid):
        return next(c["title"] for c in self.conv.list_conversations(self.user)
                    if c["id"] == cid)

    def test_first_turn_renames_default_title(self):
        cid = self.conv.create_conversation(self.user, now=1.0)
        self.conv.append_turn(cid, self.user, "高血压患者在生活方式上要注意什么？", "答",
                              now=2.0)
        self.assertEqual(self._title(cid), "高血压患者在生活方式上要注意什么？")

    def test_title_truncated_to_20_chars(self):
        cid = self.conv.create_conversation(self.user, now=1.0)
        long_q = "这是一个非常非常长的问题超过二十个字需要被截断处理才行"
        self.conv.append_turn(cid, self.user, long_q, "答", now=2.0)
        self.assertEqual(self._title(cid), long_q[:20])

    def test_second_turn_keeps_title(self):
        cid = self.conv.create_conversation(self.user, now=1.0)
        self.conv.append_turn(cid, self.user, "第一个问题", "答", now=2.0)
        self.conv.append_turn(cid, self.user, "第二个问题", "答", now=3.0)
        self.assertEqual(self._title(cid), "第一个问题")

    def test_custom_title_preserved(self):
        cid = self.conv.create_conversation(self.user, "我的自定义标题", now=1.0)
        self.conv.append_turn(cid, self.user, "任意问题", "答", now=2.0)
        self.assertEqual(self._title(cid), "我的自定义标题")

    def test_newlines_replaced_with_space(self):
        cid = self.conv.create_conversation(self.user, now=1.0)
        self.conv.append_turn(cid, self.user, "第一行\n第二行", "答", now=2.0)
        self.assertEqual(self._title(cid), "第一行 第二行")


class TestTitleBackfillMigration(unittest.TestCase):
    """迁移 v3：存量默认标题会话按首轮用户问题回填，幂等。"""

    def setUp(self):
        _fresh_db(self)
        from app.auth.passwords import hash_password
        from app.auth import users
        from app import conversations, db
        self.conv, self.db = conversations, db
        self.user = users.create_user("user-m", hash_password("pw-m-123456"))

    def test_backfill_renames_and_is_idempotent(self):
        cid = self.conv.create_conversation(self.user, now=1.0)
        self.conv.append_turn(cid, self.user, "狗咬了怎么办", "答", now=2.0)
        empty = self.conv.create_conversation(self.user, now=3.0)
        with self.db.connect() as conn:
            conn.execute("UPDATE conversations SET title='新的对话' WHERE id=?", (cid,))
            conn.execute("PRAGMA user_version=2")
        self.db.init_db()  # 重跑 v3 回填
        titles = {c["id"]: c["title"] for c in self.conv.list_conversations(self.user)}
        self.assertEqual(titles[cid], "狗咬了怎么办")
        self.assertEqual(titles[empty], "新的对话")  # 无消息的会话不动
        with self.db.connect() as conn:
            self.assertEqual(
                conn.execute("PRAGMA user_version").fetchone()[0], self.db._SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
    unittest.main()
