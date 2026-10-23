"""接口级安全测试：认证、CSRF、越权矩阵、上传治理（登录与用户隔离第二阶段）。

确定性：临时 SQLite / 临时 SESSION_PERSIST_DIR / 打桩 orchestrator（不触网、
不加载模型/向量库）；TestClient 不进入上下文管理器，startup 事件不触发
（与 acceptance/runner 同一路径）。
"""

import json
import os
import tempfile
import types
import unittest
from unittest import mock

from fastapi.testclient import TestClient


def _fresh_env(testcase):
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    os.environ["APP_DB_PATH"] = os.path.join(tmp.name, "app.db")
    os.environ["SESSION_PERSIST_DIR"] = os.path.join(tmp.name, "sessions")
    testcase.addCleanup(os.environ.pop, "APP_DB_PATH", None)
    testcase.addCleanup(os.environ.pop, "SESSION_PERSIST_DIR", None)
    return tmp.name


def _make_users():
    from app.auth.passwords import hash_password
    from app.auth import users as auth_users
    uid_a = auth_users.create_user("user-a", hash_password("pw-a-123456"))
    uid_b = auth_users.create_user("user-b", hash_password("pw-b-123456"))
    uid_admin = auth_users.create_user("admin-1", hash_password("pw-admin-123"),
                                       is_admin=True)
    return uid_a, uid_b, uid_admin


_XHR = {"X-Requested-With": "XMLHttpRequest"}


def _login(client, username, password):
    return client.post("/auth/login",
                       json={"username": username, "password": password},
                       headers=_XHR)


def _fake_state(question):
    from app.medical_agent.state import QuestionCategory, TerminationReason
    return types.SimpleNamespace(
        request_id="reqtest00001",
        answer_text=f"答<{question}>",
        category=QuestionCategory.KNOWLEDGE,
        termination_reason=TerminationReason.COMPLETED,
        risk_notice="",
        sources=[],
    )


class ApiTestBase(unittest.TestCase):
    def setUp(self):
        _fresh_env(self)
        from app.auth import deps
        # 限流器是模块级单例：每个用例清零，避免相互锁定
        with deps.login_limiter._guard:
            deps.login_limiter._fails.clear()
            deps.login_limiter._ip_hits.clear()
        self.uid_a, self.uid_b, self.uid_admin = _make_users()
        from app.rag.main import app
        self.app = app

    def _set_owner(self, owner):
        """在测试线程设置请求作用域 owner；用完必须复位（contextvars 会污染
        同一线程的后续测试，例如 test_session_persist 读不到 owner 才会走
        扁平目录——泄漏会让它的断言落到子目录而失败）。"""
        from app.auth import deps
        token = deps.current_owner.set(owner)
        self.addCleanup(deps.current_owner.reset, token)
        return token

    def _client(self):
        # base_url 用 https：Secure Cookie 不会经 http 发送（http.cookiejar
        # 策略），与真实浏览器对 localhost 的"可信源"待遇对齐
        return TestClient(self.app, base_url="https://testserver")

    def _chat(self, client, sid, question):
        return client.post("/agent/chat",
                           json={"session_id": sid, "question": question},
                           headers=_XHR)


# ==================== 认证 ====================


class TestAuthFlow(ApiTestBase):
    def test_unauthenticated_rejected_everywhere(self):
        c = self._client()
        for method, url, kwargs in [
            ("get", "/conversations", {}),
            ("get", "/conversations/abc/messages", {}),
            ("delete", "/conversations/abc", {"headers": _XHR}),
            ("post", "/agent/chat",
             {"json": {"session_id": "s1", "question": "q"}, "headers": _XHR}),
            ("post", "/upload", {"headers": _XHR}),
        ]:
            r = getattr(c, method)(url, **kwargs)
            self.assertEqual(r.status_code, 401, f"{method} {url} 应 401")

    def test_login_me_logout_flow(self):
        c = self._client()
        # 错密码 401
        r = _login(c, "user-a", "wrong-password")
        self.assertEqual(r.status_code, 401)
        # 未知用户名：同一状态码同一话术（防枚举）
        r_unknown = _login(c, "no-such-user", "whatever-pass")
        self.assertEqual(r_unknown.status_code, 401)
        self.assertEqual(r_unknown.json()["detail"], r.json()["detail"])
        # 正确登录：Cookie 属性检查
        r = _login(c, "user-a", "pw-a-123456")
        self.assertEqual(r.status_code, 200)
        set_cookie = r.headers.get("set-cookie", "")
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("Secure", set_cookie)
        self.assertIn("Path=/", set_cookie)
        body = r.json()
        self.assertEqual(body["username"], "user-a")
        self.assertNotIn("password", json.dumps(body))
        # /auth/me
        r = c.get("/auth/me")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["id"], self.uid_a)
        # 登出后 me 401
        r = c.post("/auth/logout", headers=_XHR)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(c.get("/auth/me").status_code, 401)

    def test_forged_cookie_rejected(self):
        c = self._client()
        c.cookies.set("med_session", "forged-token-value")
        self.assertEqual(c.get("/conversations").status_code, 401)

    def test_login_csrf_header_required(self):
        c = self._client()
        r = c.post("/auth/login",
                   json={"username": "user-a", "password": "pw-a-123456"})
        self.assertEqual(r.status_code, 403)

    def test_bruteforce_lockout(self):
        c = self._client()
        for _ in range(5):
            _login(c, "user-a", "bad-password")
        # 锁定后即使密码正确也拒绝（403 锁定话术）
        r = _login(c, "user-a", "pw-a-123456")
        self.assertEqual(r.status_code, 403)
        # 其他用户/来源不受影响
        self.assertEqual(_login(c, "user-b", "pw-b-123456").status_code, 200)


# ==================== 会话归属 ====================


class TestConversationAuthz(ApiTestBase):
    def test_list_isolation(self):
        ca, cb = self._client(), self._client()
        _login(ca, "user-a", "pw-a-123456")
        _login(cb, "user-b", "pw-b-123456")
        a1 = ca.post("/conversations", json={"title": "A1"}, headers=_XHR).json()
        ca.post("/conversations", json={"title": "A2"}, headers=_XHR)
        cb.post("/conversations", json={"title": "B1"}, headers=_XHR)
        a_ids = {c["id"] for c in ca.get("/conversations").json()["conversations"]}
        b_ids = {c["id"] for c in cb.get("/conversations").json()["conversations"]}
        self.assertEqual(len(a_ids), 2)
        self.assertEqual(len(b_ids), 1)
        self.assertFalse(a_ids & b_ids)

    def test_history_cross_user_forbidden_same_as_missing(self):
        ca, cb = self._client(), self._client()
        _login(ca, "user-a", "pw-a-123456")
        _login(cb, "user-b", "pw-b-123456")
        cid = ca.post("/conversations", json={}, headers=_XHR).json()["id"]
        r_forbid = cb.get(f"/conversations/{cid}/messages")
        r_missing = cb.get("/conversations/no-such/messages")
        self.assertEqual(r_forbid.status_code, 403)
        self.assertEqual(r_missing.status_code, 403)
        self.assertEqual(r_forbid.json()["detail"], r_missing.json()["detail"])

    def test_delete_cross_user_forbidden(self):
        ca, cb = self._client(), self._client()
        _login(ca, "user-a", "pw-a-123456")
        _login(cb, "user-b", "pw-b-123456")
        cid = ca.post("/conversations", json={}, headers=_XHR).json()["id"]
        r = cb.request("DELETE", f"/conversations/{cid}", headers=_XHR)
        self.assertEqual(r.status_code, 403)
        #  owner's list unchanged
        self.assertEqual(len(ca.get("/conversations").json()["conversations"]), 1)


# ==================== /agent/chat ====================


class TestAgentChatAuthz(ApiTestBase):
    def setUp(self):
        super().setUp()
        # 打桩 run_agent，但保留真实 session 读写：让 owner 上下文注入与
        # 按用户目录持久化走真实代码路径（真编排器只是多做了节点逻辑）
        from app.medical_agent import session as runtime
        self.runtime = runtime

        def fake_run_agent(sid, q):
            got, _, _ = runtime.acquire(sid)
            state = _fake_state(q)
            runtime.complete(sid, q, state.answer_text, got_lock=got)
            return state

        self._patcher = mock.patch.multiple(
            "app.medical_agent.orchestrator",
            run_agent=mock.DEFAULT, display_sources=mock.DEFAULT,
        )
        self.mocks = self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self.mocks["run_agent"].side_effect = fake_run_agent
        self.mocks["display_sources"].return_value = []

    def test_chat_requires_csrf_header(self):
        c = self._client()
        _login(c, "user-a", "pw-a-123456")
        r = c.post("/agent/chat",
                   json={"session_id": "s1", "question": "q"})
        self.assertEqual(r.status_code, 403)

    def test_chat_invalid_session_id_400(self):
        c = self._client()
        _login(c, "user-a", "pw-a-123456")
        for bad in ("../evil", "x" * 65, "", "a b"):
            r = self._chat(c, bad, "问题")
            self.assertEqual(r.status_code, 400, f"sid={bad!r}")

    def test_chat_happy_path_writes_history_and_owner_file(self):
        c = self._client()
        _login(c, "user-a", "pw-a-123456")
        cid = c.post("/conversations", json={}, headers=_XHR).json()["id"]
        r = self._chat(c, cid, "高血压注意什么")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["answer"], "答<高血压注意什么>")
        # 全量历史双写
        msgs = c.get(f"/conversations/{cid}/messages").json()["messages"]
        self.assertEqual([(m["role"], m["content"]) for m in msgs],
                         [("user", "高血压注意什么"), ("assistant", "答<高血压注意什么>")])
        # 运行时上下文按 owner 子目录持久化
        persist = os.environ["SESSION_PERSIST_DIR"]
        owner_file = os.path.join(persist, str(self.uid_a), f"{cid}.json")
        self.assertTrue(os.path.exists(owner_file), owner_file)
        payload = json.load(open(owner_file, encoding="utf-8"))
        self.assertEqual(payload["messages"][0], ["user", "高血压注意什么"])

    def test_chat_cross_user_forbidden_and_untouched(self):
        ca, cb = self._client(), self._client()
        _login(ca, "user-a", "pw-a-123456")
        _login(cb, "user-b", "pw-b-123456")
        cid = ca.post("/conversations", json={}, headers=_XHR).json()["id"]
        before = ca.get(f"/conversations/{cid}/messages").json()["messages"]
        r = self._chat(cb, cid, "窃取问题")
        self.assertEqual(r.status_code, 403)
        self.assertEqual(ca.get(f"/conversations/{cid}/messages").json()["messages"],
                         before)
        # B 的越权请求未进入编排（run_agent 只可能被合法路径调用，此处应零调用）
        self.mocks["run_agent"].assert_not_called()

    def test_chat_unknown_session_claimed_by_sender(self):
        ca, cb = self._client(), self._client()
        _login(ca, "user-a", "pw-a-123456")
        _login(cb, "user-b", "pw-b-123456")
        legacy_sid = "sess-legacy0001"
        r = self._chat(ca, legacy_sid, "问题")
        self.assertEqual(r.status_code, 200)
        # 已被 A 认领：B 再用同一 sid → 403
        self.assertEqual(self._chat(cb, legacy_sid, "问题").status_code, 403)


# ==================== 删除级联 ====================


class TestDeleteCascade(ApiTestBase):
    def test_delete_removes_runtime_state(self):
        from app.medical_agent import session as runtime
        runtime._SESSIONS.clear()
        self._set_owner(self.uid_a)  # 与真实请求一致：运行时上下文带 owner
        c = self._client()
        _login(c, "user-a", "pw-a-123456")
        cid = c.post("/conversations", json={}, headers=_XHR).json()["id"]
        got, _, _ = runtime.acquire(cid)
        self.assertTrue(got)
        runtime.complete(cid, "问", "答", got_lock=True)
        persist = os.environ["SESSION_PERSIST_DIR"]
        owner_file = os.path.join(persist, str(self.uid_a), f"{cid}.json")
        self.assertTrue(os.path.exists(owner_file))

        r = c.request("DELETE", f"/conversations/{cid}", headers=_XHR)
        self.assertEqual(r.status_code, 200)
        self.assertFalse(os.path.exists(owner_file))
        runtime._SESSIONS.clear()
        # 级联后按新会话处理（历史随软删不可读，运行时无残留）
        self.assertIsNone(runtime._load_from_disk(cid, self.uid_a))


# ==================== /upload 治理 ====================


class TestUploadGate(ApiTestBase):
    def _upload(self, client, filename):
        return client.post(
            "/upload", files={"file": (filename, b"%PDF-fake", "application/pdf")},
            headers=_XHR,
        )

    def test_unauthenticated_and_non_admin_rejected(self):
        c = self._client()
        self.assertEqual(self._upload(c, "a.pdf").status_code, 401)
        _login(c, "user-a", "pw-a-123456")
        self.assertEqual(self._upload(c, "a.pdf").status_code, 403)

    def test_admin_filename_sanitized(self):
        from app.rag import main as main_mod
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        old_dir = main_mod.UPLOAD_DIR
        main_mod.UPLOAD_DIR = tmp.name
        self.addCleanup(setattr, main_mod, "UPLOAD_DIR", old_dir)

        c = self._client()
        _login(c, "admin-1", "pw-admin-123")
        r = self._upload(c, "../../evil.pdf")
        self.assertEqual(r.status_code, 200)
        saved = os.listdir(tmp.name)
        self.assertEqual(len(saved), 1)
        self.assertNotIn("..", saved[0])
        self.assertTrue(saved[0].endswith("_evil.pdf"), saved[0])
        # 净化名应在临时目录内，且穿越目标不存在
        self.assertFalse(os.path.exists(os.path.join(tmp.name, "..", "evil.pdf")))


# ==================== 历史导入 ====================


class TestImportHistory(ApiTestBase):
    def test_import_requires_auth_and_writes_history(self):
        c = self._client()
        payload = {"conversations": [{
            "title": "旧对话",
            "created_at": 1700000000,
            "messages": [{"role": "user", "content": "旧问题"},
                         {"role": "assistant", "content": "旧回答"},
                         {"role": "hacker", "content": "应被跳过"},
                         {"role": "user", "content": "第二问"}],
        }]}
        self.assertEqual(c.post("/conversations/import", json=payload,
                                headers=_XHR).status_code, 401)
        _login(c, "user-a", "pw-a-123456")
        r = c.post("/conversations/import", json=payload, headers=_XHR)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["imported"], 1)
        convs = c.get("/conversations").json()["conversations"]
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0]["title"], "旧对话")
        msgs = c.get(f"/conversations/{convs[0]['id']}/messages").json()["messages"]
        self.assertEqual([(m["role"], m["content"], m["turn"]) for m in msgs],
                         [("user", "旧问题", 1), ("assistant", "旧回答", 1),
                          ("user", "第二问", 2)])
        # 他人不可见
        cb = self._client()
        _login(cb, "user-b", "pw-b-123456")
        self.assertEqual(cb.get("/conversations").json()["conversations"], [])
        self.assertEqual(cb.get(f"/conversations/{convs[0]['id']}/messages")
                         .status_code, 403)

    def test_import_csrf_required_and_caps(self):
        c = self._client()
        _login(c, "user-a", "pw-a-123456")
        self.assertEqual(c.post("/conversations/import",
                                json={"conversations": []}).status_code, 403)
        many = [{"title": f"c{i}", "messages": [{"role": "user", "content": "q"}]}
                for i in range(250)]
        r = c.post("/conversations/import", json={"conversations": many},
                   headers=_XHR)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["imported"], 200)  # 封顶 200


# ==================== 启动自检 ====================


class TestStartupConfigCheck(unittest.TestCase):
    def test_star_origin_rejected(self):
        from app.rag import main as main_mod
        with mock.patch.object(main_mod, "_ALLOWED_ORIGINS", ["*"]):
            with self.assertRaises(RuntimeError):
                main_mod._check_auth_config()

    def test_samesite_none_requires_secure(self):
        from app.rag import main as main_mod
        env = {"SESSION_COOKIE_SAMESITE": "none", "SESSION_COOKIE_SECURE": "false"}
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaises(RuntimeError):
                main_mod._check_auth_config()

    def test_valid_config_passes(self):
        from app.rag import main as main_mod
        env = {"SESSION_COOKIE_SAMESITE": "lax", "SESSION_COOKIE_SECURE": "true"}
        with mock.patch.object(main_mod, "_ALLOWED_ORIGINS",
                               ["http://localhost:3000"]):
            with mock.patch.dict(os.environ, env, clear=False):
                main_mod._check_auth_config()  # 不抛即通过


# ==================== 会话模块 owner 语义 ====================


class TestSessionOwnerSemantics(unittest.TestCase):
    def setUp(self):
        _fresh_env(self)
        from app.medical_agent import session as runtime
        self.runtime = runtime
        runtime._SESSIONS.clear()
        self.addCleanup(runtime._SESSIONS.clear)

    def _set_owner(self, owner):
        from app.auth import deps
        token = deps.current_owner.set(owner)
        # contextvars 会污染同一线程后续测试：用完必须复位（否则
        # test_session_persist 等旧测试会读到 owner、文件落进子目录而失败）
        self.addCleanup(deps.current_owner.reset, token)
        return token

    def test_owner_mismatch_read_gives_no_history_write_refused(self):
        rt = self.runtime
        self._set_owner(1)
        rt.acquire("conv-x")
        rt.complete("conv-x", "问A", "答A", got_lock=True)
        # 另一个 owner 读取：宁可无历史也不泄露
        self._set_owner(2)
        got, history, status = rt.acquire("conv-x")
        self.assertEqual(history, [])
        # 另一个 owner 写入：被拒绝，A 的消息不受影响
        rt.complete("conv-x", "问B", "答B", got_lock=True)
        self.assertEqual(list(rt._SESSIONS["conv-x"].messages),
                         [("user", "问A"), ("assistant", "答A")])

    def test_owner_file_layout_and_drop(self):
        rt = self.runtime
        self._set_owner(7)
        rt.acquire("conv-y")
        rt.complete("conv-y", "问", "答", got_lock=True)
        persist = os.environ["SESSION_PERSIST_DIR"]
        owner_file = os.path.join(persist, "7", "conv-y.json")
        self.assertTrue(os.path.exists(owner_file))
        rt.drop_conversation("conv-y", owner=7)
        self.assertFalse(os.path.exists(owner_file))
        self.assertNotIn("conv-y", rt._SESSIONS)

    def test_drop_wrong_owner_keeps_data(self):
        rt = self.runtime
        self._set_owner(7)
        rt.acquire("conv-z")
        rt.complete("conv-z", "问", "答", got_lock=True)
        rt.drop_conversation("conv-z", owner=8)  # 归属不符
        persist = os.environ["SESSION_PERSIST_DIR"]
        self.assertTrue(os.path.exists(os.path.join(persist, "7", "conv-z.json")))
        self.assertIn("conv-z", rt._SESSIONS)


if __name__ == "__main__":
    unittest.main()
