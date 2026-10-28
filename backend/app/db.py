"""应用数据库：SQLite 连接、幂等建表与版本迁移。

设计决策（2026-10-23 登录与用户隔离实施，见 docs/登录与用户隔离-设计方案.md）：

- 只用标准库 sqlite3，不引入 ORM/连接池。每次操作开短连接，避免跨线程共享
  sqlite3 连接（同步端点跑在 FastAPI 线程池，多线程写由 WAL + busy_timeout 覆盖）。
- 每个连接强制 PRAGMA foreign_keys=ON（外键默认关闭，不开等于没建）。
- 路径读 env（调用时读取，与 session.py 同风格），便于测试切到临时库：
  APP_DB_PATH 未设置时默认 backend/data/app.db。
- 版本迁移用 PRAGMA user_version，重复执行安全。
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

_DB_ENV = "APP_DB_PATH"

# 迁移按版本号顺序执行；user_version 记录已执行到的版本。
_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
        password_hash TEXT NOT NULL,
        display_name  TEXT NOT NULL DEFAULT '',
        is_admin      INTEGER NOT NULL DEFAULT 0,
        created_at    TEXT NOT NULL,
        last_login_at TEXT
    );

    -- 登录态（服务端会话）。token 只存 SHA-256 摘要，库里不存可用凭证。
    CREATE TABLE IF NOT EXISTS auth_sessions (
        token_hash TEXT PRIMARY KEY,
        user_id    INTEGER NOT NULL REFERENCES users(id),
        created_at REAL NOT NULL,
        expires_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id);

    -- 业务会话归属：id 即 /agent/chat 的 session_id（服务端 uuid4 生成）。
    CREATE TABLE IF NOT EXISTS conversations (
        id         TEXT PRIMARY KEY,
        user_id    INTEGER NOT NULL REFERENCES users(id),
        title      TEXT NOT NULL DEFAULT '新的对话',
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        deleted    INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_conv_user
        ON conversations(user_id, deleted, updated_at);

    -- 全量聊天历史（append-only）。不冗余 user_id：归属只能经 conversations 校验。
    CREATE TABLE IF NOT EXISTS messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL REFERENCES conversations(id),
        role            TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
        content         TEXT NOT NULL,
        created_at      REAL NOT NULL,
        turn            INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, id);
    """,
    2: """
    -- 仅多轮待续任务需要持久化；普通问答不创建任务记录。
    CREATE TABLE IF NOT EXISTS agent_tasks (
        task_id         TEXT PRIMARY KEY,
        owner_user_id   INTEGER NOT NULL REFERENCES users(id),
        conversation_id TEXT NOT NULL REFERENCES conversations(id),
        task_type       TEXT NOT NULL,
        status          TEXT NOT NULL CHECK(status IN (
            'waiting_for_input', 'ready', 'running',
            'completed', 'failed', 'cancelled')),
        required_inputs TEXT NOT NULL,
        provided_inputs TEXT NOT NULL,
        missing_inputs  TEXT NOT NULL,
        created_at      REAL NOT NULL,
        updated_at      REAL NOT NULL,
        attempt         INTEGER NOT NULL DEFAULT 0,
        result_ref      TEXT NOT NULL DEFAULT '',
        error_code      TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_agent_tasks_pending
        ON agent_tasks(owner_user_id, conversation_id, status, updated_at);

    -- 用户明确提供的事实；来源消息必须同属一个用户和会话。
    CREATE TABLE IF NOT EXISTS user_facts (
        fact_id           TEXT PRIMARY KEY,
        owner_user_id     INTEGER NOT NULL REFERENCES users(id),
        conversation_id   TEXT NOT NULL REFERENCES conversations(id),
        source_message_id INTEGER NOT NULL REFERENCES messages(id),
        fact_type         TEXT NOT NULL,
        value             REAL NOT NULL,
        unit              TEXT NOT NULL,
        event_time        REAL NOT NULL,
        recorded_at       REAL NOT NULL,
        status            TEXT NOT NULL DEFAULT 'active'
    );
    CREATE INDEX IF NOT EXISTS idx_user_facts_context
        ON user_facts(owner_user_id, conversation_id, fact_type, status, recorded_at);
    """,
}

_SCHEMA_VERSION = max(_MIGRATIONS)


def db_path() -> Path:
    raw = os.environ.get(_DB_ENV, "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parent.parent / "data" / "app.db"


def _conn_factory():
    class _Conn(sqlite3.Connection):
        """连接即上下文管理器：退出时先 commit/rollback 再 close。

        Windows 下连接不关闭会占用文件句柄，导致临时目录清理失败；
        调用方一律用 ``with connect() as conn:``，不手工 close。
        """

        def __exit__(self, exc_type, exc, tb):
            try:
                if exc_type is None:
                    self.commit()
                else:
                    self.rollback()
            finally:
                self.close()
            return False

    return _Conn


def connect() -> sqlite3.Connection:
    """开一个配置完整的新连接（用 ``with`` 管理，退出自动提交并关闭）。"""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0, factory=_conn_factory())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db() -> None:
    """幂等建表/迁移；可重复调用。user_version < 目标版本时按序补齐。"""
    with connect() as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        for v in sorted(k for k in _MIGRATIONS if k > version):
            conn.executescript(_MIGRATIONS[v])
        conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        conn.commit()


def now_ts() -> float:
    return time.time()


__all__ = ["connect", "init_db", "db_path", "now_ts", "_SCHEMA_VERSION"]
