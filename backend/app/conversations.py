"""业务会话与全量历史：归属校验的唯一权威层（授权在这里，不在 session.py）。

职责（设计方案 §2/§3.3）：

- 会话 id 由本模块以 uuid4 生成（服务端生成，前端无权决定归属）；
- 每个资源级操作先查 conversations 并校验 ``user_id == 当前用户``，
  失败一律 403——**不区分"会话不存在"与"不属于你"**，避免通过状态码差异
  探测他人会话是否存在；
- messages 表 append-only；读取只能经归属校验后的会话；
- 删除 = 软删（审计/可恢复）+ 级联清 session.py 的运行时条目与持久化文件。

运行时上下文窗口（最近 5 轮）仍归 session.py；本模块双写全量历史，
两者职责不重叠（设计方案 §5.2）。
"""

from __future__ import annotations

import sqlite3
import uuid

from fastapi import HTTPException, status

from .db import connect, init_db, now_ts

# 与 orchestrator 一致：401 用固定话术；403 同样不泄露资源存在性。
_FORBIDDEN = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="无权访问该会话。",
)

_SESSION_CASCADE_ERROR = (
    "删除业务会话时级联清理运行时上下文失败（不影响软删结果）"
)


def create_conversation(user_id: int, title: str = "新的对话",
                        *, now: float | None = None,
                        conv_id: str | None = None) -> str:
    """创建会话，返回服务端生成的会话 id（uuid4 hex）。

    conv_id 仅允许 claim_or_verify 传入（客户端提交的、已通过格式白名单的
    既有会话标识）；直接调用方必须保持 None，由服务端生成。
    """
    init_db()
    ts = now_ts() if now is None else now
    conv_id = conv_id or uuid.uuid4().hex
    with connect() as conn:
        conn.execute(
            "INSERT INTO conversations(id, user_id, title, created_at, updated_at)"
            " VALUES(?,?,?,?,?)",
            (conv_id, user_id, title, ts, ts),
        )
        conn.commit()
    return conv_id


def _get_row(conn: sqlite3.Connection, conv_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM conversations WHERE id=? AND deleted=0", (conv_id,),
    ).fetchone()


def get_owned_conversation(conn: sqlite3.Connection, conv_id: str,
                           user_id: int) -> sqlite3.Row:
    """归属校验的唯一入口：不存在或不属于当前用户都抛同一 403。"""
    row = _get_row(conn, conv_id)
    if row is None or int(row["user_id"]) != int(user_id):
        raise _FORBIDDEN
    return row


def list_conversations(user_id: int, *, limit: int = 200) -> list[dict]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations"
            " WHERE user_id=? AND deleted=0"
            " ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def append_turn(conv_id: str, user_id: int, question: str, answer: str,
                *, now: float | None = None) -> dict[str, int]:
    """双写一轮问答，并返回消息 id 供结构化事实追溯来源。"""
    init_db()
    ts = now_ts() if now is None else now
    with connect() as conn:
        get_owned_conversation(conn, conv_id, user_id)
        turn = conn.execute(
            "SELECT COALESCE(MAX(turn), 0) AS t FROM messages"
            " WHERE conversation_id=?", (conv_id,),
        ).fetchone()["t"] + 1
        user_message_id = conn.execute(
            "INSERT INTO messages(conversation_id, role, content, created_at, turn)"
            " VALUES(?,?,?,?,?)", (conv_id, "user", question, ts, turn),
        ).lastrowid
        assistant_message_id = conn.execute(
            "INSERT INTO messages(conversation_id, role, content, created_at, turn)"
            " VALUES(?,?,?,?,?)", (conv_id, "assistant", answer, ts, turn),
        ).lastrowid
        conn.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?", (ts, conv_id),
        )
        conn.commit()
        return {"user": user_message_id, "assistant": assistant_message_id}


def list_messages(conv_id: str, user_id: int, *, limit: int = 500) -> list[dict]:
    init_db()
    with connect() as conn:
        get_owned_conversation(conn, conv_id, user_id)
        rows = conn.execute(
            "SELECT id, role, content, created_at, turn FROM messages"
            " WHERE conversation_id=? ORDER BY id LIMIT ?",
            (conv_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_conversation(conv_id: str, user_id: int) -> None:
    """软删 + 级联清理运行时上下文（内存条目与持久化文件）。

    级联失败只记日志，不回滚软删（运行时上下文是易失缓存，可空）。
    """
    init_db()
    with connect() as conn:
        get_owned_conversation(conn, conv_id, user_id)
        conn.execute(
            "UPDATE conversations SET deleted=1, updated_at=? WHERE id=?",
            (now_ts(), conv_id),
        )
        conn.commit()
    _cascade_drop_runtime(conv_id, user_id)


def _cascade_drop_runtime(conv_id: str, user_id: int) -> None:
    from .medical_agent import session as runtime_session
    drop = getattr(runtime_session, "drop_conversation", None)
    if drop is None:
        return  # session.py 级联删除接口属第二阶段改造，未接入前跳过
    try:
        drop(conv_id, owner=user_id)
    except Exception as exc:  # noqa: BLE001 —— 级联清理不得影响删除主流程
        print(f"[conversations] {_SESSION_CASCADE_ERROR}: {type(exc).__name__}")


def claim_or_verify(conv_id: str, user_id: int) -> str:
    """POST /agent/chat 入口策略（设计方案 §3.3）：

    - 会话已存在且属于他人 → 403；
    - 不存在 → 隐式创建并归属当前用户（保留直接开聊体验；
      他人的 id 必然已存在，不存在抢注他人数据的风险）；
    - 存在且属于本人 → 原样返回。
    """
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT user_id, deleted FROM conversations WHERE id=?", (conv_id,),
        ).fetchone()
        if row is not None:
            if int(row["user_id"]) != int(user_id):
                raise _FORBIDDEN
            if row["deleted"]:
                # 已删会话被重新使用：复活并刷新归属时间（仍是本人）
                conn.execute(
                    "UPDATE conversations SET deleted=0, updated_at=? WHERE id=?",
                    (now_ts(), conv_id),
                )
                conn.commit()
            return conv_id
    return create_conversation(user_id, conv_id=conv_id)


def import_conversations(user_id: int, items: list[dict],
                         *, now: float | None = None) -> int:
    """一次性导入历史会话（前端 localStorage 迁移，决策 #10）。

    每个 item: {"title": str, "created_at": float, "messages": [{"role",
    "content"}...]}。逐会话建归属并 append-only 写消息；turn 按 user 消息
    自增（与 append_turn 口径一致）。非法角色/空内容跳过，不中断整批。
    返回导入的会话数。
    """
    init_db()
    ts = now_ts() if now is None else now
    imported = 0
    for item in items:
        raw_msgs = item.get("messages") or []
        if not isinstance(raw_msgs, list) or not raw_msgs:
            continue
        title = str(item.get("title") or "新的对话")[:100]
        created = item.get("created_at")
        conv_id = create_conversation(
            user_id, title, now=float(created) if isinstance(created, (int, float)) else ts)
        turn = 0
        rows = []
        for m in raw_msgs:
            if not isinstance(m, dict):
                continue
            role, content = m.get("role"), m.get("content")
            if role not in ("user", "assistant") or not content:
                continue
            if role == "user":
                turn += 1
            rows.append((conv_id, role, str(content), ts, max(turn, 1)))
        if rows:
            with connect() as conn:
                conn.executemany(
                    "INSERT INTO messages(conversation_id, role, content,"
                    " created_at, turn) VALUES(?,?,?,?,?)", rows)
                conn.commit()
        imported += 1
    return imported


__all__ = [
    "create_conversation",
    "list_conversations",
    "append_turn",
    "list_messages",
    "delete_conversation",
    "claim_or_verify",
    "get_owned_conversation",
    "import_conversations",
]
