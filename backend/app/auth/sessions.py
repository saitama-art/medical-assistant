"""服务端登录会话：签发、校验（滑动过期）、吊销、惰性清理。

安全要点：

- Cookie 里是可逆的随机 token（secrets.token_urlsafe(32)），数据库只存其
  SHA-256 摘要——库文件泄露不泄露任何可用凭证。
- 滑动过期：空闲 7 天；每次校验通过时剩余不足一半即续期。绝对上限 30 天
  （created_at 起算），到期必须重新登录。
- 登出 = 删除该行（不留 revoked 标记行，直接删更干净）。
- 所有时间经 now 参数可注入，测试确定性（审查报告 §2）。

与 app/medical_agent/session.py 的关系：那是业务会话的"上下文窗口"存储；
本模块是"登录态"。两者同名不同物，调用方不要混用。
"""

from __future__ import annotations

import hashlib
import secrets
import time

from ..db import connect, init_db, now_ts

SESSION_IDLE_TTL = 7 * 24 * 3600      # 空闲 7 天
SESSION_ABSOLUTE_TTL = 30 * 24 * 3600  # 绝对 30 天
_RENEW_FRACTION = 0.5                  # 剩余不足一半时滑动续期


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(user_id: int, *, now: float | None = None) -> str:
    """签发登录态，返回明文 token（只应进 Cookie，不进日志/数据库）。"""
    init_db()
    ts = now_ts() if now is None else now
    token = secrets.token_urlsafe(32)
    with connect() as conn:
        conn.execute(
            "INSERT INTO auth_sessions(token_hash, user_id, created_at, expires_at)"
            " VALUES(?,?,?,?)",
            (_hash_token(token), user_id, ts, ts + SESSION_IDLE_TTL),
        )
        conn.commit()
    return token


def validate_token(token: str, *, now: float | None = None) -> int | None:
    """校验并滑动续期。返回 user_id；无效/过期一律 None（不区分原因）。"""
    if not token:
        return None
    init_db()
    ts = now_ts() if now is None else now
    digest = _hash_token(token)
    with connect() as conn:
        row = conn.execute(
            "SELECT user_id, created_at, expires_at FROM auth_sessions"
            " WHERE token_hash=?", (digest,),
        ).fetchone()
        if row is None:
            return None
        if ts >= row["expires_at"] or ts >= row["created_at"] + SESSION_ABSOLUTE_TTL:
            conn.execute("DELETE FROM auth_sessions WHERE token_hash=?", (digest,))
            conn.commit()
            return None
        # 滑动续期：剩余不足一半时延长到 ts + 空闲 TTL
        if row["expires_at"] - ts < SESSION_IDLE_TTL * _RENEW_FRACTION:
            conn.execute(
                "UPDATE auth_sessions SET expires_at=? WHERE token_hash=?",
                (ts + SESSION_IDLE_TTL, digest),
            )
            conn.commit()
        return int(row["user_id"])


def revoke_session(token: str) -> None:
    """登出：删除登录态（幂等）。"""
    if not token:
        return
    with connect() as conn:
        conn.execute(
            "DELETE FROM auth_sessions WHERE token_hash=?", (_hash_token(token),),
        )
        conn.commit()


def cleanup_expired(*, now: float | None = None) -> int:
    """惰性清理过期与超绝对上限的登录态。返回删除行数。"""
    init_db()
    ts = now_ts() if now is None else now
    with connect() as conn:
        cur = conn.execute(
            "DELETE FROM auth_sessions WHERE expires_at<?"
            " OR created_at + ? < ?",
            (ts, SESSION_ABSOLUTE_TTL, ts),
        )
        conn.commit()
        return cur.rowcount


__all__ = [
    "create_session",
    "validate_token",
    "revoke_session",
    "cleanup_expired",
    "SESSION_IDLE_TTL",
    "SESSION_ABSOLUTE_TTL",
]
