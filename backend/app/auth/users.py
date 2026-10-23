"""用户资料存取：创建、按用户名/ID 查询、登录时间戳。

用户名大小写不敏感（COLLATE NOCASE 已在表定义）；密码哈希只经 passwords.py
进出，本模块不出现明文密码的日志或异常文本。
"""

from __future__ import annotations

import sqlite3

from ..db import connect, init_db, now_ts


class UserStoreError(Exception):
    """用户名已存在等写入冲突（接口层转 409/统一话术，不泄露内部细节）。"""


def create_user(username: str, password_hash: str, display_name: str = "",
                is_admin: bool = False, *, now: float | None = None) -> int:
    init_db()
    ts = now_ts() if now is None else now
    try:
        with connect() as conn:
            cur = conn.execute(
                "INSERT INTO users(username, password_hash, display_name, is_admin,"
                " created_at) VALUES(?,?,?,?,?)",
                (username.strip(), password_hash, display_name.strip(),
                 1 if is_admin else 0, ts),
            )
            conn.commit()
            return int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise UserStoreError("username_taken") from exc


def get_by_username(username: str) -> sqlite3.Row | None:
    init_db()
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username=?", (username.strip(),),
        ).fetchone()


def get_by_id(user_id: int) -> sqlite3.Row | None:
    init_db()
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE id=?", (user_id,),
        ).fetchone()


def mark_login(user_id: int, *, now: float | None = None) -> None:
    ts = now_ts() if now is None else now
    with connect() as conn:
        conn.execute(
            "UPDATE users SET last_login_at=? WHERE id=?", (ts, user_id),
        )
        conn.commit()


__all__ = [
    "UserStoreError",
    "create_user",
    "get_by_username",
    "get_by_id",
    "mark_login",
]
