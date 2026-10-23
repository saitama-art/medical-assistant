"""FastAPI 认证依赖、CSRF/Origin 防线与登录限流。

分工（审查报告 §2/§3）：

- ``get_current_user``：唯一可信身份来源——从 HttpOnly Cookie 解析 token，
  经 auth/sessions 校验后返回 users 行。任何接口不得从 body/参数取用户身份。
- ``current_owner``（contextvars）：请求作用域内的 user_id 上下文，供
  session.py 做纵深防御（owner 注入，见设计方案 §5.1）；未设置时为 None，
  旧行为完全不变。
- ``LoginRateLimiter``：内存双层计数（username+IP 锁定 / IP 总频次），
  单 worker 边界内有效；锁定期满自动解除，成功登录清零，不存在永久锁定。
- CSRF：变更类请求必须带 ``X-Requested-With``；Origin 存在时必须在白名单。
  两者由 ``csrf_protect`` 依赖在受保护路由上强制执行。
"""

from __future__ import annotations

import contextvars
import threading
import time

from fastapi import Cookie, HTTPException, Request, status

from . import sessions

COOKIE_NAME = "med_session"

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="请先登录后再操作。",
)
_FORBIDDEN_CSRF = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="请求被拒绝。",
)

# ---- 请求作用域的用户上下文（session.py 纵深防御用） ----

current_owner: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "current_owner", default=None,
)


def get_current_user(request: Request,
                     med_session: str | None = Cookie(default=None)):
    """认证依赖：无有效登录态一律 401（不区分"未登录/凭证坏/已过期"）。"""
    user_id = sessions.validate_token(med_session)
    if user_id is None:
        raise _UNAUTHORIZED
    from .users import get_by_id
    user = get_by_id(user_id)
    if user is None:
        raise _UNAUTHORIZED
    # 注入请求作用域 owner：后续 session.py 持久化按用户目录存放/校验。
    current_owner.set(int(user["id"]))
    request.state.user = user
    return user


def csrf_protect(request: Request):
    """变更类请求的 CSRF 防线：自定义头 + Origin 白名单（见模块 docstring）。"""
    if request.headers.get("X-Requested-With") != "XMLHttpRequest":
        raise _FORBIDDEN_CSRF
    origin = request.headers.get("Origin")
    if origin is not None and origin not in _allowed_origins():
        raise _FORBIDDEN_CSRF
    return True


def _allowed_origins() -> set[str]:
    import os
    raw = os.environ.get("CORS_ORIGINS",
                         "http://localhost:3000,http://127.0.0.1:3000")
    return {o.strip() for o in raw.split(",") if o.strip()}


# ---- 登录限流（内存实现，单 worker 边界） ----

class LoginRateLimiter:
    """双层计数：username+IP 连续失败锁定；IP 总尝试频次封顶。

    时间经 clock 注入，测试确定性。锁定窗口到点自动解除——
    避免"知道用户名就能让对方长期锁死"（审查报告 §2）。
    """

    def __init__(self, *, max_fails: int = 5, lock_seconds: float = 15 * 60,
                 ip_per_minute: int = 30, clock=None):
        self.max_fails = max_fails
        self.lock_seconds = lock_seconds
        self.ip_per_minute = ip_per_minute
        self._clock = clock or time.monotonic
        self._guard = threading.Lock()
        self._fails: dict[tuple[str, str], list[float]] = {}
        self._ip_hits: dict[str, list[float]] = {}

    def _prune(self, ts: float) -> None:
        window = self.lock_seconds
        self._fails = {k: [t for t in v if ts - t < window]
                       for k, v in self._fails.items() if any(ts - t < window for t in v)}
        self._ip_hits = {k: [t for t in v if ts - t < 60]
                         for k, v in self._ip_hits.items() if any(ts - t < 60 for t in v)}

    def is_locked(self, username: str, ip: str) -> bool:
        with self._guard:
            ts = self._clock()
            self._prune(ts)
            attempts = self._fails.get((username.lower(), ip), [])
            return (len(attempts) >= self.max_fails
                    and ts - attempts[0] < self.lock_seconds)

    def check_ip_budget(self, ip: str) -> bool:
        """IP 分钟窗口内仍有额度返回 True。"""
        with self._guard:
            ts = self._clock()
            self._prune(ts)
            return len(self._ip_hits.get(ip, [])) < self.ip_per_minute

    def record_fail(self, username: str, ip: str) -> None:
        with self._guard:
            ts = self._clock()
            self._fails.setdefault((username.lower(), ip), []).append(ts)
            self._ip_hits.setdefault(ip, []).append(ts)

    def record_success(self, username: str, ip: str) -> None:
        with self._guard:
            self._fails.pop((username.lower(), ip), None)


# 模块级单例：登录路由共用（单 worker 边界内即全局有效）。
login_limiter = LoginRateLimiter()


__all__ = [
    "COOKIE_NAME",
    "current_owner",
    "get_current_user",
    "csrf_protect",
    "LoginRateLimiter",
    "login_limiter",
]
