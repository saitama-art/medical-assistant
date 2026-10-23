"""认证路由：登录 / 登出 / 当前用户。

安全要点（技术审查报告 §2/§3）：
- 登录统一响应：用户名不存在与密码错误返回完全相同的 401，防用户枚举；
- 限流：IP 分钟额度 + username+IP 失败锁定（自动解除，见 deps.LoginRateLimiter）；
- 登录也过 CSRF 防线（Origin 校验防登录 CSRF：攻击者把受害者登进攻击者账号）；
- Cookie：HttpOnly + Secure + SameSite 环境可配； SameSite=None 必须 Secure
  （启动自检兜底，见 main.startup_checks）；
- 日志纪律：只记 username 与结果，绝不记密码与 token。
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from ..auth import deps, sessions as auth_sessions, users
from ..auth.passwords import verify_password

router = APIRouter(prefix="/auth", tags=["auth"])

LOGIN_FAIL_REPLY = "用户名或密码错误。"
LOGIN_LOCKED_REPLY = "失败次数过多，账号已临时锁定，请稍后再试。"
LOGIN_THROTTLED_REPLY = "尝试过于频繁，请稍后再试。"


class LoginRequest(BaseModel):
    username: str
    password: str


_DUMMY_HASH: str | None = None


def _dummy_hash() -> str:
    """用户不存在时用于对齐校验耗时的哑哈希（惰性生成一次）。

    使"用户名不存在"与"密码错误"的响应耗时不可区分，堵住计时侧信道。
    """
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        from ..auth.passwords import hash_password
        _DUMMY_HASH = hash_password("timing-equalization-dummy")
    return _DUMMY_HASH


def cookie_secure() -> bool:
    return os.environ.get("SESSION_COOKIE_SECURE", "true").lower() != "false"


def cookie_samesite() -> str:
    return os.environ.get("SESSION_COOKIE_SAMESITE", "none")


def _user_payload(user) -> dict:
    return {
        "id": int(user["id"]),
        "username": user["username"],
        "display_name": user["display_name"],
        "is_admin": bool(user["is_admin"]),
    }


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response):
    deps.csrf_protect(request)  # Origin 白名单，防登录 CSRF
    ip = request.client.host if request.client else "unknown"
    if not deps.login_limiter.check_ip_budget(ip):
        raise HTTPException(status_code=429, detail=LOGIN_THROTTLED_REPLY)
    if deps.login_limiter.is_locked(body.username, ip):
        raise HTTPException(status_code=403, detail=LOGIN_LOCKED_REPLY)

    user = users.get_by_username(body.username)
    # 统一校验路径：用户不存在时也执行一次校验，使两种失败耗时不可区分
    stored_hash = user["password_hash"] if user is not None else _dummy_hash()
    if not verify_password(body.password, stored_hash):
        deps.login_limiter.record_fail(body.username, ip)
        print(f"[auth] 登录失败 username={body.username!r} ip={ip}")
        raise HTTPException(status_code=401, detail=LOGIN_FAIL_REPLY)

    deps.login_limiter.record_success(body.username, ip)
    user_id = int(user["id"])
    token = auth_sessions.create_session(user_id)
    users.mark_login(user_id)
    deps.current_owner.set(user_id)
    response.set_cookie(
        deps.COOKIE_NAME, token,
        max_age=auth_sessions.SESSION_IDLE_TTL,
        httponly=True, secure=cookie_secure(),
        samesite=cookie_samesite(), path="/",
    )
    print(f"[auth] 登录成功 username={body.username!r} ip={ip}")
    return _user_payload(user)


@router.post("/logout")
def logout(request: Request, response: Response,
           med_session: str | None = Cookie(default=None)):
    deps.csrf_protect(request)
    # 登出对过期凭证也要能清 Cookie：吊销尽力而为，不要求登录态有效
    auth_sessions.revoke_session(med_session)
    response.delete_cookie(deps.COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
def me(user=Depends(deps.get_current_user)):
    return _user_payload(user)


__all__ = ["router", "cookie_secure", "cookie_samesite"]
