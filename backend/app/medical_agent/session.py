"""会话上下文存储：按 session_id 存取最近 N 轮、会话级并发串行、超限与多 worker 降级。

V1.0 运行边界（设计文档 3.5）：

    - 单进程、单 worker。启动时读 WEB_CONCURRENCY，> 1 时记录告警并把会话功能
      整体降级为无历史模式（degraded_no_history），不提供"看起来有上下文、
      实际片段化"的历史。
    - 一轮 = 1 问 1 答 = 2 条消息；MAX_CONTEXT_TURNS = 5 ⇒ deque(maxlen=10)。
    - 超出 maxlen 后继续写入 ⇒ 最早的消息被丢弃，置 truncated 标记；此后读取
      方拿到 context_status = missing，由编排器走"请重述"分支，不静默。
    - 服务重启即内存清空；无法区分"已知 session_id"与"新会话"，统一按新会话处理。
    - 锁粒度 = session_id：acquire 带超时（2s），超时降级为 degraded_no_history，
      照常回答但由编排器明确告知"本次未使用历史对话"。
    - 不引入 Redis；后续要替换存储只改本文件。

C-6 可选持久化（2026-10-20 追加）：设置环境变量 SESSION_PERSIST_DIR 后，会话写回时
把该会话的消息落盘（每会话一个 JSON 文件），进程重启后同一 session_id 的历史自动
恢复；未设置时保持纯内存行为（V1.0 默认）。文件名只允许 [A-Za-z0-9_-]，非法
session_id 静默跳过持久化（功能不受影响，内存上下文照常工作）。

多用户扩展（2026-10-23，登录与用户隔离第二阶段）：引入"归属 owner"（请求作用域
user_id，经 auth.deps.current_owner 注入，未接入认证时为 None = 旧行为完全不变）：
owner 非空时持久化文件存 {persist_dir}/{owner}/{sid}.json，按用户子目录隔离，
避免跨用户文件名冲突与误删；内存条目记录 owner，归属不匹配时宁可无历史/拒绝写入
也不泄露他人上下文（纵深防御，正常路径 API 层已完成归属校验）。drop_conversation
供删除会话时级联清理内存条目与持久化文件。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .state import ContextStatus

# ==================== 配置（设计文档 5.8，最终收敛 config.py） ====================

MAX_CONTEXT_TURNS = 5
_MAX_MESSAGES = 2 * MAX_CONTEXT_TURNS
SESSION_LOCK_TIMEOUT = 2.0
SESSION_IDLE_TTL = 30 * 60

# C-6：会话文件持久化目录（None/空 = 关闭）。读 env 放在调用时，便于测试切换。
_PERSIST_ENV = "SESSION_PERSIST_DIR"
_SAFE_SID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _persist_dir() -> Path | None:
    raw = os.environ.get(_PERSIST_ENV, "").strip()
    if not raw:
        return None
    try:
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as exc:
        print(f"[session] 持久化目录不可用，按内存模式运行: {type(exc).__name__}")
        return None


# ==================== 多 worker 自检 ====================


_MULTI_WORKER = False


def _check_web_concurrency() -> None:
    global _MULTI_WORKER
    try:
        workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
    except ValueError:
        workers = 1
    if workers > 1:
        _MULTI_WORKER = True
        print(
            f"[session] 启动自检：WEB_CONCURRENCY={workers} > 1，"
            "会话功能降级为无历史模式（单进程边界，见设计文档 3.5）"
        )


_check_web_concurrency()


# ==================== 存储 ====================


@dataclass
class _SessionEntry:
    messages: deque = field(default_factory=lambda: deque(maxlen=_MAX_MESSAGES))
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_active: float = field(default_factory=time.monotonic)
    truncated: bool = False
    owner: int | None = None  # 归属 user_id；None = 未接入认证（旧行为）


_SESSIONS: dict[str, _SessionEntry] = {}
_GUARD = threading.Lock()


def _current_owner() -> int | None:
    """请求作用域归属（auth.deps.current_owner）。未接入认证/直接单测时为 None。"""
    try:
        from ..auth import deps as _auth_deps
        return _auth_deps.current_owner.get()
    except Exception:  # noqa: BLE001 —— 认证模块不可用时按无归属降级
        return None


def _owner_mismatch(entry: _SessionEntry, owner: int | None) -> bool:
    return (entry.owner is not None and owner is not None
            and entry.owner != owner)


# ---- C-6 文件持久化 ----

def _sid_safe(session_id: str) -> bool:
    return bool(_SAFE_SID.fullmatch(session_id or ""))


def _session_file(persist_dir: Path, session_id: str,
                  owner: int | None = None) -> Path:
    """owner 非空按用户子目录存放（多用户隔离）；owner 为空维持扁平结构（兼容旧文件）。"""
    name = f"{session_id}.json"
    if owner is None:
        return persist_dir / name
    return persist_dir / str(owner) / name


def _load_from_disk(session_id: str, owner: int | None = None) -> _SessionEntry | None:
    """从持久化文件恢复会话；文件损坏/格式非法时返回 None（按新会话处理）。

    查找顺序：owner 子目录 → 扁平目录（兼容改造前的旧文件）。
    """
    persist_dir = _persist_dir()
    if persist_dir is None or not _sid_safe(session_id):
        return None
    candidates = []
    if owner is not None:
        candidates.append(_session_file(persist_dir, session_id, owner))
    candidates.append(_session_file(persist_dir, session_id, None))
    raw = None
    for path in candidates:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            break
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            return None
    if raw is None:
        return None
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return None
    entry = _SessionEntry(owner=owner)
    for m in messages[-_MAX_MESSAGES:]:
        if (isinstance(m, list) and len(m) == 2
                and m[0] in ("user", "assistant")):
            entry.messages.append((m[0], str(m[1])))
    entry.truncated = bool(raw.get("truncated")) or len(messages) > _MAX_MESSAGES
    return entry if len(entry.messages) > 0 else None


def _save_to_disk(session_id: str, entry: _SessionEntry) -> None:
    """把会话消息整体写盘（先写临时文件再替换，避免半截文件）。

    目录按 entry.owner 决定（owner 非空 → 用户子目录）。
    """
    persist_dir = _persist_dir()
    if persist_dir is None or not _sid_safe(session_id):
        return
    owner = entry.owner
    path = _session_file(persist_dir, session_id, owner)
    tmp = persist_dir / f".{session_id}.tmp"
    if owner is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.parent / f".{session_id}.tmp"
        except OSError as exc:
            print(f"[session] 会话持久化目录不可用（不影响响应）: {type(exc).__name__}")
            return
    payload = {
        "messages": [[r, c] for r, c in entry.messages],
        "truncated": entry.truncated,
    }
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[session] 会话持久化失败（不影响响应）: {type(exc).__name__}")


def drop_conversation(session_id: str, owner: int | None = None) -> None:
    """级联删除：清内存条目 + 删持久化文件（owner 子目录与扁平两处都尝试）。

    归属不匹配时不动（防御误删他人数据）。删除与在途请求的重叠竞争接受
    "在途写回可能重建文件"的残余窗口——软删后的会话 id 不会被复用，
    残余文件只会被 owner 本人经 claim_or_verify 复活时读取，无泄露面。
    """
    if not _sid_safe(session_id):
        return
    with _GUARD:
        entry = _SESSIONS.pop(session_id, None)
    if entry is not None and _owner_mismatch(entry, owner):
        with _GUARD:
            _SESSIONS.setdefault(session_id, entry)
        return
    persist_dir = _persist_dir()
    if persist_dir is None:
        return
    targets = [_session_file(persist_dir, session_id, None)]
    if owner is not None:
        targets.insert(0, _session_file(persist_dir, session_id, owner))
    for path in targets:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _prune_locked() -> None:
    now = time.monotonic()
    stale = [sid for sid, e in _SESSIONS.items() if now - e.last_active > SESSION_IDLE_TTL]
    for sid in stale:
        del _SESSIONS[sid]


def _entry(session_id: str, owner: int | None = None) -> _SessionEntry:
    with _GUARD:
        _prune_locked()
        entry = _SESSIONS.get(session_id)
        if entry is None:
            entry = _load_from_disk(session_id, owner) or _SessionEntry()
            if owner is not None and entry.owner is None:
                entry.owner = owner  # 旧文件无归属字段：由首个合法请求认领
            _SESSIONS[session_id] = entry
        elif owner is not None and entry.owner is None:
            entry.owner = owner
        return entry


# ==================== 对外接口 ====================


def acquire(session_id: str) -> tuple[bool, list[tuple[str, str]], ContextStatus]:
    """加会话锁并取出历史。

    Returns:
        (got_lock, history, context_status)
            got_lock=False        锁等待超时或多 worker 降级，未持有锁，
                                  history 为空，status = degraded_no_history；
            context_status=missing 历史已发生截断（最早的消息已丢弃），
                                  追问指代由编排器走"请重述"分支。
    """
    if _MULTI_WORKER:
        return False, [], ContextStatus.DEGRADED_NO_HISTORY

    owner = _current_owner()
    entry = _entry(session_id, owner)
    got_lock = entry.lock.acquire(timeout=SESSION_LOCK_TIMEOUT)
    if not got_lock:
        # 同一会话有另一请求正在处理；不阻塞等待，降级回答
        return False, [], ContextStatus.DEGRADED_NO_HISTORY

    if _owner_mismatch(entry, owner):
        # 纵深防御：正常路径 API 层已校验归属；走到这里宁可无历史也不泄露
        print(f"[session] owner 不匹配，按无历史处理: {session_id}")
        return True, [], ContextStatus.NOT_NEEDED

    entry.last_active = time.monotonic()
    history = list(entry.messages)
    if entry.truncated:
        return True, history, ContextStatus.MISSING
    if not history:
        return True, [], ContextStatus.NOT_NEEDED
    return True, history, ContextStatus.OK


def complete(session_id: str, question: str, answer: str, *, got_lock: bool) -> None:
    """写回本轮问答并释放锁。日志/会话写入失败不影响响应（设计文档 N12）。

    got_lock=True 而本地无条目（如锁由其他途径持有）时按契约补建条目，
    消息不得静默丢失。归属不匹配的写入被拒绝（只记日志，不影响响应）。
    """
    if _MULTI_WORKER or not got_lock:
        return
    owner = _current_owner()
    entry = _SESSIONS.get(session_id)
    if entry is None:
        entry = _entry(session_id, owner)
    if _owner_mismatch(entry, owner):
        print(f"[session] owner 不匹配，拒绝写入: {session_id}")
        try:
            entry.lock.release()
        except RuntimeError:
            pass
        return
    try:
        if len(entry.messages) == entry.messages.maxlen:
            # 继续写入必然挤掉最早的一轮：此后上下文不完整，必须显式标记
            entry.truncated = True
        entry.messages.append(("user", question))
        entry.messages.append(("assistant", answer))
        entry.last_active = time.monotonic()
        _save_to_disk(session_id, entry)
    except Exception as exc:  # noqa: BLE001 —— 收尾环节，不允许影响响应
        print(f"[session] 会话写入失败（不影响响应）: {type(exc).__name__}")
    finally:
        try:
            entry.lock.release()
        except RuntimeError:
            pass


__all__ = [
    "MAX_CONTEXT_TURNS",
    "SESSION_LOCK_TIMEOUT",
    "SESSION_IDLE_TTL",
    "acquire",
    "complete",
    "drop_conversation",
]
