"""业务会话路由：创建 / 列表 / 历史 / 删除。

所有资源级接口的归属校验在 conversations.py 内完成（统一 403，不泄露
资源存在性）；本层负责认证依赖与 CSRF 防线。读接口（GET）不需要 CSRF 头，
变更类接口（POST/DELETE）必须带 X-Requested-With。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from .. import conversations
from ..auth import deps

router = APIRouter(tags=["conversations"])

_TITLE_MAX = 100
_IMPORT_MAX_ITEMS = 200          # 单次最多导入会话数
_IMPORT_MSGS_PER_CONV = 500      # 单会话最多导入消息数
_IMPORT_MSG_MAX = 10_000         # 单条消息最长字符数


class CreateConversationRequest(BaseModel):
    title: str = "新的对话"


class ImportRequest(BaseModel):
    conversations: list  # [{"title", "created_at", "messages":[{"role","content"}]}]


@router.post("/conversations")
def create_conversation(body: CreateConversationRequest, request: Request,
                        user=Depends(deps.get_current_user)):
    deps.csrf_protect(request)
    title = (body.title or "新的对话").strip()[:_TITLE_MAX] or "新的对话"
    conv_id = conversations.create_conversation(int(user["id"]), title)
    return {"id": conv_id, "title": title}


@router.post("/conversations/import")
def import_history(body: ImportRequest, request: Request,
                   user=Depends(deps.get_current_user)):
    """一次性导入历史会话（localStorage 迁移，决策 #10）。

    批量上限与单条长度做了封顶，防止凭证被盗后借此灌库。
    """
    deps.csrf_protect(request)
    items = [it for it in (body.conversations or []) if isinstance(it, dict)]
    items = items[:_IMPORT_MAX_ITEMS]
    for it in items:
        msgs = it.get("messages") or []
        if isinstance(msgs, list):
            it["messages"] = [
                {**m, "content": str(m.get("content", ""))[:_IMPORT_MSG_MAX]}
                for m in msgs if isinstance(m, dict)
            ][: _IMPORT_MSGS_PER_CONV]
    imported = conversations.import_conversations(int(user["id"]), items)
    return {"imported": imported}


@router.get("/conversations")
def list_conversations(user=Depends(deps.get_current_user)):
    return {"conversations": conversations.list_conversations(int(user["id"]))}


@router.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: str, user=Depends(deps.get_current_user)):
    return {"messages": conversations.list_messages(conv_id, int(user["id"]))}


@router.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str, request: Request,
                        user=Depends(deps.get_current_user)):
    deps.csrf_protect(request)
    conversations.delete_conversation(conv_id, int(user["id"]))
    return {"ok": True}


__all__ = ["router"]
