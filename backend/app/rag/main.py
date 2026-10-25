from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import shutil
import uuid
from pathlib import Path
# 从 chain.py 导入正确的函数名：get_vector_store
from app.rag import get_rag_chain, get_llm, get_vector_store

app = FastAPI(title="医疗知识库问答")

# CORS 白名单（登录与用户隔离，2026-10-23）：禁止 "*" + credentials 组合。
# 开发默认放行 React dev server 两个口径；生产用 CORS_ORIGINS 环境变量收紧。
_ALLOWED_ORIGINS = sorted({
    o.strip() for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",") if o.strip()
})

# 上传保存目录：原 "../../uploads" 依赖启动 CWD，改为锚定 backend/ 目录
UPLOAD_DIR = os.environ.get("UPLOAD_DIR") or str(
    Path(__file__).resolve().parents[2] / "uploads"
)


def _check_auth_config() -> None:
    """启动自检（审查报告 §3）：配置错误必须 fail-fast，不得带病服务。"""
    from app.api.auth_routes import cookie_samesite, cookie_secure
    if "*" in _ALLOWED_ORIGINS:
        raise RuntimeError(
            "CORS_ORIGINS 含 '*'：凭证模式下不允许通配源，请配置具体来源。"
        )
    if cookie_samesite().lower() == "none" and not cookie_secure():
        raise RuntimeError(
            "SameSite=None 的 Cookie 必须带 Secure，请修正 SESSION_COOKIE_* 配置。"
        )


@app.on_event("startup")
def startup_checks():
    """启动自检（设计文档 Step 4 / 5.6）：

    1. 证据判定能力自检——两段判定都不可用时直接报错，禁止无判定能力地提供服务；
    2. 向量库/embedding 预热——一次 warm-up 检索，避免首次请求承担模型加载耗时；
    3. 认证配置与数据库自检（2026-10-23）：CORS/Cookie 配置 fail-fast + 幂等建表
       + 清理过期登录态。
    """
    from app.rag import service as rag_service
    from app.rag.chain import get_vector_store

    _check_auth_config()

    from app import db
    db.init_db()
    from app.auth import sessions as auth_sessions
    auth_sessions.cleanup_expired()

    rag_service.assert_evidence_judge_available()  # 失败即拒绝启动（5.4 fail-safe）

    try:
        vector_store = get_vector_store()
        vector_store.similarity_search_with_score("健康", k=1)
        print("[startup] 向量库预热完成")
    except Exception as exc:
        # 预热失败不阻断启动：首个请求会走 RETRIEVAL_ERROR 分支并给出固定文案
        print(f"[startup] 向量库预热失败（不阻断启动）: {type(exc).__name__}")


# ---- 认证与业务会话路由（登录与用户隔离，2026-10-23） ----
from app.api import auth_routes, conversation_routes  # noqa: E402
from app.auth import deps  # noqa: E402

app.include_router(auth_routes.router)
app.include_router(conversation_routes.router)


class AgentChatRequest(BaseModel):
    session_id: str
    question: str


@app.post("/agent/chat")
def agent_chat(req: AgentChatRequest, request: Request,
               user=Depends(deps.get_current_user)):
    """主 Agent 问答接口（F4 编排链路）。同步端点，FastAPI 会在线程池执行，
    不阻塞事件循环；原有 /chat/stream、/upload 行为不变。

    登录与用户隔离（2026-10-23）：须登录；session_id 归属当前用户
    （他人会话 403，未知 id 隐式创建并归属本人）；双写全量历史到 SQLite。
    """
    from app.auth import deps
    from app import conversations
    from app.medical_agent import orchestrator
    from app.medical_agent import respond

    deps.csrf_protect(request)

    # session_id 口径：非空、≤64 字符即接受（历史客户端/验收用例含中文 id；
    # 落盘文件名由 session.py 做哈希兜底，见 _file_stem）。 question 非空。
    if (not req.session_id or not req.session_id.strip()
            or len(req.session_id) > 64 or not req.question.strip()):
        raise HTTPException(status_code=400, detail=respond.INVALID_ARGS_REPLY)

    owner = int(user["id"])
    deps.current_owner.set(owner)  # session.py 按 owner 子目录持久化/校验
    conv_id = conversations.claim_or_verify(req.session_id, owner)

    try:
        state = orchestrator.run_agent(conv_id, req.question)
    except orchestrator.AgentRequestError:
        # N00 参数缺失 → 400，不进入编排
        raise HTTPException(status_code=400, detail=respond.INVALID_ARGS_REPLY)
    except Exception:
        # 编排器承诺不抛异常；这里是双保险，不向用户泄露内部信息
        raise HTTPException(status_code=500, detail=respond.SERVICE_UNAVAILABLE)

    # 全量历史双写（上下文窗口仍由 session.complete 负责）；失败不影响响应
    try:
        conversations.append_turn(conv_id, owner,
                                  req.question, state.answer_text)
    except Exception as exc:  # noqa: BLE001 —— 历史落库失败不应吞掉已生成的回答
        print(f"[chat] 历史落库失败（不影响响应）: {type(exc).__name__}")

    return {
        "request_id": state.request_id,
        "answer": state.answer_text,
        "sources": orchestrator.display_sources(state),
        "category": state.category.value if state.category else None,
        "outcome": state.termination_reason.value if state.termination_reason else None,
        # WEAK 急症风险提示原文（emergency_handled 时为空，其文案已在 answer 内）。
        # 前端用它渲染安全提醒横幅；answer 内的文字保持不变（向后兼容）。
        "risk_notice": state.risk_notice or None,
        # 调试契约（评估体系 S1-D4）：本轮补全后的独立问题；None=未触发
        # （急症等 N02 之前终止的路径）。仅本会话 owner 可见，前端不消费，
        # 不落 SQLite——可观测性字段，供约束保留率等指标取数。
        "resolved_question": state.resolved_question,
    }

# 配置 CORS：白名单 + 凭证（禁止 * + credentials，见 _check_auth_config）
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= 上传 PDF 接口 =================
# 文件名净化白名单（与 session.py 的 sid 白名单同思路：只放行明确安全的字符）
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._\-一-鿿]+")


def _sanitize_filename(name: str | None) -> str:
    """防路径穿越：去目录成分（含 Windows 反斜杠）+ 字符白名单 + 去首尾点。

    返回空串表示不可接受（调用方转 400）。
    """
    if not name:
        return ""
    base = os.path.basename(name.replace("\\", "/"))
    base = _FILENAME_SAFE.sub("_", base).strip("._")
    return base[:80]


@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...),
                     user=Depends(deps.get_current_user)):
    from app.auth import deps

    deps.csrf_protect(request)
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="无上传权限。")
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="只允许上传 PDF 文件")

    safe_name = _sanitize_filename(file.filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="文件名不合法")

    # 1. 保存文件到本地（uuid 前缀杜绝重名覆盖与任何残留路径成分）
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{safe_name}")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # 2. 加载并处理 PDF（核心步骤）
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from .chapter_split import split_documents  # 与离线建库同一套章节切分

        # 读取 PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"成功读取 PDF，共 {len(documents)} 页。")

        # 切分文档：与 ingest 相同的章节切分（800/100），上传文档与建库文档口径一致
        docs = split_documents(documents, chunk_size=800, chunk_overlap=100)
        print(f"切分完成，共 {len(docs)} 个片段。")

        # 写入向量库
        vector_store = get_vector_store()  # 注意：这里的函数名要和 chain.py 里的一致（有下划线）
        vector_store.add_documents(docs)

        # 审计：谁、何时、哪个文件、入库多少（不记录文件内容）
        print(f"[upload] 管理员 {user['username']} 上传 {safe_name}"
              f"（{len(documents)} 页 → {len(docs)} 片段）")
        return {"message": f"文件 {safe_name} 上传成功，并已成功入库！"}

    except Exception as e:
        # 捕获解析或入库的异常，返回给前端
        return {"message": f"文件 {safe_name} 已上传，但知识库处理失败: {str(e)}"}


# ================= 测试接口 =================
# @app.get("/test_llm")

async def test_llm():
    from langchain_core.prompts import ChatPromptTemplate
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([("user", "你好，请用中文回复")])
    chain = prompt | llm
    res = await chain.ainvoke({})
    return {"content": res.content}


# ================= 流式问答接口 =================
async def stream_chat(question: str):
    rag_chain = get_rag_chain()
    try:
        result = rag_chain({"question": question})
        answer = result.get("answer", "暂无回答")
    except Exception as e:
        print(f"RAG 链路发生错误: {e}")
        yield f"data: 系统内部错误，无法生成回答: {str(e)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 逐字输出答案（这里已经包含了来源）
    for char in answer:
        yield f"data: {char}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/chat/stream")
async def chat_stream(question: str):
    return StreamingResponse(
        stream_chat(question),
        media_type="text/event-stream"
    )