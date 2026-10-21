from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
# 从 chain.py 导入正确的函数名：get_vector_store
from app.rag import get_rag_chain, get_llm, get_vector_store

app = FastAPI(title="医疗知识库问答")


@app.on_event("startup")
def startup_checks():
    """启动自检（设计文档 Step 4 / 5.6）：

    1. 证据判定能力自检——两段判定都不可用时直接报错，禁止无判定能力地提供服务；
    2. 向量库/embedding 预热——一次 warm-up 检索，避免首次请求承担模型加载耗时。
    """
    from app.rag import service as rag_service
    from app.rag.chain import get_vector_store

    rag_service.assert_evidence_judge_available()  # 失败即拒绝启动（5.4 fail-safe）

    try:
        vector_store = get_vector_store()
        vector_store.similarity_search_with_score("健康", k=1)
        print("[startup] 向量库预热完成")
    except Exception as exc:
        # 预热失败不阻断启动：首个请求会走 RETRIEVAL_ERROR 分支并给出固定文案
        print(f"[startup] 向量库预热失败（不阻断启动）: {type(exc).__name__}")


class AgentChatRequest(BaseModel):
    session_id: str
    question: str


@app.post("/agent/chat")
def agent_chat(req: AgentChatRequest):
    """主 Agent 问答接口（F4 编排链路）。同步端点，FastAPI 会在线程池执行，
    不阻塞事件循环；原有 /chat/stream、/upload 行为不变。"""
    from app.medical_agent import orchestrator
    from app.medical_agent import respond

    try:
        state = orchestrator.run_agent(req.session_id, req.question)
    except orchestrator.AgentRequestError:
        # N00 参数缺失 → 400，不进入编排
        raise HTTPException(status_code=400, detail=respond.INVALID_ARGS_REPLY)
    except Exception:
        # 编排器承诺不抛异常；这里是双保险，不向用户泄露内部信息
        raise HTTPException(status_code=500, detail=respond.SERVICE_UNAVAILABLE)

    return {
        "request_id": state.request_id,
        "answer": state.answer_text,
        "sources": orchestrator.display_sources(state),
        "category": state.category.value if state.category else None,
        "outcome": state.termination_reason.value if state.termination_reason else None,
        # WEAK 急症风险提示原文（emergency_handled 时为空，其文案已在 answer 内）。
        # 前端用它渲染安全提醒横幅；answer 内的文字保持不变（向后兼容）。
        "risk_notice": state.risk_notice or None,
    }

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置上传文件保存的目录
UPLOAD_DIR = "../../uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


# ================= 上传 PDF 接口 =================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="只允许上传 PDF 文件")

    # 1. 保存文件到本地
    file_path = os.path.join(UPLOAD_DIR, file.filename)
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

        return {"message": f"文件 {file.filename} 上传成功，并已成功入库！"}

    except Exception as e:
        # 捕获解析或入库的异常，返回给前端
        return {"message": f"文件 {file.filename} 已上传，但知识库处理失败: {str(e)}"}


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