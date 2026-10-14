from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
# 从 chain.py 导入正确的函数名：get_vector_store
from app.rag import get_rag_chain, get_llm, get_vector_store

app = FastAPI(title="医疗知识库问答")

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
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # 读取 PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"成功读取 PDF，共 {len(documents)} 页。")

        # 切分文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
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