# rag/chain.py
import os
from pathlib import Path

from dotenv import load_dotenv

# 从本文件所在目录逐级向上查找 .env，同时兼容两种放置方式：
#   项目根目录/.env（本机现状）与 backend/.env（README 第 4 节的写法）。
_env_file = next(
    (p / ".env" for p in Path(__file__).resolve().parents if (p / ".env").is_file()),
    None,
)
# override=True：项目 .env 优先于系统环境变量。
# 原因：本机系统环境变量里的 DEEPSEEK_API_KEY 是缺了 sk- 前缀的坏值（实测 401），
# 而 .env 里是有效值（实测 200）。不加 override 就改不动，读到的仍是坏值。
if _env_file is not None:
    load_dotenv(_env_file, override=True)


def get_llm():
    from langchain_openai import ChatOpenAI
    import httpx
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY。请在项目根目录或 backend/ 下创建 .env 并写入："
            "DEEPSEEK_API_KEY=sk-xxxxxxxx"
        )
    http_client = httpx.Client(trust_env=False, timeout=35)
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        http_client=http_client,
        temperature=0.1,
        max_tokens=512
    )
    return llm


def get_embedding():
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


# 单独的向量库获取函数，方便 main.py 上传时调用
def get_vector_store():
    from langchain_chroma import Chroma
    embeddings = get_embedding()

    vector_store = Chroma(
        persist_directory=r"F:\medical-assistant\backend\chroma_db",
        embedding_function=embeddings
    )
    return vector_store


# ==================== 医疗问答 Prompt（唯一定义处） ====================
# 下面的 system 文案是生成环节的唯一事实约束（只用上下文作答 + 免责声明）。
# 定义在模块级是为了让 rag/service.py 能复用同一份 Prompt，
# 避免出现第二份副本之后两处措辞不同步。不要在别处重新写一遍。

MEDICAL_SYSTEM_PROMPT = """
你是医疗知识库问答助手，请严格依据提供的参考文档回答用户问题。
1.只使用上下文给出的资料作答，不要编造没有的医学信息。
2.如果上下文没有相关内容，直接告知：知识库暂无该问题相关资料，请咨询专业医务人员。
3.输出回答通俗易懂，条理清晰。
4.重要免责提示：本内容仅供科普参考，不能替代医生诊断，身体不适请及时线下就医。

参考上下文：
{context}
"""

MEDICAL_HUMAN_PROMPT = "{question}"

_medical_prompt = None


def get_medical_prompt():
    """返回医疗问答 Prompt 模板（首次调用时构造并缓存）。

    惰性构造是为了保持本模块「导入时不依赖 langchain」这一既有性质：
    main.py 目前在 langchain_chroma 未安装时仍能启动，靠的正是这一性质。
    """
    global _medical_prompt
    if _medical_prompt is None:
        from langchain_core.prompts import ChatPromptTemplate

        _medical_prompt = ChatPromptTemplate.from_messages([
            ("system", MEDICAL_SYSTEM_PROMPT),
            ("human", MEDICAL_HUMAN_PROMPT),
        ])
    return _medical_prompt


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def extract_sources(docs):
    """从 Document.metadata 提取展示用的「文档名 - 第N页」字符串。

    仅供 /chat/stream 使用（返回字符串列表）。主 Agent 请用 rag/service.py 的
    extract_sources()：那里返回结构化 Source，且元数据缺 page 时不推测页码。
    """
    sources = []
    for doc in docs:
        try:
            source_name = doc.metadata.get('source', '未知来源')
            try:
                page = doc.metadata.get('page', 0) + 1
                full_source = f"{source_name} - 第{page}页"
            except Exception:
                full_source = source_name

            if full_source not in sources:
                sources.append(full_source)
        except Exception as e:
            print(f"提取来源失败: {e}")
            continue
    return sources


def get_rag_chain():
    from langchain_chroma import Chroma

    # 这里必须显式调用，确保 llm 和 embeddings 不为 None
    llm = get_llm()
    embeddings = get_embedding()
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    medical_prompt = get_medical_prompt()

    # 自定义的函数来同时返回答案和来源
    def rag_with_sources(input_dict):
        question = input_dict["question"]

        # 1. 检索文档
        docs = retriever.invoke(question)
        context = format_docs(docs)
        sources = extract_sources(docs)
        # source_text = "\n\n---\n\n**📄 参考来源：**\n" + "\n".join([f"- {s}" for s in sources])
        # 直接把 \n 替换成 <br/>，因为前端 ReactMarkdown 可能会吞掉 \n
        # 用分号(；)连接，前端自然换行
        # 改为纯文本格式，避免 Markdown 解析错误
        source_text = "\n\n【参考来源】：" + "；".join(sources)        # 2. 构造 Prompt
        formatted_prompt = medical_prompt.format_messages(context=context, question=question)

        # 3. 调用 LLM（确保 llm 不是 None）
        response = llm.invoke(formatted_prompt)

        # 4. 返回包含答案和来源的字典
        return {
            "answer": response.content +source_text,
            "sources": sources
        }

    return rag_with_sources