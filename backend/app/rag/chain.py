# rag/chain.py
import os
import threading
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


def get_llm(timeout: float = 35):
    """返回 LLM 客户端。timeout 为 httpx 原生超时（秒），真正生效、会中止连接。

    默认值 35s 保持不变，不影响现有 /chat/stream；主 Agent 各节点按设计文档
    5.6 的环节预算显式传入（分配到的环节预算 = 该次调用的超时值）。
    """
    from langchain_openai import ChatOpenAI
    import httpx
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY。请在项目根目录或 backend/ 下创建 .env 并写入："
            "DEEPSEEK_API_KEY=sk-xxxxxxxx"
        )
    http_client = httpx.Client(trust_env=False, timeout=timeout)
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        http_client=http_client,
        # temperature=0：判定/分类必须确定性——实测 0.1 时同一证据的证据判定
        # 会 flip（同一题两次判定一次通过一次拒绝），医疗门禁不允许掷硬币。
        # max_tokens=2048：512 会把生成回答截断在长答案中段（判定引文也留足余量）。
        temperature=0.0,
        max_tokens=2048
    )
    return llm


# embedding 与 vectorstore 的进程内单例（惰性构造，见 get_embedding 的说明）
_embedding = None
_embedding_lock = threading.Lock()
_vector_store = None
_vector_store_lock = threading.Lock()


def get_embedding():
    """返回**进程内共享**的 embedding 客户端（首次调用时构造，之后一律复用）。

    为什么必须缓存（实测取证，2026-10-16）：
    设计文档 5.6 规定「embedding 模型与 vectorstore 在**启动时预热**，避免首次
    请求承担模型加载耗时（可达数十秒），这部分耗时**不计入请求预算**」，并以此
    作为「检索不引入线程池、只靠异常捕获 + 整体预算」的前提。原实现每次调用都
    新建 `HuggingFaceEmbeddings`（其构造函数即加载模型权重），导致：

        启动预热带来的是一次性开销，请求路径完全不受益（预热对象被丢弃）；
        实测 N07 单次检索 9.7~15.6s，而 N07 的配置预算是 5s（设计 5.6 预算表）；
        实测整体请求 50.1s > `REQUEST_DEADLINE` 45s —— 门禁 G-3 不通过。

    缓存为单例后模型只加载一次，且这次加载发生在启动预热阶段，N07 的真实成本
    回落到「查一次向量库」，门禁 G-3 的前提（设计 5.6：p95 < 1s）才成立。

    实现约束：
    - **保持惰性构造**：本模块「导入时不依赖 langchain」这一性质不能破坏，
      main.py 在缺 langchain_chroma 时仍要能启动（见 get_medical_prompt 的同类说明）。
    - **加锁**：`/agent/chat` 是同步端点，FastAPI 在线程池中执行，多个首个请求
      可能并发到达；双重检查避免重复加载。构造失败时不写缓存，下次调用重试。
    """
    global _embedding
    if _embedding is None:
        with _embedding_lock:
            if _embedding is None:
                from langchain_huggingface import HuggingFaceEmbeddings

                embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-zh",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
                _embedding = embeddings
    return _embedding


# 单独的向量库获取函数，方便 main.py 上传时调用
def get_vector_store():
    """返回**进程内共享**的 vectorstore（首次调用时构造，之后一律复用）。

    缓存理由同 get_embedding()：原实现每次调用都新建 Chroma + 新建 embedding
    客户端，使设计 5.6 的启动预热对请求路径失效。

    这一改动同时消除了 `/upload` 与 `/agent/chat` 之间的实例不一致：两者现在
    通过**同一个** vectorstore 实例读写，上传的文档对后续问答立即可见，不再依赖
    chromadb 内部 client 复用行为（main.py:108 的 /upload 也走本函数）。
    """
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                from langchain_chroma import Chroma

                vector_store = Chroma(
                    persist_directory=r"F:\medical-assistant\backend\chroma_db",
                    embedding_function=get_embedding()
                )
                _vector_store = vector_store
    return _vector_store


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