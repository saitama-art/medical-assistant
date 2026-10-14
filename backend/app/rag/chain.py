# rag/chain.py
import os

def get_llm():
    from langchain_openai import ChatOpenAI
    import httpx
    http_client = httpx.Client(trust_env=False, timeout=35)
    llm = ChatOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
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


def get_rag_chain():
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate

    # 这里必须显式调用，确保 llm 和 embeddings 不为 None
    llm = get_llm()
    embeddings = get_embedding()
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    medical_prompt = ChatPromptTemplate.from_messages([
        ("system", """
你是医疗知识库问答助手，请严格依据提供的参考文档回答用户问题。
1.只使用上下文给出的资料作答，不要编造没有的医学信息。
2.如果上下文没有相关内容，直接告知：知识库暂无该问题相关资料，请咨询专业医务人员。
3.输出回答通俗易懂，条理清晰。
4.重要免责提示：本内容仅供科普参考，不能替代医生诊断，身体不适请及时线下就医。

参考上下文：
{context}
"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # 提取文档来源的函数
    def extract_sources(docs):
        sources = []
        for doc in docs:
            try:
                source_name = doc.metadata.get('source', '未知来源')
                try:
                    page = doc.metadata.get('page', 0) + 1
                    full_source = f"{source_name} - 第{page}页"
                except:
                    full_source = source_name

                if full_source not in sources:
                    sources.append(full_source)
            except Exception as e:
                print(f"提取来源失败: {e}")
                continue
        return sources

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