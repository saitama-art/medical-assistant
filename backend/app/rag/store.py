import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

# ← 从 chain.py 导入 LLM
from chain import get_llm


# ==================== 路径 ====================
CHROMA_DIR = r"/chroma_db"
EMBEDDING_PATH = r"C:\Users\樊海涛\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5\snapshots\bge"
RERANKER_PATH = r"F:\models\bge-reranker-base"


# ==================== Embedding ====================
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ==================== 向量库 ====================
os.makedirs(CHROMA_DIR, exist_ok=True)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)


# ==================== Reranker ====================
cross_encoder = HuggingFaceCrossEncoder(
    model_name=RERANKER_PATH,
    model_kwargs={"device": "cpu"},
)
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)


# ==================== LLM（从 chain.py 导入） ====================
llm = get_llm()


# ==================== Query Rewrite ====================
def rewrite_query(original_query: str) -> str:
    """
    用 LLM 把用户问题改写成更适合检索的形式
    """
    prompt = f"""你是一个检索优化助手。请把用户的问题改写成更适合在医疗知识库中检索的形式。
要求：
1. 补全模糊指代（如"这个药"→具体药名，如果能从上下文推断）
2. 使用标准医学术语
3. 只输出改写后的问题，不要解释，不要加引号

用户问题：{original_query}
改写后："""

    try:
        response = llm.invoke(prompt)
        rewritten = response.content.strip()
        return rewritten if rewritten else original_query
    except Exception as e:
        print(f"Query Rewrite 失败: {e}")
        return original_query


# ==================== 对外函数 ====================
def add_documents(docs):
    """批量写入向量库"""
    vectorstore.add_documents(docs)
    vectorstore.persist()


def search(query: str, k: int = 3):
    """纯向量检索（Baseline）"""
    return vectorstore.similarity_search(query, k=k)


def search_with_rerank(query: str, top_k: int = 20, rerank_k: int = 3):
    """向量检索 + Reranker 精排"""
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )
    return compression_retriever.invoke(query)


def search_with_rewrite(query: str, k: int = 3):
    """Query Rewrite + 向量检索"""
    rewritten = rewrite_query(query)
    print(f"  原问题: {query}")
    print(f"  改写后: {rewritten}")
    return vectorstore.similarity_search(rewritten, k=k)


# ==================== 测试 ====================
if __name__ == "__main__":
    vec = embeddings.embed_query("来那度胺的禁忌症")
    print(f"向量维度: {len(vec)}")

    q = "这个药孕妇能吃吗？"

    print("\n【纯向量检索】")
    for d in search(q, k=3):
        print(f"  {os.path.basename(d.metadata.get('source', ''))}")

    print("\n【Query Rewrite + 向量检索】")
    for d in search_with_rewrite(q, k=3):
        print(f"  {os.path.basename(d.metadata.get('source', ''))}")