from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name=r"C:\Users\樊海涛\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5\snapshots\bge",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
# 用绝对路径，避免路径混乱
CHROMA_DIR = r"/chroma_db"

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

# 先确认数据库里有东西
print("总向量数:", vectorstore._collection.count())

questions = [
    "来那度胺的禁忌症是什么？",
    "布洛芬的常见副作用有哪些？",
    "二甲双胍怎么吃？",
    "阿昔洛韦治疗什么病？",
    "氟康唑的适应症是什么？",
]

for q in questions:
    docs = vectorstore.similarity_search(q, k=3)
    print(f"\n问题：{q}")
    if not docs:
        print("  ⚠️ 没有检索到任何内容")
        continue
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] {doc.page_content[:80]}...")
        print(f"      来源: {doc.metadata.get('source', 'unknown')}")