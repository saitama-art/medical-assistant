import os
import sys
sys.path.append(r"/app/rag")
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from store import search_with_rerank, search_with_rewrite

# ==================== 当前实验参数 ====================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
K_LIST = [3]

print("=" * 70)
print(f"【Eval】chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
print("=" * 70)
print()


# ==================== 路径 ====================
CHROMA_DIR = r"F:\medical-assistant\backend\chroma_db"
MODEL_PATH = r"C:\Users\樊海涛\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5\snapshots\bge"


# ==================== 测试集：28 条 ====================
TEST_CASES = [
    # 明确问题（18 条）
    {"question": "来那度胺的禁忌症是什么？", "source": "国药准字H20193006_来那度胺胶囊.pdf"},
    {"question": "来那度胺会导致什么血液学毒性？", "source": "国药准字H20193006_来那度胺胶囊.pdf"},
    {"question": "孕妇可以服用来那度胺吗？", "source": "国药准字H20193006_来那度胺胶囊.pdf"},
    {"question": "布洛芬的常见副作用有哪些？", "source": "国药准字H20055226_布洛芬颗粒.pdf"},
    {"question": "布洛芬颗粒怎么吃？", "source": "国药准字H20055226_布洛芬颗粒.pdf"},
    {"question": "二甲双胍怎么吃？", "source": "国药准字H20184130_盐酸二甲双胍片.pdf"},
    {"question": "二甲双胍的适应症是什么？", "source": "国药准字H20184130_盐酸二甲双胍片.pdf"},
    {"question": "阿昔洛韦治疗什么病？", "source": "国药准字H10983103_阿昔洛韦片.pdf"},
    {"question": "阿昔洛韦的常见副作用有哪些？", "source": "国药准字H10983103_阿昔洛韦片.pdf"},
    {"question": "氟康唑的适应症是什么？", "source": "国药准字H10970299_氟康唑片.pdf"},
    {"question": "氟康唑怎么吃？", "source": "国药准字H10970299_氟康唑片.pdf"},
    {"question": "格列美脲的副作用有哪些？", "source": "国药准字H20061256_格列美脲片.pdf"},
    {"question": "格列美脲的适应症是什么？", "source": "国药准字H20061256_格列美脲片.pdf"},
    {"question": "多潘立酮的禁忌症是什么？", "source": "国药准字H20123058_多潘立酮片.pdf"},
    {"question": "拉米夫定怎么吃？", "source": "国药准字H20103284_拉米夫定片.pdf"},
    {"question": "依那普利的适应症是什么？", "source": "国药准字H20066383_马来酸依那普利片.pdf"},
    {"question": "氨氯地平的副作用有哪些？", "source": "国药准字H20020390_苯磺酸氨氯地平片.pdf"},
    {"question": "克林霉素的适应症是什么？", "source": "国药准字H13020795_盐酸克林霉素胶囊.pdf"},

    # 模糊问题（3 条）
    {"question": "这个药孕妇能吃吗？", "source": "国药准字H20193006_来那度胺胶囊.pdf"},
    {"question": "它的常见副作用有哪些？", "source": "国药准字H20055226_布洛芬颗粒.pdf"},
    {"question": "这个药一天吃几次？", "source": "国药准字H20184130_盐酸二甲双胍片.pdf"},

    # 同义词问题（3 条）
    {"question": "退烧药怎么吃？", "source": "国药准字H20055226_布洛芬颗粒.pdf"},
    {"question": "止痛药有哪些副作用？", "source": "国药准字H20055226_布洛芬颗粒.pdf"},
    {"question": "血糖高吃什么药？", "source": "国药准字H20184130_盐酸二甲双胍片.pdf"},

    # 跨文档问题（2 条）
    {"question": "降糖药的常见副作用？", "source": "国药准字H20184130_盐酸二甲双胍片.pdf"},
    {"question": "抗生素的适应症有哪些？", "source": "国药准字H13020795_盐酸克林霉素胶囊.pdf"},

    # 否定/排除问题（2 条）
    {"question": "哪些药不能给孕妇吃？", "source": "国药准字H20193006_来那度胺胶囊.pdf"},
    {"question": "哪些药会导致胎儿畸形？", "source": "国药准字H20193006_来那度胺胶囊.pdf"},
]


# ==================== 加载 ====================
print("加载 BGE 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("加载 Chroma...")
vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

chunk_count = vs._collection.count()
print(f"向量库 chunk 数: {chunk_count}")
print()


# ==================== 评估函数 ====================
def evaluate(k=3):
    """纯向量检索"""
    hit = 0
    mrr_sum = 0.0
    for case in TEST_CASES:
        docs = vs.similarity_search(case["question"], k=k)
        sources = [os.path.basename(d.metadata.get("source", "")) for d in docs]
        if case["source"] in sources:
            hit += 1
            mrr_sum += 1.0 / (sources.index(case["source"]) + 1)
    return hit / len(TEST_CASES), mrr_sum / len(TEST_CASES)


def evaluate_with_rerank(k=3):
    """向量检索 + Reranker"""
    hit = 0
    mrr_sum = 0.0
    for case in TEST_CASES:
        docs = search_with_rerank(case["question"], top_k=20, rerank_k=k)
        sources = [os.path.basename(d.metadata.get("source", "")) for d in docs]
        if case["source"] in sources:
            hit += 1
            mrr_sum += 1.0 / (sources.index(case["source"]) + 1)
    return hit / len(TEST_CASES), mrr_sum / len(TEST_CASES)


def evaluate_with_rewrite(k=3):
    """Query Rewrite + 向量检索"""
    hit = 0
    mrr_sum = 0.0
    for case in TEST_CASES:
        docs = search_with_rewrite(case["question"], k=k)
        sources = [os.path.basename(d.metadata.get("source", "")) for d in docs]
        if case["source"] in sources:
            hit += 1
            mrr_sum += 1.0 / (sources.index(case["source"]) + 1)
    return hit / len(TEST_CASES), mrr_sum / len(TEST_CASES)


# ==================== 主程序 ====================
if __name__ == "__main__":
    print(f"测试集大小: {len(TEST_CASES)}")
    print("=" * 70)

    print("\n【方法 1：Vector Search】")
    r1, m1 = evaluate(k=3)
    print(f"Recall@3 = {r1:.4f}")
    print(f"MRR      = {m1:.4f}")

    print("\n【方法 2：Vector Search + Reranker】")
    r2, m2 = evaluate_with_rerank(k=3)
    print(f"Recall@3 = {r2:.4f}")
    print(f"MRR      = {m2:.4f}")

    print("\n【方法 3：Query Rewrite + Vector Search】")
    r3, m3 = evaluate_with_rewrite(k=3)
    print(f"Recall@3 = {r3:.4f}")
    print(f"MRR      = {m3:.4f}")

    # 对比表
    print("\n【对比】")
    print(f"{'方法':<38}{'Recall@3':<15}{'MRR':<15}")
    print("-" * 68)
    print(f"{'Vector Search':<38}{r1:<15.4f}{m1:<15.4f}")
    print(f"{'Vector Search + Reranker':<38}{r2:<15.4f}{m2:<15.4f}")
    print(f"{'Query Rewrite + Vector Search':<38}{r3:<15.4f}{m3:<15.4f}")