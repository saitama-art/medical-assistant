from langchain_huggingface import HuggingFaceEmbeddings
import json

embeddings = HuggingFaceEmbeddings(model_name = r"C:\Users\user\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5\snapshots\bge",
)

vec = embeddings.embed_query("来那度胺")
print("长度:", len(vec))

# 全部打印（会刷屏）
print(vec)

# 保存到文件（推荐）
with open("vec.json", "w", encoding="utf-8") as f:
    json.dump(vec, f, ensure_ascii=False, indent=2)
print("已保存到 vec.json")