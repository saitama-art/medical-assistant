import os
import shutil
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ========== 1. 准备两个假文档 ==========
docs = [
    Document(page_content="来那度胺是一种治疗多发性骨髓瘤的药物", metadata={"source": "test1"}),
    Document(page_content="布洛芬是常见的退烧止痛药", metadata={"source": "test2"}),
]
print(f"步骤1: 准备了 {len(docs)} 个文档")

# ========== 2. 准备 embedding ==========
print("步骤2: 加载 embedding 模型...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
print("步骤2: embedding 加载完成")

# ========== 3. 删掉旧的测试目录 ==========
test_dir = r"F:\medical-assistant\backend\test_chroma"
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
    print(f"步骤3: 删除了旧的 {test_dir}")

# ========== 4. 写入 Chroma ==========
print("步骤4: 写入 Chroma...")
vs = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=test_dir,
)

# ========== 5. 立刻检查 ==========
print(f"步骤5: 写入后 _collection.count() = {vs._collection.count()}")
print(f"步骤5: 写入后 len(vs.get()['ids']) = {len(vs.get()['ids'])}")

# ========== 6. 检查磁盘 ==========
print(f"步骤6: test_chroma 目录内容 = {os.listdir(test_dir) if os.path.exists(test_dir) else '不存在'}")

# ========== 7. 重新加载 ==========
print("步骤7: 重新加载...")
vs2 = Chroma(persist_directory=test_dir, embedding_function=embeddings)
print(f"步骤7: 重新加载 count = {vs2._collection.count()}")

# ========== 8. 检索测试 ==========
print("步骤8: 检索测试...")
results = vs2.similarity_search("来那度胺", k=1)
print(f"步骤8: 检索结果数 = {len(results)}")
if results:
    print(f"步骤8: 内容 = {results[0].page_content}")