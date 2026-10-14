
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


from langchain_chroma import Chroma
# ==================== 当前实验参数 ====================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

print("=" * 70)
print(f"【Ingest】chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
print("=" * 70)
# ========== 路径 ==========
CHROMA_DIR = r"/chroma_db"
DATA_DIR = r"/data"

# ========== 1. 加载所有 PDF ==========
documents = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_DIR, filename))
        documents.extend(loader.load())
        print(f"加载 {filename}")

print(f"总共加载 {len(documents)} 页")

# ========== 2. 切片 ==========
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
chunks = splitter.split_documents(documents)
print(f"切成 {len(chunks)} 个 chunk")

# ========== 3. embedding ==========
embeddings = HuggingFaceEmbeddings(
    model_name=r"C:\Users\user\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5\snapshots\bge",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ========== 4. 删旧的 ==========
if os.path.exists(CHROMA_DIR):
    print(f"删除旧的: {CHROMA_DIR}")
    shutil.rmtree(CHROMA_DIR)

# ========== 5. 写入 ==========
vs = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)

# ========== 6. 确认 ==========
print(f"写入后 count: {vs._collection.count()}")
print(f"目录内容: {os.listdir(CHROMA_DIR)}")