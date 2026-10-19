import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ========== 1. 只加载 data 目录 ==========
data_dir = "data"   # ← 不要扫 uploads
documents = []
seen_files = set()  # 去重

for filename in os.listdir(data_dir):
    if filename.endswith(".pdf"):
        if filename in seen_files:
            print(f"跳过重复: {filename}")
            continue
        seen_files.add(filename)
        loader = PyPDFLoader(os.path.join(data_dir, filename))
        documents.extend(loader.load())
        print(f"加载 {filename}")

print(f"总共加载 {len(documents)} 页")

# ========== 2. 切片 ==========
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
chunks = text_splitter.split_documents(documents)
print(f"切成 {len(chunks)} 个 chunk")

# ========== 3. 向量化 ==========
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="F:\medical-assistant\backend\chroma_db",
)
print(f"存入 Chroma: {vectorstore._collection.count()} 个向量")