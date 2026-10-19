import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = r"/data"

# 只加载一份 PDF
docs = PyPDFLoader(os.path.join(DATA_DIR, "国药准字H20193006_来那度胺胶囊.pdf")).load()
print(f"页数: {len(docs)}")

# 切片
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
chunks = splitter.split_documents(docs)
print(f"chunk 数: {len(chunks)}")

# 打印前 5 个 chunk
for i, chunk in enumerate(chunks[:5]):
    print(f"\n{'=' * 60}")
    print(f"chunk {i}，长度: {len(chunk.page_content)} 字")
    print(f"来源: {chunk.metadata.get('source')}")
    print(f"页码: {chunk.metadata.get('page')}")
    print(f"内容:\n{chunk.page_content}")