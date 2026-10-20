"""离线建库脚本：把 backend/data/ 的 PDF 切分、向量化后写入 Chroma。

与查询侧共用同一份 embedding 配置的保证方式：**直接复用 chain.get_embedding()**
（chain.py 是查询侧唯一 embedding 出口，ingest 不再自建模型）——建库与查询
使用同一模型由代码结构保证，而不是靠两份配置人工对齐。

用法（在 backend/ 目录下、ai_agent 环境）：

    python -m app.rag.ingest

注意：本脚本会删除并重建 CHROMA_DIR 下的既有索引，只应在明确的重建窗口执行。
"""

import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from .chain import get_embedding  # 复用查询侧唯一 embedding 出口（无循环导入：chain 不依赖 ingest）
from .chapter_split import split_documents  # FR-01：说明书按【章节】边界切，其余维持递归切分

# ==================== 当前实验参数 ====================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ========== 路径（相对本文件解析，落到 backend/chroma_db 与 backend/data） ==========
# ingest.py 位于 backend/app/rag/：parents[0]=rag，[1]=app，[2]=backend
_BACKEND_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = str(_BACKEND_DIR / "chroma_db")
DATA_DIR = str(_BACKEND_DIR / "data")


def main() -> None:
    print("=" * 70)
    print(f"【Ingest】chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    print(f"【Ingest】DATA_DIR={DATA_DIR}")
    print(f"【Ingest】CHROMA_DIR={CHROMA_DIR}")
    print("=" * 70)

    # ========== 1. 加载所有 PDF ==========
    documents = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, filename))
            documents.extend(loader.load())
            print(f"加载 {filename}")

    print(f"总共加载 {len(documents)} 页")

    # ========== 2. 切片 ==========
    # FR-01：含【章节】标记的说明书按章节边界切（章节标题保留为 chunk 前缀，
    # 小章节合并、大章节细切）；无标记文档维持原 RecursiveCharacterTextSplitter。
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"切成 {len(chunks)} 个 chunk")

    # ========== 3. embedding（与查询侧同一配置） ==========
    embeddings = get_embedding()

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


if __name__ == "__main__":
    # 模块级执行曾是隐患：import 本模块就会删库重建。必须经 main 入口显式运行。
    main()
