"""运行条件指纹（S4d）：把"这次 run 是在什么条件下跑的"变成可机检的数据。

评分卡的可比性前提：分数差必须可归因于"系统变化"而非"运行条件差异"。
本模块采集每次 run 的条件指纹，供 baseline 任命与逐维度 diff 校验使用。

采集项（方案 V2 §四）：
- git_sha：代码版本（HEAD 短哈希）；
- prompt_hashes：5 份生产 Prompt + safety 词表短哈希（复用 S4a promptmeta，
  AST 静态提取，不 import 生产模块）；
- index_stats：知识库份数与 chunks（chromadb PersistentClient 直读计数，
  不加载 embedding 模型）；
- model / testset_versions / suites_executed。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import promptmeta

BACKEND_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BACKEND_DIR / "chroma_db"


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=BACKEND_DIR.parent, capture_output=True,
                             text=True, check=True, timeout=10)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def index_stats() -> dict:
    """知识库统计（只读）。docs = backend/data 下 PDF 数；chunks = collection.count()。"""
    docs = len(list((BACKEND_DIR / "data").glob("*.pdf")))
    chunks = None
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        for coll in client.list_collections():
            name = getattr(coll, "name", coll)
            if name == "langchain":
                chunks = coll.count()
                break
        if chunks is None:
            cols = client.list_collections()
            if cols:
                chunks = cols[0].count()
    except Exception as exc:  # noqa: BLE001
        chunks = f"unavailable:{type(exc).__name__}"
    return {"docs": docs, "chunks": chunks}


def current_conditions(suites: list[str], *, git_sha: str | None = None,
                       stats: dict | None = None) -> dict:
    """组装条件指纹。git_sha/stats 可注入（测试免 IO）。"""
    meta = promptmeta.prompt_meta(BACKEND_DIR)
    return {
        "git_sha": git_sha if git_sha is not None else _git_sha(),
        "promptmeta": meta,
        "index_stats": stats if stats is not None else index_stats(),
        "model": "deepseek-chat",
        "testset_versions": {"coverage": "v2026-10-22", "health72": "2026-10-21",
                             "scenarios": "2026-10-25"},
        "suites_executed": sorted(suites),
    }


def fingerprint_differences(a: dict, b: dict) -> list[str]:
    """返回两份条件指纹的差异项（用于可比性校验；空列表=同条件可比）。

    suites_executed 不参与比较（它是这次跑了哪些套件的事实，不是系统条件）。
    """
    diffs = []
    if a.get("git_sha") != b.get("git_sha"):
        diffs.append(f"git_sha: {a.get('git_sha')} → {b.get('git_sha')}")
    pa, pb = a.get("promptmeta") or {}, b.get("promptmeta") or {}
    ha, hb = pa.get("prompt_hashes", {}), pb.get("prompt_hashes", {})
    for key in sorted(set(ha) | set(hb)):
        if ha.get(key) != hb.get(key):
            diffs.append(f"prompt[{key}]: {ha.get(key)} → {hb.get(key)}")
    if pa.get("safety_wordlist_hash") != pb.get("safety_wordlist_hash"):
        diffs.append("safety_wordlist 已变化")
    sa, sb = a.get("index_stats") or {}, b.get("index_stats") or {}
    if sa != sb:
        diffs.append(f"index_stats: {sa} → {sb}")
    if a.get("testset_versions") != b.get("testset_versions"):
        diffs.append("testset_versions 已变化")
    return diffs
