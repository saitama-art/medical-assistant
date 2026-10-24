"""PA-03 rerank 延迟调优实验（只读，零 LLM）。

eval_pa03_rerank.py 首测结论：rerank 有效（期望文档 rank 14 → 1）但 k=50 全量
交叉编码 13.9s，超 TOOL_BUDGET(5s)。本脚本测三个降本手段能否把 CE 压进预算：

    1. max_length 截断（512 → 256 → 192）：序列变短，CPU 推理加速
    2. 候选集缩小（k=50 → 30 → 20）：期望文档在 k50 排 14，k20 可能已含
    3. batch 推理默认行为下的端到端耗时

同时验证：截断/缩小后 PA-03 期望文档是否仍在 rerank top-10，
及回归批次抽样的 top1/hit 是否保持稳定（截断不应改变排序结论）。

运行（cwd 必须是 backend/）：
    G:\\annacoda\\envs\\ai_agent\\python.exe -I scripts/eval_pa03_rerank_tune.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.rag.chain import get_vector_store  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

RERANKER_PATH = r"F:\models\bge-reranker-base"
TARGET_DOC = "门诊就诊指南与看病前准备"
PA03_QUESTION = "看医生前应该准备哪些信息？"

REGRESSION_SAMPLE = [
    ("出现持续性头痛应该挂什么科？", ["头痛概述"]),
    ("每天睡几个小时合适？", ["睡眠概述"]),
    ("成人腹泻时应该怎么补水？", ["蒙脱石散"]),
    ("高血压患者在生活方式上要注意什么？", ["高血压"]),
]


def doc_name(doc) -> str:
    raw = (getattr(doc, "metadata", None) or {}).get("source") or ""
    return os.path.basename(str(raw))


def expected_rank(names, expects):
    for i, name in enumerate(names, start=1):
        if any(e in name for e in expects):
            return i
    return None


def ce_rerank(ce, question, candidates, top_n=10):
    t0 = time.monotonic()
    scores = ce.predict([(question, d.page_content) for d in candidates])
    latency = (time.monotonic() - t0) * 1000
    order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
    names = [doc_name(candidates[i]) for i in order[:top_n]]
    return names, round(latency, 1)


def main() -> None:
    vs = get_vector_store()
    results = {"target": {}, "regression_sample": []}

    cand50 = [d for d, _ in vs.similarity_search_with_score(PA03_QUESTION, k=50)]
    names50 = [doc_name(d) for d in cand50]
    results["target"]["vector_rank_in_k50"] = expected_rank(names50, [TARGET_DOC])

    for ml in (512, 256, 192):
        ce = CrossEncoder(RERANKER_PATH, device="cpu", max_length=ml)
        names, latency = ce_rerank(ce, PA03_QUESTION, cand50)
        results["target"][f"ce_k50_ml{ml}"] = {
            "latency_ms": latency, "target_rank": expected_rank(names, [TARGET_DOC]),
            "top5": names[:5],
        }

    ce256 = CrossEncoder(RERANKER_PATH, device="cpu", max_length=256)
    for k in (30, 20):
        cand = cand50[:k]
        names, latency = ce_rerank(ce256, PA03_QUESTION, cand)
        results["target"][f"ce_k{k}_ml256"] = {
            "vector_has_target": expected_rank([doc_name(d) for d in cand], [TARGET_DOC]),
            "latency_ms": latency, "target_rank": expected_rank(names, [TARGET_DOC]),
            "top5": names[:5],
        }

    for q, expects in REGRESSION_SAMPLE:
        cand20 = [d for d, _ in vs.similarity_search_with_score(q, k=20)]
        names, latency = ce_rerank(ce256, q, cand20)
        results["regression_sample"].append({
            "question": q, "latency_ms": latency,
            "hit_rank_k20_ml256": expected_rank(names, expects),
            "top3": names[:3],
        })

    path = BACKEND / "scripts" / "eval_pa03_tune_20261024.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("== 目标题 PA-03：CE 延迟调优 ==")
    for k, v in results["target"].items():
        if k == "vector_rank_in_k50":
            print(f"向量检索 k50 中期望文档名次 = {v}")
        else:
            print(f"[{k}] latency={v['latency_ms']}ms  target_rank={v.get('target_rank')}"
                  f"  vector_has_target={v.get('vector_has_target', '-')}")
            print("   top5:", " | ".join(v["top5"][:5]))
    print("\n== 回归抽样（k20 + ml256） ==")
    for r in results["regression_sample"]:
        print(f"  hit={r['hit_rank_k20_ml256']}  {r['latency_ms']}ms  {r['question'][:20]}")
        print("     top3:", " | ".join(r["top3"]))
    print(f"\n原始数据：{path}")


if __name__ == "__main__":
    main()
