"""O1 优化实战：Cross-Encoder 重排对 R@5/top1 的离线评估（只读、零 LLM 调用 DeepSeek）。

背景：coverage_all 全组 R@5=0.8818、top1=0.6182；9 题期望文档排 6-10。
归因三类：语义近邻竞争 / 词形语域不匹配 / 专门文档被泛化文档压制。
本脚本对全 110 题的 top-10 候选做 bge-reranker-base（max_length=192，CPU 快档）
重排，量化"如果生产引入重排"的指标变化——**不改任何生产行为**。

输出：Evaluation/reports/<date>/rerank_r5_eval.json + stdout 汇总。
运行（cwd=backend/，模型在 F:/models/bge-reranker-base）：
    G:\\annacoda\\envs\\ai_agent\\python.exe -I scripts/eval_rerank_r5.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.rag.chain import get_vector_store  # noqa: E402
from acceptance.coverage import CASE_GROUPS, _hit_rank  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

RERANKER_PATH = r"F:\models\bge-reranker-base"


def doc_name(doc) -> str:
    raw = (getattr(doc, "metadata", None) or {}).get("source") or ""
    return os.path.basename(str(raw))


def main() -> None:
    vs = get_vector_store()
    ce = CrossEncoder(RERANKER_PATH, device="cpu", max_length=192)
    rows = []
    latencies = []
    for gk, (_name, cases) in CASE_GROUPS.items():
        for case in cases:
            q = case["question"]
            pairs = vs.similarity_search_with_score(q, k=10)
            docs = [d for d, _ in pairs]
            base_names = [doc_name(d) for d in docs]
            base_top = [{"doc": n, "distance": float(s)} for n, (_d, s) in zip(base_names, pairs)]
            base_rank = _hit_rank(base_top, case["expect_docs"])

            t0 = time.monotonic()
            scores = ce.predict([(q, d.page_content) for d in docs])
            latencies.append((time.monotonic() - t0) * 1000)
            order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
            rr_names = [doc_name(docs[i]) for i in order]
            rr_top = [{"doc": n, "distance": None} for n in rr_names]
            rr_rank = _hit_rank(rr_top, case["expect_docs"])

            rows.append(dict(group=gk, id=case["id"], question=q,
                             expect_docs=case["expect_docs"],
                             base_rank=base_rank, rr_rank=rr_rank,
                             base_top1=base_names[0] if base_names else None,
                             rr_top1=rr_names[0] if rr_names else None))
            if base_rank != rr_rank or (base_names[:1] != rr_names[:1]):
                print(f"[{case['id']}] base={base_rank} rr={rr_rank}  "
                      f"top1: {(base_names[0] or '')[:22]} → {(rr_names[0] or '')[:22]}")

    def _metrics(key):
        valid = [r for r in rows if r["expect_docs"]]
        n = len(valid)
        hits = lambda r: sum(1 for x in valid if x[key] is not None)  # noqa: E731
        r5 = sum(1 for x in valid if x[key] is not None and x[key] <= 5) / n
        r10 = hits(key) / n
        mrr = sum((1.0 / x[key]) if x[key] else 0.0 for x in valid) / n
        top1 = sum(1 for x in valid if x[key] == 1) / n
        return dict(recall5=round(r5, 4), recall10=round(r10, 4),
                    mrr=round(mrr, 4), top1=round(top1, 4))

    flips = [r for r in rows if r["base_top1"] != r["rr_top1"]]
    result = dict(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        n_questions=len(rows),
        baseline=_metrics("base_rank"), reranked=_metrics("rr_rank"),
        top1_flips=len(flips),
        flip_details=[dict(id=r["id"], base=r["base_top1"], rr=r["rr_top1"]) for r in flips],
        latency_ms=dict(p50=sorted(latencies)[len(latencies) // 2],
                        p95=sorted(latencies)[int(len(latencies) * 0.95) - 1],
                        mean=round(sum(latencies) / len(latencies), 1)),
        rows=rows,
    )
    out = BACKEND.parent / "Evaluation" / "reports" / datetime.now().strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rerank_r5_eval.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n== 汇总（110 题）==")
    print("baseline :", result["baseline"])
    print("reranked :", result["reranked"])
    print("top1 翻转数:", result["top1_flips"], "  延迟 p50/p95(ms):",
          result["latency_ms"]["p50"], result["latency_ms"]["p95"])
    print("报告:", path)


if __name__ == "__main__":
    main()
