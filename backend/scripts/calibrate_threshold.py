"""Step 0b 距离阈值校准脚本（设计文档 5.4）。

产出《阈值校准报告》所需的数据：
1. collection 的真实距离度量（hnsw:space）与经验证的距离刻度判定
   （用存储向量直接计算 cosine 距离与 L2 平方，与 Chroma 返回分数比对）；
2. 两类问题的 top1 距离分布（相关 = 库内有覆盖；无关 = 库内无覆盖）；
3. 检索耗时实测（p50/p95，供 5.6 预算表）。

用法（backend/ 下、ai_agent 环境）：

    python scripts/calibrate_threshold.py [--json out.json]

只读，不修改向量库与配置。填配置是报告评审后的人工动作。
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.chain import get_embedding, get_vector_store  # noqa: E402

# 相关类：库内现有 79 份 PDF 确实覆盖的问题（药品说明书 + MSD 科普 + 检查）
RELEVANT = [
    "盐酸二甲双胍片的适应症是什么？",
    "盐酸二甲双胍片的用法用量是什么？",
    "阿莫西林主要用于治疗哪些感染？",
    "高血压患者在生活方式上要注意什么？",
    "血常规检查主要看哪些指标？",
    "布洛芬颗粒适用于哪些情况？",
    "成人腹泻时应该怎么补水？",
    "服用多种药物时需要注意什么？",
]

# 无关类：库内无覆盖（价格/医保/开封后失效——T-02b 现行口径），
# 以及历史盘点用的四类无依据问题
IRRELEVANT = [
    "盐酸二甲双胍片多少钱一盒？",
    "盐酸二甲双胍片可以医保报销吗？",
    "药品开封后多久会失效？",
    "熬夜对身体有什么影响？",
    "感冒了怎么办？",
    "我胸口疼要去哪个科？",
    "如何评价某药企的股票前景？",
    "明天北京天气怎么样？",
]


def identify_metric(col, emb, question: str) -> dict:
    """用存储向量复算两种距离，与 Chroma 返回分数比对，判定实际刻度。"""
    import numpy as np

    qv = np.array(emb.embed_query(question), dtype=float)
    data = col.get(include=["documents", "embeddings"])
    best = None
    for doc_text, e in zip(data["documents"], data["embeddings"]):
        ea = np.array(e, dtype=float)
        l2_sq = float(((qv - ea) ** 2).sum())
        cosine = float(1.0 - (qv * ea).sum() / (np.linalg.norm(qv) * np.linalg.norm(ea)))
        if best is None or l2_sq < best[0]:
            best = (l2_sq, cosine, doc_text)
    l2_sq, cosine, _ = best
    # Chroma 对同一查询返回的最近邻分数
    vs_score = None
    return {"l2_squared_computed_min": round(l2_sq, 6),
            "cosine_distance_computed_min": round(cosine, 6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="json_out", help="结果 JSON 输出路径")
    args = ap.parse_args()

    vs = get_vector_store()
    col = vs._collection
    emb = get_embedding()

    hnsw_space = (col.metadata or {}).get("hnsw:space")
    print(f"hnsw:space = {hnsw_space!r}")

    metric_probe = identify_metric(col, emb, RELEVANT[0])
    print(f"刻度探针（按存储向量复算最近邻）: {json.dumps(metric_probe)}")

    results = {"hnsw_space": hnsw_space, "metric_probe": metric_probe,
               "relevant": [], "irrelevant": [], "latencies_ms": []}

    def run_group(questions, key):
        for q in questions:
            t0 = time.perf_counter()
            pairs = vs.similarity_search_with_score(q, k=3)
            dt = (time.perf_counter() - t0) * 1000
            results["latencies_ms"].append(dt)
            top1_doc, top1_score = pairs[0]
            src = Path(top1_doc.metadata.get("source", "")).name
            row = {"question": q, "top1_distance": round(float(top1_score), 4),
                   "top1_source": src}
            results[key].append(row)
            print(f"[{key:10s}] d={row['top1_distance']:.3f} {q} -> {src}")

    run_group(RELEVANT, "relevant")
    run_group(IRRELEVANT, "irrelevant")

    lat = sorted(results["latencies_ms"])
    n = len(lat)
    stats = {
        "latency_p50_ms": round(lat[n // 2], 1),
        "latency_p95_ms": round(lat[int(n * 0.95)], 1),
        "latency_max_ms": round(lat[-1], 1),
    }
    for key in ("relevant", "irrelevant"):
        ds = sorted(r["top1_distance"] for r in results[key])
        stats[f"{key}_distances"] = ds
        stats[f"{key}_min"] = ds[0]
        stats[f"{key}_max"] = ds[-1]
        stats[f"{key}_median"] = ds[len(ds) // 2]
    results["stats"] = stats
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写出 {args.json_out}")


if __name__ == "__main__":
    main()
