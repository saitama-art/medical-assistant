"""PA-03 检索排序评估（只读，零 LLM）。

背景（docs/开发进度.md 2026-10-22）：PA-03「看医生前应该准备哪些信息？」的期望文档
《门诊就诊指南与看病前准备（北京市中关村医院）》在库（149 份 PDF / 1696 chunks）
但该问法下未进 top-10——72 题任务收官后的唯一系统侧遗留。本脚本按 10-22 记录
「PA-03 类检索边际案例攒批后统一评估 rerank」的约定，对两个候选手段做零 LLM
的对比实验，产出数据后由数据决定下一步，不在本脚本里改任何生产行为：

    A. 基线：现有链路  bge-small-zh 向量检索，k=10（生产口径）
    B. rerank：k=50 候选 + bge-reranker-base 交叉编码器重排，取 top-10
    C. query 扩展（embedding 侧手工同义改写，零 LLM）

同时用一个已知良好的回归批次（PA/P1A/P1B 组中 top1 稳定命中的 11 题）验证
手段 B/C 是否引入回归（top1 翻转 / 期望文档掉出 top-k）。任何手段只有在
「PA-03 期望文档进 top-k」与「回归批次零翻转」同时成立时，才值得进生产评估
（后续还需过 72 题 run5 全量回归与 G-3 预算门禁）。

输出：原始数据 scripts/eval_pa03_20261024.json + stdout 摘要。
运行（cwd 必须是 backend/）：
    G:\\annacoda\\envs\\ai_agent\\python.exe -I scripts/eval_pa03_rerank.py
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

RERANKER_PATH = r"F:\models\bge-reranker-base"
TARGET_DOC = "门诊就诊指南与看病前准备"  # PA-03 期望文档（按子串匹配）

PA03_QUESTION = "看医生前应该准备哪些信息？"

# query 扩展变体（embedding 侧，零 LLM）：就诊场景同义词 + 材料枚举
QEXP_VARIANTS = [
    "看医生前应该准备哪些信息？就诊前准备 病史资料 症状记录 用药清单 检查报告",
    "门诊就诊前需要准备什么材料？既往病史 正在服用的药物 过敏史 体检报告",
]

# 回归批次：覆盖集里 top1 稳定命中的题（来源 docs/开发进度.md 各轮实测记录）。
# expect = 期望文档名子串（任一命中即算 top-k 命中；top1 单独记录）。
REGRESSION_BATCH = [
    ("出现持续性头痛应该挂什么科？", ["头痛概述"]),
    ("腹痛、胸闷、咳嗽等症状通常如何选择就诊科室？", ["慢性腹痛", "胸痛"]),
    ("如何记录家庭血压和血糖？", ["高血压", "2型糖尿病"]),
    ("体检发现异常后应该如何安排复查？", ["常见的医疗检查"]),
    ("每天睡几个小时合适？", ["睡眠概述"]),
    ("打鼾严重是否需要就医？", ["打鼾"]),
    ("失眠了怎么办？", ["失眠"]),
    ("成人腹泻时应该怎么补水？", ["蒙脱石散"]),
    ("高血压患者在生活方式上要注意什么？", ["高血压"]),
    ("感冒和流感有什么区别？", ["流行性感冒", "普通感冒"]),
    ("盐酸二甲双胍片的适应症是什么？", ["二甲双胍"]),
]


def doc_name(doc) -> str:
    raw = (getattr(doc, "metadata", None) or {}).get("source") or ""
    return os.path.basename(str(raw))


def expected_rank(names, expects) -> int | None:
    """expects 中任一子串首次命中的名次（1 基）；未命中返回 None。"""
    for i, name in enumerate(names, start=1):
        if any(e in name for e in expects):
            return i
    return None


def main() -> None:
    vs = get_vector_store()
    cross_encoder = None  # 惰性加载

    def rerank_top10(question: str, candidates: list) -> list[tuple[str, float]]:
        nonlocal cross_encoder
        if cross_encoder is None:
            from sentence_transformers import CrossEncoder

            # max_length=192：调优实测（eval_pa03_tune_20261024.json）50 对 4.7s < 5s
            # 预算，且排序结论与 512 一致（目标题 rank 1 不变）；192 截断只影响
            # 长 chunk 尾部，rerank 的相对排序对此不敏感。
            cross_encoder = CrossEncoder(RERANKER_PATH, device="cpu", max_length=192)
        t0 = time.monotonic()
        pairs = [(question, d.page_content) for d in candidates]
        scores = cross_encoder.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        latency = (time.monotonic() - t0) * 1000
        rerank_top10.last_latency = latency  # type: ignore[attr-defined]
        return [(doc_name(candidates[i]), float(scores[i])) for i in order[:10]]

    out = {"target": {"question": PA03_QUESTION, "doc": TARGET_DOC}, "target_results": {}}

    # ---------- A. 基线 + 期望文档在更大候选集里的位置 ----------
    base10 = vs.similarity_search_with_score(PA03_QUESTION, k=10)
    names10 = [doc_name(d) for d, _ in base10]
    cand50 = vs.similarity_search_with_score(PA03_QUESTION, k=50)
    names50 = [doc_name(d) for d, _ in cand50]
    out["target_results"]["A_baseline_k10"] = {
        "top10": names10,
        "target_rank": expected_rank(names10, [TARGET_DOC]),
        "target_rank_in_k50": expected_rank(names50, [TARGET_DOC]),
    }

    # ---------- B. rerank（k=50 → CE top-10） ----------
    rr = rerank_top10(PA03_QUESTION, [d for d, _ in cand50])
    out["target_results"]["B_rerank_k50"] = {
        "top10": [n for n, _ in rr],
        "target_rank": expected_rank([n for n, _ in rr], [TARGET_DOC]),
        "latency_ms": round(getattr(rerank_top10, "last_latency", 0.0), 1),
    }

    # ---------- C. query 扩展 ----------
    for label, q in zip(("C1_qexp_materials", "C2_qexp_visit"), QEXP_VARIANTS):
        r = vs.similarity_search_with_score(q, k=10)
        names = [doc_name(d) for d, _ in r]
        out["target_results"][label] = {
            "query": q,
            "top10": names,
            "target_rank": expected_rank(names, [TARGET_DOC]),
        }

    # ---------- 回归批次：基线 vs rerank ----------
    reg_rows = []
    for q, expects in REGRESSION_BATCH:
        b = vs.similarity_search_with_score(q, k=10)
        b_names = [doc_name(d) for d, _ in b]
        c50 = vs.similarity_search_with_score(q, k=50)
        rr_names = [n for n, _ in rerank_top10(q, [d for d, _ in c50])]
        reg_rows.append({
            "question": q,
            "expects": expects,
            "baseline_top1": b_names[0] if b_names else None,
            "baseline_hit_rank": expected_rank(b_names, expects),
            "rerank_top1": rr_names[0] if rr_names else None,
            "rerank_hit_rank": expected_rank(rr_names, expects),
            "top1_flipped": (b_names[0] if b_names else None) != (rr_names[0] if rr_names else None),
        })
    out["regression_batch"] = reg_rows

    path = BACKEND / "scripts" / "eval_pa03_20261024.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- stdout 摘要 ----------
    print("== PA-03 目标题 ==")
    for k, v in out["target_results"].items():
        if k.startswith("A"):
            print(f"[{k}] target_rank_in_top10={v['target_rank']}  in_k50={v['target_rank_in_k50']}")
            print("  top10:", " | ".join(v["top10"][:5]))
        elif k.startswith("B"):
            print(f"[{k}] target_rank_in_top10={v['target_rank']}  CE延迟={v['latency_ms']}ms")
            print("  top10:", " | ".join(v["top10"][:5]))
        else:
            print(f"[{k}] target_rank_in_top10={v['target_rank']}")
            print("  top10:", " | ".join(v["top10"][:5]))
    print("\n== 回归批次（基线 vs rerank） ==")
    n_flip = 0
    for r in reg_rows:
        flag = "FLIP" if r["top1_flipped"] else "ok"
        if r["top1_flipped"]:
            n_flip += 1
        print(f"[{flag}] {r['question'][:22]}... base_hit={r['baseline_hit_rank']} rr_hit={r['rerank_hit_rank']}")
        if r["top1_flipped"]:
            print(f"        base_top1={r['baseline_top1']}  rr_top1={r['rerank_top1']}")
    print(f"\ntop1 翻转 {n_flip}/{len(reg_rows)}")
    print(f"原始数据：{path}")


if __name__ == "__main__":
    main()
