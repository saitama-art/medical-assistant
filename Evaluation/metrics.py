"""检索指标纯函数：从 coverage/health72 的 run JSON（或内存 records）计算指标。

设计原则（实施方案 S1）：
- 纯函数、零 IO：输入 records 列表（dict），输出指标 dict——与执行解耦，
  单测锁公式，历史 JSON 也能重算（见 main() 的 CLI 入口）；
- 分母纪律：``expect_docs`` 为空的题是**已确认的知识缺口**
  （KNOWLEDGE_MISSING，见 coverage.classify），不是检索失败——
  一律排除在 Recall/MRR/top1 的分母之外，单独以 n_gap 报告。
  「覆盖率」（含缺口）= n_valid_hits / n_total，报告里与 Recall 并列展示，
  两个数各有各的用途，不混用；
- 命中口径与 coverage.classify 一致：hit_rank 为期望文档在 top-k 中的名次
  （1 起），None = 未命中；
- 指标只在**固定测试集**上有意义，报告消费方负责加口径声明（设计 6.0 通则）。

用法：
    python Evaluation/metrics.py backend/acceptance_results/coverage_p1b_xxx.json ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_KS = (5, 10)


def _valid(records: list[dict]) -> list[dict]:
    """有标注（expect_docs 非空）的题——指标分母。"""
    return [r for r in records if r.get("expect_docs")]


def recall_at_k(records: list[dict], k: int) -> float | None:
    """Recall@K = hit_rank ≤ K 的题数 / 有效题数；无有效题返回 None。"""
    valid = _valid(records)
    if not valid:
        return None
    hits = sum(1 for r in valid
               if r.get("hit_rank") is not None and r["hit_rank"] <= k)
    return round(hits / len(valid), 4)


def mrr(records: list[dict]) -> float | None:
    """MRR = mean(1/hit_rank)，未命中计 0；无有效题返回 None。"""
    valid = _valid(records)
    if not valid:
        return None
    s = sum((1.0 / r["hit_rank"]) if r.get("hit_rank") else 0.0 for r in valid)
    return round(s / len(valid), 4)


def top1_rate(records: list[dict]) -> float | None:
    """top1 命中率 = hit_rank == 1 的题数 / 有效题数（观察指标，不设门槛）。"""
    valid = _valid(records)
    if not valid:
        return None
    hits = sum(1 for r in valid if r.get("hit_rank") == 1)
    return round(hits / len(valid), 4)


def coverage_rate(records: list[dict]) -> float | None:
    """覆盖率（含缺口口径）= 命中题数 / 总题数。与 Recall 并列展示，不替代。"""
    if not records:
        return None
    hits = sum(1 for r in records if r.get("hit_rank") is not None)
    return round(hits / len(records), 4)


def failure_composition(records: list[dict]) -> dict[str, int]:
    """按 verdict 计数（含成功类）。归因不丢：调检索参数前先看病种。"""
    comp: dict[str, int] = {}
    for r in records:
        v = r.get("verdict") or "UNKNOWN"
        comp[v] = comp.get(v, 0) + 1
    return dict(sorted(comp.items()))


# ---- 拒答指标（实施方案 S3，场景集 run JSON 契约驱动）----
# 分母纪律（与 S3 交接契约一致）：
# - blocked=True 记 N/A（n_na），不入任何分母；
# - flags 含 'sentinel'（哨兵观察题，如 CS-09/10）只单独计数（n_sentinel_refused），
#   不进正确/误拒任何分母分子——「只记录不入断言」，供评分卡拒答健康维度引用；
# - 正确拒答还需 answer 无「【参考来源】」段（拒答题不得带来源段，T-02b 口径）。
SOURCE_SECTION_MARK = "【参考来源】"


def refusal_rates(records: list[dict]) -> dict:
    """场景集拒答指标。返回::

        {correct_refusal_rate, false_refusal_rate, n_refuse, n_answerable,
         n_na, n_sentinel, n_sentinel_refused, n_correct_refusal,
         n_false_refusal}

    - correct = flags 含 'refuse' 且 outcome == expected_outcome 且 answer 无
      「【参考来源】」段；
    - false_refusal = flags 含 'answerable'（且非哨兵）且
      outcome == 'insufficient_evidence'；
    - 分母为空的比率返回 None（N/A），不得记 0。
    """
    n_refuse = n_answerable = n_sentinel = n_na = 0
    n_correct = n_false = n_sentinel_refused = 0
    for r in records:
        if r.get("blocked"):
            n_na += 1
            continue
        flags = r.get("flags") or []
        if "sentinel" in flags:
            n_sentinel += 1
            if r.get("outcome") == "insufficient_evidence":
                n_sentinel_refused += 1
            continue
        if "refuse" in flags:
            n_refuse += 1
            if (r.get("outcome") is not None
                    and r.get("outcome") == r.get("expected_outcome")
                    and SOURCE_SECTION_MARK not in (r.get("answer") or "")):
                n_correct += 1
        elif "answerable" in flags:
            n_answerable += 1
            if r.get("outcome") == "insufficient_evidence":
                n_false += 1
    return {
        "correct_refusal_rate": round(n_correct / n_refuse, 4) if n_refuse else None,
        "false_refusal_rate": round(n_false / n_answerable, 4) if n_answerable else None,
        "n_refuse": n_refuse,
        "n_answerable": n_answerable,
        "n_na": n_na,
        "n_sentinel": n_sentinel,
        "n_sentinel_refused": n_sentinel_refused,
        "n_correct_refusal": n_correct,
        "n_false_refusal": n_false,
    }


def refusal_health_score(records: list[dict]) -> float | None:
    """拒答健康维度分（硬性指标总表第三节）::

        max(0, 正确拒答率×100 − 误拒数×5)

    误拒数 = refusal_rates 的 n_false_refusal（哨兵题拒答不计入扣分，仅经
    n_sentinel_refused 暴露给评分卡）；正确拒答题分母为空（无 refuse 题）时
    返回 None（N/A），不得记 0。
    """
    rates = refusal_rates(records)
    if rates["correct_refusal_rate"] is None:
        return None
    return max(0, rates["correct_refusal_rate"] * 100
               - rates["n_false_refusal"] * 5)


def summarize_records(records: list[dict], ks: tuple[int, ...] = DEFAULT_KS) -> dict:
    """单组指标汇总。hit_rank 记录的是本次运行的 top-k 深度（默认 10），
    因此 Recall@K 只在 K ≤ 该深度时有意义；CLI 层不额外约束，消费方自知。"""
    valid = _valid(records)
    out: dict = {
        "n_total": len(records),
        "n_valid": len(valid),
        "n_gap": len(records) - len(valid),
        "coverage_rate": coverage_rate(records),
        "mrr": mrr(records),
        "top1_rate": top1_rate(records),
        "failure_composition": failure_composition(records),
    }
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(records, k)
    return out


def summarize_groups(group_reports: dict[str, dict],
                     ks: tuple[int, ...] = DEFAULT_KS) -> dict:
    """多组汇总：每组一行 + 全组合计（按各组有效题数加权）。"""
    groups = {name: summarize_records(rep.get("records", []), ks)
              for name, rep in group_reports.items()}
    all_records = [r for rep in group_reports.values()
                   for r in rep.get("records", [])]
    return {"groups": groups, "overall": summarize_records(all_records, ks)}


def _load_records(path: str) -> list[dict]:
    rep = json.loads(Path(path).read_text(encoding="utf-8"))
    if "records" not in rep:
        raise ValueError(f"{path}: 不是 coverage/health72 报告（无 records 字段）")
    return rep["records"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python Evaluation/metrics.py",
                                 description="从历史 run JSON 重算检索指标（零 LLM）")
    ap.add_argument("files", nargs="+", help="coverage 报告 JSON 路径（一个或多个）")
    ap.add_argument("--k", type=int, action="append", default=None,
                    help="额外指定 K（可多次，默认 5/10）")
    args = ap.parse_args(argv)
    ks = tuple(args.k) if args.k else DEFAULT_KS
    for path in args.files:
        records = _load_records(path)
        s = summarize_records(records, ks)
        print(f"== {path}")
        print(f"   n_valid={s['n_valid']} n_gap={s['n_gap']} "
              + " ".join(f"R@{k}={s[f'recall@{k}']}" for k in ks)
              + f" MRR={s['mrr']} top1={s['top1_rate']} cov={s['coverage_rate']}")
        print(f"   构成: {s['failure_composition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
