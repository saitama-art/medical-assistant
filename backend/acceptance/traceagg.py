"""trace 时延/成本聚合（评估体系 S3）：消费 coverage/health72/门禁/场景集 run JSON。

纯函数 + 薄 IO 层（aggregate_runs 只读 JSON），与执行解耦，单测锁公式，
历史 run JSON 可直接重算。口径（硬性指标总表第三/四节）：

- 时延分档：P95 ≤5s→100 / ≤8s→80 / ≤15s→50 / ≤30s→20 / >30s→0（★ 初值，
  首轮全量 run 后按分布校准）；
- 百分位口径：nearest-rank（排序后取第 ceil(p/100×n) 个，1 起），确定性、
  与手算一致；
- 分母纪律：blocked=True 的题（API/额度失败）不是时延观测也不是 completed
  产出，时延与成本分母均剔除，单独以 n_blocked 报告（诚实纪律：blocked 不
  得记 0 或剔除出报告）；
- 成本为**估算口径**（标注「估算」）：知识路径每 completed 题 3 次 LLM 调用
  （分类/判定/生成，硬性指标总表四节），token ≈ answer 字符数 / 4。
  真实 token 用量在 trace 补齐前不可得，评分卡成本维度以基线比值消费。

用法（backend/ 下）：
    python -m acceptance.traceagg ../backend/acceptance_results/health72_run4.json ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# 时延分档（秒 → 维度分），硬性指标总表第三节（★ 初值待校准）。
TIER_SECONDS = ((5, 100), (8, 80), (15, 50), (30, 20))
TIER_FALLBACK = 0  # >30s

# 知识路径 LLM 调用结构（硬性指标总表四节）：分类/判定/生成各 1 次。
CALLS_PER_COMPLETED = 3


def _percentile(sorted_vals: list[float], p: float) -> float:
    """nearest-rank：排序后第 ceil(p/100×n) 个（1 起）。调用方保证非空。"""
    n = len(sorted_vals)
    rank = min(n, max(1, math.ceil(p / 100 * n)))
    return sorted_vals[rank - 1]


def latency_summary(records_ms: list[float]) -> dict | None:
    """{p50_ms, p95_ms, n, tier}；tier 按 P95 秒值分档；空列表返回 None。"""
    vals = sorted(float(m) for m in records_ms)
    if not vals:
        return None
    p95_ms = _percentile(vals, 95)
    p95_s = p95_ms / 1000
    tier = TIER_FALLBACK
    for limit_s, score in TIER_SECONDS:
        if p95_s <= limit_s:
            tier = score
            break
    return {"p50_ms": _percentile(vals, 50), "p95_ms": p95_ms,
            "n": len(vals), "tier": tier}


def cost_estimate(records: list[dict]) -> dict:
    """成本代理（估算口径，note 标注「估算」）。

    - avg_calls_per_completed：知识路径 3 次/题；无 completed 题返回 None；
    - est_input_tokens：completed 题 answer 字符数 / 4 的合计（token≈字符/4）；
    - blocked 题不计入 completed。
    """
    completed = [r for r in records
                 if not r.get("blocked") and r.get("outcome") == "completed"]
    est_tokens = round(sum(len(r.get("answer") or "") for r in completed) / 4, 1)
    return {
        "avg_calls_per_completed": (float(CALLS_PER_COMPLETED)
                                    if completed else None),
        "est_input_tokens": est_tokens,
        "n_completed": len(completed),
        "note": "估算",
    }


def aggregate_runs(paths: list[str]) -> dict:
    """读多个 run JSON → {latency, cost, sources, warnings, n_blocked}。

    取各文件 records[] 的 ms（时延）与 outcome/answer（成本）；任何文件缺
    records 字段、读不到或 JSON 损坏 → 记 warning 不中断；blocked 题从时延
    样本与成本分母剔除，计数暴露于 n_blocked。
    """
    all_ms: list[float] = []
    all_records: list[dict] = []
    n_blocked = 0
    sources: list[str] = []
    warnings: list[str] = []
    for path in paths:
        name = Path(path).name
        try:
            rep = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            warnings.append(f"{name}: 读取/解析失败（{e}），已跳过")
            continue
        records = rep.get("records")
        if not isinstance(records, list):
            warnings.append(f"{name}: 无 records 字段，已跳过")
            continue
        sources.append(name)
        for r in records:
            if r.get("blocked"):
                n_blocked += 1
                continue
            ms = r.get("ms")
            if isinstance(ms, (int, float)) and not isinstance(ms, bool):
                all_ms.append(float(ms))
            all_records.append(r)
    return {
        "latency": latency_summary(all_ms),
        "cost": cost_estimate(all_records),
        "sources": sources,
        "warnings": warnings,
        "n_blocked": n_blocked,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m acceptance.traceagg",
                                 description="历史 run JSON 重算时延/成本（零 LLM）")
    ap.add_argument("files", nargs="+", help="run JSON 路径（一个或多个）")
    args = ap.parse_args(argv)
    agg = aggregate_runs(args.files)
    lat, cost = agg["latency"], agg["cost"]
    print(f"sources: {agg['sources']}  n_blocked={agg['n_blocked']}")
    for w in agg["warnings"]:
        print(f"WARNING: {w}")
    if lat:
        print(f"时延: p50={lat['p50_ms']:.0f}ms p95={lat['p95_ms']:.0f}ms "
              f"n={lat['n']} tier={lat['tier']}")
    else:
        print("时延: 无样本")
    print(f"成本(估算): avg_calls={cost['avg_calls_per_completed']} "
          f"est_input_tokens={cost['est_input_tokens']} "
          f"n_completed={cost['n_completed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
