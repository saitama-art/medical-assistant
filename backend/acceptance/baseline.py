"""baseline 任命与可比性校验（S4d）。

规则（方案 V2 §四，机器强制而非口头纪律）：
- **仅全量 run 可任命为基线**：scorecard.valid 且全部维度有分（非 partial）；
  部分运行可出分但永远不可成为基线；
- **逐维度指纹校验**：当前 run 与 baseline 的 git/Prompt/索引/测试集版本任一
  不同 → 禁止出 diff，只输出"条件已变化，需重定基线"并列出差异项；
- 同条件 → 输出总分差与各维度分差。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .conditions import current_conditions, fingerprint_differences

DEFAULT_BASELINE = Path(__file__).resolve().parents[2] / "Evaluation" / "baseline.json"


def appoint_baseline(report: dict, conditions: dict, out_path: Path | None = None) -> dict:
    """任命基线。非全量 run 抛 ValueError（部分口径不可成为基线）。"""
    sc = report.get("scorecard") or {}
    if not sc.get("valid"):
        raise ValueError("该 run 的评分卡无效（安全否决或数据缺失），不能任命为基线")
    dims = sc.get("dimensions") or {}
    required = {"retrieval", "answer", "refusal", "latency", "cost"}
    missing = sorted(required - set(dims))
    if missing or sc.get("partial"):
        raise ValueError(
            f"仅全量 run 可任命为基线（缺维度: {', '.join(missing) or 'partial=True'}）。"
            "部分运行可出分，但不得提升为基线")
    record = {"appointed_at": report.get("generated_at"),
              "conditions": conditions, "scorecard": sc}
    path = out_path or DEFAULT_BASELINE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def compare_with_baseline(report: dict, conditions: dict,
                          baseline_path: Path | None = None) -> dict:
    """与基线对比。指纹不一致 → comparable=False 且禁止 diff。"""
    path = baseline_path or DEFAULT_BASELINE
    if not path.exists():
        return {"comparable": False,
                "reason": f"基线文件不存在：{path}（先 --appoint 一次全量 run）"}
    baseline = json.loads(path.read_text(encoding="utf-8"))
    diffs = fingerprint_differences(conditions, baseline.get("conditions") or {})
    if diffs:
        return {"comparable": False,
                "reason": "条件已变化，需重定基线（禁止跨条件比较）",
                "condition_diffs": diffs}
    cur, base = report.get("scorecard") or {}, baseline.get("scorecard") or {}
    dim_diffs = {}
    for name, dim in (cur.get("dimensions") or {}).items():
        old = ((base.get("dimensions") or {}).get(name) or {}).get("score")
        new = dim.get("score")
        if old is not None and new is not None:
            dim_diffs[name] = round(new - old, 2)
    total_delta = None
    if cur.get("total") is not None and base.get("total") is not None:
        total_delta = round(cur["total"] - base["total"], 2)
    return {"comparable": True, "total_delta": total_delta,
            "dimension_deltas": dim_diffs,
            "baseline_total": base.get("total"), "current_total": cur.get("total")}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m acceptance.baseline",
                                 description="baseline 任命与可比性校验")
    ap.add_argument("--appoint", metavar="REPORT.json",
                    help="把 report.py 产出的聚合 JSON 任命为基线（仅全量 run）")
    ap.add_argument("--compare", metavar="REPORT.json", help="与基线对比")
    ap.add_argument("--suites", default="coverage,scenarios",
                    help="suites_executed 记录（默认 coverage,scenarios）")
    args = ap.parse_args(argv)
    if not args.appoint and not args.compare:
        ap.error("必须指定 --appoint 或 --compare")
    path = args.appoint or args.compare
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    conditions = current_conditions([s for s in args.suites.split(",") if s])
    if args.appoint:
        record = appoint_baseline(report, conditions)
        print(f"已任命基线：{DEFAULT_BASELINE}（总分 {record['scorecard']['total']}）")
    else:
        result = compare_with_baseline(report, conditions)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
