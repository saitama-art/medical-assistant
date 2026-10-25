"""上下文工程五指标纯函数（S2，消费 acceptance.scenarios 的 run JSON）。

口径权威：docs/评估体系实施方案V2 §二（分母纪律 / N/A 四规则 / 标注粒度）。
全部确定性、零 LLM：输入 records 列表（scenario run JSON 的 records 字段），
输出 {value, n_denom, n_num} 或 None（N/A——分母为空/全部 blocked，绝不记 0）。

N/A 规则实现：
  1. watch 题（观察题）→ 不入任何比率，只进哨兵计数；
  2. blocked=True → 该题全部指标剔除；
  3. 分母为空 → 返回 None（调用方展示"N/A"，不得当 0）；
  4. sentinel 误拒事件单独计数（sentinel_refused），不进 false 类分子。
"""

from __future__ import annotations


def _ratable(rec: dict) -> bool:
    """是否入指标分母：非观察题且非 blocked。"""
    return rec.get("mode") != "watch" and not rec.get("blocked")


def _has_flag(rec: dict, flag: str) -> bool:
    return flag in (rec.get("flags") or [])


def _result(value, denom, num):
    if not denom:
        return None
    return dict(value=round(num / denom, 4), n_denom=denom, n_num=num)


def completion_correct_rate(records: list[dict]):
    """补全正确率：多轮 lock 题中 outcome 与对照一致 且 top1 来源一致的占比。

    对照口径（§六 CS-14/15/17）：outcome == control_outcome ∧ top1 == control_top1；
    对照 blocked 时该题剔除（N/A 规则 2 的对照版）。"""
    denom, num = 0, 0
    for r in records:
        if not (_ratable(r) and _has_flag(r, "multi_turn")):
            continue
        if not r.get("control_top1") or r.get("control_blocked"):
            continue
        denom += 1
        if (r.get("outcome") == r.get("control_outcome")
                and (r.get("top1") or "") == (r.get("control_top1") or "")):
            num += 1
    return _result(num / max(denom, 1) if denom else None, denom, num)


def constraint_retention_rate(records: list[dict]):
    """约束保留率：resolved_question 逐词含题面标注的全部限定词的占比（§二逐词校验）。"""
    denom, num = 0, 0
    for r in records:
        if not (_ratable(r) and r.get("constraint")):
            continue
        words = (r["constraint"] or {}).get("words") or []
        if not words:
            continue
        denom += 1
        resolved = r.get("resolved_question") or ""
        if all(w in resolved for w in words):
            num += 1
    return _result(num / max(denom, 1) if denom else None, denom, num)


def interference_rate(records: list[dict]):
    """历史干扰率：干扰题回答来源落入干扰域/张冠李戴的占比（越低越好，0 为优）。

    判定（程序可判部分）：top1 含题面标注的 interference_domain 子串即记 1 次干扰；
    语义判读留 ai_assisted 人工复核（review_status 字段由 runner 消费方补）。"""
    denom, num = 0, 0
    for r in records:
        if not (_ratable(r) and _has_flag(r, "interference")):
            continue
        domain = r.get("interference_domain")
        if not domain:
            continue
        denom += 1
        if domain in (r.get("top1") or ""):
            num += 1
    return _result(num / max(denom, 1) if denom else None, denom, num)


def degrade_correct_rate(records: list[dict]):
    """降级正确率：不猜指代的合规降级占比——outcome 符合预期集合 ∧ sources 空
    ∧ 回答含澄清引导（「请补充说明」或「重新完整描述」）。"""
    clues = ("请补充说明", "重新完整描述", "请把问题重新完整描述")
    denom, num = 0, 0
    for r in records:
        if not (_ratable(r) and _has_flag(r, "degrade")):
            continue
        denom += 1
        allowed = r.get("expected_outcome_any") or [r.get("expected_outcome")]
        if (r.get("outcome") in allowed
                and not r.get("sources")
                and any(c in (r.get("answer") or "") for c in clues)):
            num += 1
    return _result(num / max(denom, 1) if denom else None, denom, num)


def backfill_correct_rate(records: list[dict]):
    """参数回填正确率：回填数值与历史陈述一致（expect_answer_all 全部命中）且
    含参数来源说明的占比。"""
    denom, num = 0, 0
    for r in records:
        if not (_ratable(r) and _has_flag(r, "backfill")):
            continue
        expects = r.get("expect_answer_all") or []
        if not expects:
            continue
        denom += 1
        answer = r.get("answer") or ""
        if all(e in answer for e in expects):
            num += 1
    return _result(num / max(denom, 1) if denom else None, denom, num)


def sentinel_counts(records: list[dict]):
    """哨兵观察计数（不入比率）：误拒答哨兵中实际被判 insufficient 的事件数。"""
    total = refused = 0
    for r in records:
        if _has_flag(r, "sentinel"):
            total += 1
            if r.get("outcome") == "insufficient_evidence":
                refused += 1
    return dict(n_sentinel=total, n_sentinel_refused=refused)


def summarize(records: list[dict]) -> dict:
    """五指标 + 哨兵计数汇总（报告/评分卡消费）。N/A 为 None。"""
    out = {
        "completion_correct_rate": completion_correct_rate(records),
        "constraint_retention_rate": constraint_retention_rate(records),
        "interference_rate": interference_rate(records),
        "degrade_correct_rate": degrade_correct_rate(records),
        "backfill_correct_rate": backfill_correct_rate(records),
    }
    out.update(sentinel_counts(records))
    return out


def _fmt(metric):
    if metric is None:
        return "N/A"
    return f"{metric['value']:.4f} ({metric['n_num']}/{metric['n_denom']})"


def main(argv=None):
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(prog="python -m acceptance.context_metrics",
                                 description="上下文工程五指标（从历史/当前 scenario run JSON 重算）")
    ap.add_argument("files", nargs="+", help="scenario run JSON 路径")
    args = ap.parse_args(argv)
    for path in args.files:
        records = json.loads(Path(path).read_text(encoding="utf-8"))["records"]
        s = summarize(records)
        print(f"== {path}")
        for k, v in s.items():
            print(f"   {k}: {_fmt(v) if isinstance(v, dict) else v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
