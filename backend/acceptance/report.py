"""report.py：消费各 run JSON → 评分卡 JSON + Markdown 报告（评估体系 S4c）。

定位（方案 V2 §四）：报告是**指标的汇总视图**，不是重新执行测试。各 runner
（coverage / health72 / scenarios / 门禁）负责产出 run JSON，本模块只做聚合、
计分与呈现；打分走 S4b `scorecard.compute_scorecard`，不在此重复实现公式。

包含：数据来源表、评分卡（总分/维度/否决/部分口径）、安全红线、检索与上下文
指标明细、组×归因交叉、失败样本 topN 钻取、口径声明页脚。
不含：baseline 任命与逐维度指纹校验（S4d）；人工正确性判定（只留录入入口）。

口径要点：
- 时延只取 records[].ms（端到端）；coverage 的 elapsed_ms 是检索环节耗时，
  不混入同一分布。coverage-all 的 records 在 groups 内，展开后取 ms。
- blocked 题不进时延/成本/任何比率分母，单独计数；分母空 → None（N/A）。
- 答案正确率（accuracy）是**人工口径**，本模块不自动判；未录入 → None，
  评分卡按方案以引用完整率单项出"部分口径"分。
- 成本为估算（traceagg.cost_estimate）；runner 会截断 answer（400 字符），
  故 token 估算是**下限**，页脚必须声明。

用法（backend/ 下，零 LLM）：
    python -m acceptance.report --scenarios <scenarios.json> \
        --coverage <coverage_all.json> [--health72 <health72.json>] \
        [--accuracy 0.85] [--baseline-cost 7000] [--top-n 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from . import context_metrics
from .traceagg import cost_estimate, latency_summary

# 拒答/检索指标纯函数在 Evaluation/metrics.py（与 coverage.py 同一跨目录约定）。
_EVAL_DIR = Path(__file__).resolve().parents[2] / "Evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
import metrics as _metrics  # noqa: E402


# ---------- 输入归一（纯函数，输入已解析的 run JSON）----------

def run_records(rep: dict) -> list[dict]:
    """取 records；coverage-all 展开各组 records。无 records → 空列表（调用方记 warning）。"""
    if isinstance(rep.get("records"), list):
        return rep["records"]
    if isinstance(rep.get("groups"), dict):
        return [r for g in rep["groups"].values() for r in g.get("records", [])]
    return []


def retrieval_metrics(rep: dict | None) -> dict | None:
    """检索指标：coverage-all 取 overall，单组 coverage 取 metrics。"""
    if not rep:
        return None
    return rep.get("overall") or rep.get("metrics")


def safety_from_scenarios(records: list[dict]) -> dict:
    """安全红线输入：急症拦截率 + 正确拒答率（均程序可判）。

    急症拦截 = flags 含 emergency 且非 blocked 的题中，outcome 符合预期且
    sources 为空（0 检索 0 来源）的占比；无此类题 → None（安全数据缺失，
    评分卡判本轮无效，不得以均分替代）。
    """
    emg = [r for r in records
           if "emergency" in (r.get("flags") or []) and not r.get("blocked")]
    n_ok = sum(1 for r in emg
               if r.get("outcome") is not None
               and r.get("outcome") == r.get("expected_outcome")
               and not r.get("sources"))
    return {
        "emergency_rate": round(n_ok / len(emg), 4) if emg else None,
        "n_emergency": len(emg),
        "n_emergency_ok": n_ok,
        "correct_refusal_rate": _metrics.refusal_rates(records)["correct_refusal_rate"],
    }


def _has_citation(rec: dict) -> bool:
    if "sources" in rec:
        return bool(rec.get("sources"))
    if rec.get("n_sources") is not None:
        return rec["n_sources"] > 0
    return _metrics.SOURCE_SECTION_MARK in (rec.get("answer") or "")


def answer_metrics(records: list[dict], accuracy: float | None = None) -> dict:
    """答案质量子指标：引用完整率（自动）+ 回答正确率（人工录入，默认 N/A）。

    分母 = 实际产出答案（outcome=completed 且非 blocked）的题，与总表第三节一致。
    """
    answered = [r for r in records
                if not r.get("blocked") and r.get("outcome") == "completed"]
    cited = sum(1 for r in answered if _has_citation(r))
    return {
        "accuracy": accuracy,
        "citation_completeness": round(cited / len(answered), 4) if answered else None,
        "n_answered": len(answered),
        "n_cited": cited,
        "accuracy_source": "人工录入" if accuracy is not None else "未核验（N/A）",
    }


def latency_cost(records: list[dict]) -> dict:
    """时延/成本：blocked 剔除并计数；ms 非数值不入时延样本。"""
    usable = [r for r in records if not r.get("blocked")]
    ms = [float(r["ms"]) for r in usable
          if isinstance(r.get("ms"), (int, float)) and not isinstance(r.get("ms"), bool)]
    return {
        "latency": latency_summary(ms),
        "cost": cost_estimate(records),
        "n_blocked": len(records) - len(usable),
        "n_records": len(records),
    }


def cross_table(coverage_rep: dict | None) -> dict | None:
    """组×归因交叉：行=组，列=verdict（含成功类），值为题数。"""
    if not coverage_rep:
        return None
    groups = coverage_rep.get("groups")
    if not groups:
        return None
    rows = {key: dict(g.get("summary") or
                      (g.get("metrics") or {}).get("failure_composition") or {})
            for key, g in groups.items()}
    verdicts = sorted({v for row in rows.values() for v in row})
    return {"verdicts": verdicts, "rows": rows}


def failure_samples(coverage_rep: dict | None, scenario_records: list[dict],
                    other_runs: dict[str, list[dict]]) -> dict:
    """失败/观察样本钻取清单。blocked 单列，绝不混入"失败=0"的结论。"""
    retrieval = []
    if coverage_rep:
        for gkey, g in (coverage_rep.get("groups") or {}).items():
            for r in g.get("records", []):
                if r.get("verdict") not in ("RETRIEVAL_OK", "OK"):
                    retrieval.append({
                        "group": gkey, "id": r.get("id"),
                        "question": r.get("question"), "verdict": r.get("verdict"),
                        "hit_rank": r.get("hit_rank"), "top1": r.get("top1"),
                        "expect_docs": r.get("expect_docs"),
                    })
    chain, watch, blocked = [], [], []
    for r in scenario_records:
        item = {"id": r.get("id"), "scenario": r.get("scenario"),
                "outcome": r.get("outcome"), "expected": r.get("expected_outcome"),
                "verdict": r.get("verdict"), "top1": r.get("top1"),
                "flags": r.get("flags")}
        if r.get("blocked"):
            blocked.append(item)
        elif r.get("mode") == "watch" or "sentinel" in (r.get("flags") or []):
            watch.append(item)
        elif r.get("verdict") not in (None, "PASS"):
            chain.append(item)
        elif (r.get("expected_outcome") and r.get("outcome") != r.get("expected_outcome")
              and not r.get("expected_outcome_any")):
            chain.append(item)
    other = []
    emergency = []
    for name, records in other_runs.items():
        for r in records:
            if r.get("blocked"):
                blocked.append({"run": name, "id": r.get("id"),
                                "outcome": r.get("outcome")})
                continue
            item = {"run": name, "id": r.get("id"),
                    "question": r.get("question"),
                    "outcome": r.get("outcome"),
                    "top1": r.get("top1") or
                    ((r.get("top_docs") or [{}])[0].get("doc"))}
            if r.get("outcome") == "emergency_handled":
                # 急症拦截是合规行为（health72 含急症题），不是"未实质回答"失败
                emergency.append(item)
            elif r.get("outcome") not in (None, "completed"):
                other.append(item)
    return {"retrieval": retrieval, "chain": chain, "watch": watch,
            "blocked": blocked, "other": other, "emergency": emergency}


# ---------- 汇总 ----------

def build_report(coverage_rep: dict | None, scenario_rep: dict | None,
                 other_reps: dict[str, dict], accuracy: float | None = None,
                 baseline_cost: float | None = None, scorecard_fn=None) -> dict:
    """聚合 → {sources, safety, retrieval, answer, refusal, context, latency_cost,
    cross, failures, scorecard}。scorecard 由 S4b 计算（不在此重写公式）。

    scorecard_fn：默认取 S4b 的 compute_scorecard；测试可注入桩以隔离聚合逻辑。
    """
    scenario_records = run_records(scenario_rep) if scenario_rep else []
    other_records = {name: run_records(rep) for name, rep in other_reps.items()}
    chain_records = scenario_records + [r for rs in other_records.values() for r in rs]
    retrieval = retrieval_metrics(coverage_rep)
    coverage_records = run_records(coverage_rep) if coverage_rep else []
    lc = latency_cost(chain_records + coverage_records)

    payload = {
        "safety": safety_from_scenarios(scenario_records) if scenario_rep else None,
        "retrieval": retrieval,
        "answer": answer_metrics(chain_records, accuracy),
        "refusal": _metrics.refusal_rates(scenario_records) if scenario_rep else None,
        "latency": lc["latency"],
        "cost": {"actual_cost": lc["cost"]["est_input_tokens"],
                 "baseline_cost": baseline_cost},
    }
    if scorecard_fn is None:
        from .scorecard import compute_scorecard as scorecard_fn  # S4b 接口
    scorecard = scorecard_fn(payload)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "payload": payload,
        "scorecard": scorecard,
        "safety": payload["safety"],
        "retrieval": retrieval,
        "answer": payload["answer"],
        "refusal": payload["refusal"],
        "context": context_metrics.summarize(scenario_records) if scenario_rep else None,
        "latency_cost": lc,
        "cross": cross_table(coverage_rep),
        "failures": failure_samples(coverage_rep, scenario_records, other_records),
        "coverage_groups": {k: (v.get("metrics") or {})
                            for k, v in ((coverage_rep or {}).get("groups") or {}).items()},
        "promptmeta": {name: rep.get("promptmeta")
                       for name, rep in
                       ({"scenarios": scenario_rep, "coverage": coverage_rep}
                        | other_reps).items() if rep},
    }


# ---------- Markdown 呈现（纯函数）----------

def _fmt(v, nd=4):
    return "N/A" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _fmt_metric(m):
    return "N/A" if not m else f"{m['value']:.4f}（{m['n_num']}/{m['n_denom']}）"


def _table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


DIM_NAMES = {"retrieval": "检索", "answer": "答案质量", "refusal": "拒答健康",
             "latency": "时延", "cost": "成本"}


def render_markdown(rep: dict, sources: list[dict], warnings: list[str],
                    top_n: int = 20) -> str:
    sc = rep["scorecard"]
    L = [f"# 评估报告（{rep['generated_at']}）", "",
         "> 固定测试集上的测量；评分卡是指标汇总视图，每个维度可向下钻取到分项与失败样本。", ""]

    L += ["## 一、数据来源", ""]
    L += _table(["套件", "文件", "记录数", "run 生成时间"],
                [[s["suite"], s["file"], s["n"], s.get("generated_at") or "—"]
                 for s in sources])
    for w in warnings:
        L.append(f"\n⚠️ {w}")
    L.append("")

    L += ["## 二、评分卡", ""]
    verdict = "不通过（安全否决/数据缺失）" if not sc["valid"] else (
        "部分口径" if sc["partial"] else "全量口径")
    L += [f"- 总分：**{_fmt(sc['total'], 2)}**（{verdict}）",
          f"- 说明：{sc.get('note') or '—'}", ""]
    if sc.get("dimensions"):
        rows = []
        for key, d in sc["dimensions"].items():
            marks = [k for k in ("capped", "partial_accuracy") if d.get(k)]
            rows.append([DIM_NAMES.get(key, key), _fmt(d["score"], 2), d["weight"],
                         "、".join(marks) or "—"])
        L += _table(["维度", "得分", "权重", "标记"], rows)
    else:
        L.append("无维度出分。")
    L.append("")

    L += ["## 三、安全红线（一票否决）", ""]
    safety = rep.get("safety")
    if safety:
        L += _table(["指标", "值", "口径"], [
            ["急症拦截率", _fmt(safety["emergency_rate"]),
             f"{safety['n_emergency_ok']}/{safety['n_emergency']}（outcome 符合预期且零来源）"],
            ["正确拒答率", _fmt(safety["correct_refusal_rate"]), "拒答题 outcome 符合预期且无来源段"]])
    else:
        L.append("未提供场景集 run：安全数据缺失 → 本轮无效（不得以均分替代）。")
    for reason in (sc.get("safety") or {}).get("reasons", []):
        L.append(f"\n🚨 {reason}")
    L.append("")

    L += ["## 四、指标明细", ""]
    r = rep.get("retrieval")
    if r:
        L += ["### 检索（coverage 固定题集）", ""]
        L += _table(["指标", "值"], [
            ["n_valid / n_gap", f"{r.get('n_valid')} / {r.get('n_gap')}"],
            ["Recall@5", _fmt(r.get("recall@5"))],
            ["Recall@10", _fmt(r.get("recall@10"))],
            ["MRR", _fmt(r.get("mrr"))], ["top1 命中率", _fmt(r.get("top1_rate"))],
            ["覆盖率（含缺口）", _fmt(r.get("coverage_rate"))]])
        if rep.get("coverage_groups"):
            L += ["", "### 分组检索指标", ""]
            L += _table(["组", "n_valid", "R@10", "MRR", "top1"],
                        [[k, m.get("n_valid"), _fmt(m.get("recall@10")),
                          _fmt(m.get("mrr")), _fmt(m.get("top1_rate"))]
                         for k, m in rep["coverage_groups"].items()])
    else:
        L.append("未提供 coverage run：检索维度 N/A。")
    L.append("")

    a = rep.get("answer") or {}
    L += ["### 答案质量", ""]
    L += _table(["子指标", "值", "口径"], [
        ["回答正确率", _fmt(a.get("accuracy")), a.get("accuracy_source", "—")],
        ["引用完整率", _fmt(a.get("citation_completeness")),
         f"{a.get('n_cited')}/{a.get('n_answered')}（completed 题中带可追溯来源）"]])
    L.append("")

    rf = rep.get("refusal")
    if rf:
        L += ["### 拒答", ""]
        L += _table(["指标", "值"], [
            ["正确拒答率", _fmt(rf["correct_refusal_rate"])],
            ["误拒答率", _fmt(rf["false_refusal_rate"])],
            ["误拒题数", rf["n_false_refusal"]],
            ["哨兵拒答（只观察不扣分）",
             f"{rf['n_sentinel_refused']}/{rf['n_sentinel']}"],
            ["blocked（N/A）", rf["n_na"]]])
        L.append("")
    ctx = rep.get("context")
    if ctx:
        L += ["### 上下文五指标（S2）", ""]
        ctx_names = {"completion_correct_rate": "补全正确率",
                     "constraint_retention_rate": "约束保留率",
                     "backfill_correct_rate": "参数回填正确率",
                     "degrade_correct_rate": "降级正确率",
                     "interference_rate": "历史干扰率（0 为优）"}
        L += _table(["指标", "值"], [[ctx_names[k], _fmt_metric(ctx[k])]
                                     for k in ctx_names])
        L.append("")

    lc = rep["latency_cost"]
    lat, cost = lc["latency"], lc["cost"]
    L += ["### 时延与成本", ""]
    L += _table(["指标", "值"], [
        ["P50 / P95（端到端 ms）",
         f"{_fmt(lat and lat['p50_ms'], 0)} / {_fmt(lat and lat['p95_ms'], 0)}"],
        ["时延分档得分", lat["tier"] if lat else "N/A"],
        ["时延样本数 / blocked 剔除", f"{lat['n'] if lat else 0} / {lc['n_blocked']}"],
        ["每 completed 题 LLM 调用（估算）", _fmt(cost["avg_calls_per_completed"])],
        ["est_input_tokens（估算，下限）", cost["est_input_tokens"]],
        ["成本基线", _fmt((rep["payload"]["cost"] or {}).get("baseline_cost"))]])
    L.append("")

    if rep.get("cross"):
        cross = rep["cross"]
        L += ["## 五、组×归因交叉", ""]
        L += _table(["组"] + cross["verdicts"],
                    [[g] + [row.get(v, 0) for v in cross["verdicts"]]
                     for g, row in cross["rows"].items()])
        L.append("")

    L += ["## 六、失败与观察样本（钻取）", ""]
    f = rep["failures"]
    L += _table(["类别", "数量"], [["检索未命中/缺口", len(f["retrieval"])],
                                  ["链路未达预期", len(f["chain"])],
                                  ["其他 run 未实质回答", len(f["other"])],
                                  ["急症合规拦截（非失败）", len(f["emergency"])],
                                  ["观察题（不入断言）", len(f["watch"])],
                                  ["blocked（N/A，非 0 通过）", len(f["blocked"])]])
    L.append("")
    sample_specs = [
        ("检索失败样本", f["retrieval"], ["group", "id", "verdict", "hit_rank", "top1", "question"]),
        ("链路未达预期样本", f["chain"], ["id", "scenario", "outcome", "expected", "verdict", "top1"]),
        ("其他 run 未实质回答样本", f["other"], ["run", "id", "outcome", "top1", "question"]),
        ("急症合规拦截实录", f["emergency"], ["run", "id", "outcome"]),
        ("观察题实录（哨兵，逐题归因入口）", f["watch"], ["id", "outcome", "expected", "top1"]),
        ("blocked 清单（本轮相关维度无数据，非通过）", f["blocked"], ["id", "run", "outcome"]),
    ]
    for title, items, cols in sample_specs:
        if not items:
            continue
        L += [f"### {title}（前 {min(top_n, len(items))}/{len(items)} 条）", ""]
        L += _table(cols, [[item.get(c) if item.get(c) not in (None, "") else "—"
                            for c in cols] for item in items[:top_n]])
        L.append("")

    L += ["## 七、口径声明", "",
          "- 所有比率均为固定测试集上的测量；分母为空记 N/A（不记 0），blocked 题剔除并单列。",
          "- 回答正确率为人工口径，未录入时答案质量按引用完整率单项出部分口径分。",
          "- 成本为估算：知识路径 3 次调用/completed 题、token≈answer 字符/4；"
          "runner 对 answer 有截断（400 字符），故 token 为下限估算。",
          "- 时延分档（5/8/15/30s）与评分卡权重为方案标注的初值，首轮全量 run 后需校准。",
          "- 基线任命、换基线规则与逐维度条件指纹校验属 S4d，本报告不做跨 run diff。",
          "- 条件指纹（promptmeta）随各 run JSON 自带；S4a 之前的历史 run 无该块，标「未采集」。"]
    L.append("")
    L += _table(["run", "promptmeta"],
                [[k, "已采集" if v else "未采集"] for k, v in rep["promptmeta"].items()])
    cond = rep.get("conditions")
    if cond:
        stats = cond.get("index_stats") or {}
        L.append(f"\n运行条件指纹：已采集（git {cond.get('git_sha')}，"
                 f"索引 docs={stats.get('docs')} chunks={stats.get('chunks')}，"
                 f"model={cond.get('model')}）——供逐轮变更归因与 S4d 可比性校验。")
    else:
        L.append("\n运行条件指纹：未采集（--no-conditions 或采集失败）→ 本轮分数变化无法归因到具体改动。")
    L.append("")
    return "\n".join(L)


# ---------- IO 层 ----------

def _load(path: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as e:
        return None, f"{Path(path).name}: 读取/解析失败（{e}），已跳过"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m acceptance.report",
                                 description="消费 run JSON → 评分卡 JSON + MD 报告（零 LLM）")
    ap.add_argument("--coverage", help="coverage 报告（单组或 --all-groups）")
    ap.add_argument("--scenarios", help="场景集 run JSON（安全/拒答/上下文指标来源）")
    ap.add_argument("--health72", help="health72 run JSON（补充链路与失败样本）")
    ap.add_argument("--accuracy", type=float, default=None,
                    help="人工核验的回答正确率（0~1）；不填 = N/A")
    ap.add_argument("--baseline-cost", type=float, default=None,
                    help="成本基线（est_input_tokens 口径）；不填 = 成本维度 N/A")
    ap.add_argument("--top-n", type=int, default=20, help="失败样本清单最多条数")
    ap.add_argument("--no-conditions", action="store_true",
                    help="不采集运行条件指纹（默认采集，复用 S4d conditions；失败仅警告）")
    ap.add_argument("--chroma-dir", default=None,
                    help="指定 chroma 库路径用于条件指纹（只读）。worktree 里生成报告时传主库，"
                         "例如 F:/medical-assistant/backend/chroma_db；缺省用本目录 backend/chroma_db。"
                         "注意：docs 数仍按本目录 backend/data 统计（跨 worktree 需知悉）")
    ap.add_argument("--out-prefix", help="输出前缀（默认 acceptance_results/report_<ts>）")
    args = ap.parse_args(argv)

    warnings: list[str] = []
    sources: list[dict] = []

    def load(suite, path):
        if not path:
            return None
        rep, err = _load(path)
        if err:
            warnings.append(err)
            return None
        if not run_records(rep) and not rep.get("overall"):
            warnings.append(f"{Path(path).name}: 无 records/overall，仅作来源登记")
        sources.append({"suite": suite, "file": Path(path).name, "n": len(run_records(rep)),
                        "generated_at": rep.get("generated_at")})
        return rep

    coverage_rep = load("coverage", args.coverage)
    scenario_rep = load("scenarios", args.scenarios)
    other = {}
    h72 = load("health72", args.health72)
    if h72:
        other["health72"] = h72
    if not (coverage_rep or scenario_rep or other):
        ap.error("至少提供一个有效 run JSON（--coverage/--scenarios/--health72）")

    rep = build_report(coverage_rep, scenario_rep, other,
                       accuracy=args.accuracy, baseline_cost=args.baseline_cost)
    # 运行条件指纹（S4d conditions）：让本轮"在什么条件下跑的"可机检——
    # 没有它，分数变化无法归因到具体改动。采集失败只降级警告，不影响报告产出。
    rep["conditions"] = None
    if not args.no_conditions:
        try:
            from .conditions import current_conditions
            cond_kwargs = {"chroma_dir": Path(args.chroma_dir)} if args.chroma_dir else {}
            rep["conditions"] = current_conditions(
                sorted({s["suite"] for s in sources}), **cond_kwargs)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"条件指纹采集失败（{type(e).__name__}: {e}）：本轮不可做条件校验")
    rep["sources"] = sources
    rep["warnings"] = warnings
    md = render_markdown(rep, sources, warnings, top_n=args.top_n)

    prefix = args.out_prefix or str(
        Path("acceptance_results") / f"report_{datetime.now():%Y%m%d_%H%M%S}")
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    Path(prefix + ".json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    Path(prefix + ".md").write_text(md, encoding="utf-8")
    sc = rep["scorecard"]
    print(f"总分={sc['total']} valid={sc['valid']} partial={sc['partial']}  ({sc.get('note')})")
    print(f"报告：{prefix}.md / {prefix}.json")
    for w in warnings:
        print(f"WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
