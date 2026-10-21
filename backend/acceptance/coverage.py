"""知识覆盖测试集：P0 饮食营养与体重管理（任务清单第三节 P0，共 20 题）。

目的不是"通过/失败"，而是**定位失败原因**，每轮扩源后复跑对比：
- 检索级（默认，零 LLM）：期望文档是否进 top-k → RETRIEVAL_OK / RETRIEVAL_MISS
  / KNOWLEDGE_MISSING（该主题确认无来源，记「待补充来源」，不伪造覆盖）；
- 链路级（--chat，真实 LLM）：outcome=completed → OK；insufficient_evidence 再按
  检索是否命中细分 EVIDENCE_INSUFFICIENT（检索到了但判定器拒答）与
  RETRIEVAL_MISS（根本没检索到期望文档）；其余 outcome 原样记录。

用法（backend/ 下，ai_agent 环境）：
    python -m acceptance.coverage              # 检索级，零 LLM
    python -m acceptance.coverage --chat       # 追加真实链路（消耗 LLM 额度）
    python -m acceptance.coverage --out x.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# ---- P0 饮食营养与体重管理测试集（任务清单第三节 P0 1~20）----
# expect_docs: 判定「已覆盖」所期望命中的文档名子串（空列表 = 已知无来源，记缺口）。
COVERAGE_CASES = [
    dict(id="P0-01", question="我想减肥，饮食如何安排？", topic="体重管理",
         expect_docs=["肥胖"]),
    dict(id="P0-02", question="减肥期间早餐、午餐、晚餐应该怎么吃？", topic="体重管理",
         expect_docs=["肥胖"]),
    dict(id="P0-03", question="减肥期间可以吃米饭、面条和馒头吗？", topic="膳食结构",
         expect_docs=["碳水化合物、蛋白质和脂肪", "肥胖"]),
    dict(id="P0-04", question="减肥期间晚上饿了，可以吃东西吗？", topic="体重管理",
         expect_docs=["肥胖"]),
    dict(id="P0-05", question="减肥期间可以吃水果吗？", topic="膳食结构",
         expect_docs=["碳水化合物、蛋白质和脂肪", "肥胖"]),
    dict(id="P0-06", question="减肥期间体重不下降，可能是什么原因？", topic="体重管理",
         expect_docs=["肥胖"]),
    dict(id="P0-07", question="怎样估算每天需要摄入多少能量？", topic="能量",
         expect_docs=["卡路里"]),
    dict(id="P0-08", question="减肥期间如何保证蛋白质、维生素和矿物质摄入？",
         topic="营养素", expect_docs=["碳水化合物、蛋白质和脂肪", "维生素和矿物质"]),
    dict(id="P0-09", question="普通成年人每天应该怎样搭配饮食？", topic="膳食结构",
         expect_docs=["营养概述", "营养需求"]),
    dict(id="P0-10", question="每天应该吃多少蔬菜和水果？", topic="膳食结构",
         expect_docs=["营养需求", "营养概述"]),
    dict(id="P0-11", question="经常吃外卖，怎样搭配相对健康？", topic="膳食结构",
         expect_docs=["营养概述"]),
    dict(id="P0-12", question="每天喝多少水比较合适？", topic="饮水",
         expect_docs=[], note="待补充来源：MSD 大众版无每日饮水量词条"),
    dict(id="P0-13", question="高蛋白饮食适合所有人吗？", topic="营养素",
         expect_docs=["碳水化合物、蛋白质和脂肪", "膳食补充剂概述"]),
    dict(id="P0-14", question="如何阅读食品营养成分表？", topic="膳食结构",
         expect_docs=[], note="待补充来源：MSD 大众版无营养成分表词条"),
    dict(id="P0-15", question="高血压患者日常饮食应该注意什么？", topic="特殊人群饮食",
         expect_docs=["高血压"]),
    dict(id="P0-16", question="血糖偏高的人应该如何安排饮食？", topic="特殊人群饮食",
         expect_docs=["2型糖尿病"]),
    dict(id="P0-17", question="血脂偏高的人应该少吃哪些食物？", topic="特殊人群饮食",
         expect_docs=["血脂异常", "胆固醇及脂代谢紊乱"]),
    dict(id="P0-18", question="尿酸偏高的人饮食应该注意什么？", topic="特殊人群饮食",
         expect_docs=["痛风（高尿酸血症）"]),
    dict(id="P0-19", question="慢性肾病患者如何根据病情安排饮食？", topic="特殊人群饮食",
         expect_docs=["慢性肾病"]),
    dict(id="P0-20", question="老年人怎样保证蛋白质和营养摄入？", topic="特殊人群饮食",
         expect_docs=["营养需求"]),
]

# ---- 失败原因分类（纯函数，可单测） ----


def classify(case: dict, hit_rank: int | None, outcome: str | None) -> str:
    """按检索命中与链路 outcome 给出失败/成功分类。

    hit_rank: 期望文档在 top-k 中的名次（1 起），None=未命中。
    outcome: 链路 outcome（None 表示只跑了检索级）。
    """
    if outcome is None:
        # 检索级判定
        if hit_rank is not None:
            return "RETRIEVAL_OK"
        return ("KNOWLEDGE_MISSING" if not case["expect_docs"]
                else "RETRIEVAL_MISS")
    if outcome == "completed":
        return "OK"
    if outcome == "insufficient_evidence":
        return ("EVIDENCE_INSUFFICIENT" if hit_rank is not None
                else ("KNOWLEDGE_MISSING" if not case["expect_docs"] else "RETRIEVAL_MISS"))
    return f"OUTCOME:{outcome}"


# ---- 执行 ----


def _hit_rank(top: list[dict], expect_docs: list[str]) -> int | None:
    for i, chunk in enumerate(top, 1):
        doc = chunk.get("doc") or ""
        if any(k in doc for k in expect_docs):
            return i
    return None


def run_coverage(do_chat: bool, client_factory=None) -> dict:
    from .runner import ClientCtx, _safe_record

    if client_factory is None:
        from fastapi.testclient import TestClient
        from app.rag.main import app
        client = ClientCtx(TestClient(app))
    else:
        client = client_factory()

    records = []
    for case in COVERAGE_CASES:
        rec = dict(case)
        t0 = time.monotonic()
        diag = client.retrieve(case["question"])
        top = diag["top"] or []
        rank = _hit_rank(top, case["expect_docs"])
        rec["hit_rank"] = rank
        rec["top1"] = top[0]["doc"] if top else None
        rec["outcome"] = None
        if do_chat and case["expect_docs"]:
            sid = f"cov-{case['id']}-{datetime.now():%H%M%S}"
            code, body, ms = client.chat(sid, case["question"])
            rec["outcome"] = body.get("outcome")
            rec["http"] = code
            rec["answer"] = (body.get("answer") or "")[:300]
            rec["ms"] = ms
        rec["verdict"] = classify(case, rank, rec["outcome"])
        rec["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        records.append(_safe_record(rec))
        print(f"{case['id']}  {rec['verdict']:22s} hit={rank} top1={rec['top1']}")
    summary = {}
    for r in records:
        summary[r["verdict"]] = summary.get(r["verdict"], 0) + 1
    return dict(group="COVERAGE-P0", generated_at=datetime.now().isoformat(timespec="seconds"),
                chat=do_chat, summary=summary, records=records)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m acceptance.coverage",
                                 description="P0 饮食营养覆盖测试集（诊断用，不计通过/失败）")
    ap.add_argument("--chat", action="store_true",
                    help="追加真实 LLM 链路判定（默认只跑零 LLM 检索级）")
    ap.add_argument("--out", help="JSON 报告路径（默认 acceptance_results/coverage_<ts>.json）")
    args = ap.parse_args(argv)

    out = args.out
    if not out:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = Path("acceptance_results")
        d.mkdir(exist_ok=True)
        out = str(d / f"coverage_{ts}.json")

    report = run_coverage(do_chat=args.chat)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n分类汇总：{report['summary']}")
    print(f"报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
