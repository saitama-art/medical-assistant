"""72 道通用健康问题覆盖测试（A01–L06，2026-10-21 冻结）。

题稿与版本：根目录 testing/datasets/health72.json。

阶段 2 执行入口：逐题真实检索（零 LLM）+ 可选真实链路（--chat，消耗 LLM 额度），
记录每题证据（命中文档/距离/chunk 正文摘录/意图/outcome），供 A~F 人工分类：
  A 覆盖充分 / B 覆盖不完整 / C 检索失败 / D 知识缺失 / E 资料可靠性时效 / F 生成引用问题。

用法（backend/ 下，ai_agent 环境）：
    python -m acceptance.health72               # 检索级
    python -m acceptance.health72 --chat        # 检索级 + 真实链路
    python -m acceptance.health72 --out x.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from .datasets import load_dataset
from .promptmeta import prompt_meta  # S4a：报告带 Prompt/词表指纹

# 数据集中在根目录，保留原常量名以兼容执行器和既有测试。
_DATASET = load_dataset("health72")
HEALTH72 = _DATASET["cases"]
HEALTH72_VERSION = _DATASET["version"]


def run(do_chat: bool) -> dict:
    from .runner import ClientCtx, _safe_record
    from fastapi.testclient import TestClient
    from app.rag.main import app

    client = ClientCtx(TestClient(app))
    records = []
    for case in HEALTH72:
        rec = dict(case)
        t0 = time.monotonic()
        diag = client.retrieve(case["question"])
        top = diag["top"] or []
        # 证据摘录：top3 chunk 正文截断；完整 top10 文档名+距离
        rec["top_docs"] = [{"doc": c["doc"], "distance": round(c["distance"], 4)
                            if c.get("distance") is not None else None}
                           for c in top]
        rec["evidence"] = [
            {"rank": i + 1, "doc": c["doc"],
             "distance": round(c["distance"], 4) if c.get("distance") is not None else None,
             "excerpt": (c.get("content") or "")[:300]}
            for i, c in enumerate(top[:3])
        ]
        rec["category"] = None
        rec["outcome"] = None
        if do_chat:
            code, body, ms = client.chat(f"h72-{case['id']}-{datetime.now():%H%M%S}",
                                         case["question"])
            rec["http"] = code
            rec["category"] = body.get("category")
            rec["outcome"] = body.get("outcome")
            rec["risk_notice"] = bool(body.get("risk_notice"))
            rec["answer"] = (body.get("answer") or "")[:400]
            rec["n_sources"] = len(body.get("sources") or [])
            rec["ms"] = ms
        rec["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        records.append(_safe_record(rec))
        print(case["id"], rec.get("outcome") or "-",
              rec["top_docs"][0]["doc"] if rec["top_docs"] else "EMPTY")
    return dict(group="HEALTH72", frozen=HEALTH72_VERSION, generated_at=datetime.now().isoformat(timespec="seconds"),
                chat=do_chat, n=len(records), records=records,
                promptmeta=prompt_meta())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m acceptance.health72",
                                 description="72 道通用健康问题覆盖测试（阶段 2 证据采集）")
    ap.add_argument("--chat", action="store_true", help="追加真实链路（消耗 LLM 额度）")
    ap.add_argument("--out", help="JSON 报告路径（默认 acceptance_results/health72_<ts>.json）")
    args = ap.parse_args(argv)

    out = args.out
    if not out:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = Path("acceptance_results")
        d.mkdir(exist_ok=True)
        out = str(d / f"health72_{ts}.json")

    report = run(do_chat=args.chat)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成 {report['n']} 题，报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
