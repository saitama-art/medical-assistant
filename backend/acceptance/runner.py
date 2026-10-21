"""验收执行核心：用例选择、上下文封装、异常归一、报告落盘。

关键语义：
- pass 只能是显式断言为真；任何异常/非 200/结构缺失都记 ok=False 并写 error 原因；
- API 不可用时整条记 status="blocked"（不是 pass，也不是 fail 断言结果）；
- 所有进入记录的字符串都过 redact()。
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .cases import CASE_GENERATORS
from .support import redact


# ---------------------------------------------------------------- 上下文


@dataclass
class ClientCtx:
    """封装真实 TestClient + 真实检索。chat/retrieve 都不抛异常。"""
    client: object
    _seq: int = field(default=0)
    last_error: str | None = None

    def chat(self, sid, q):
        self._seq += 1
        self.last_error = None
        t0 = time.monotonic()
        try:
            r = self.client.post("/agent/chat", json={"session_id": sid, "question": q},
                                 timeout=180)
            ms = int((time.monotonic() - t0) * 1000)
            body = (r.json()
                    if r.headers.get("content-type", "").startswith("application/json")
                    else {})
            if r.status_code != 200:
                self.last_error = f"HTTP {r.status_code}"
            return r.status_code, body, ms
        except Exception as e:  # noqa: BLE001
            ms = int((time.monotonic() - t0) * 1000)
            self.last_error = f"{type(e).__name__}: {e}"
            return None, {}, ms

    def retrieve(self, question):
        """真实索引检索诊断（零 LLM）。惰性导入，单测可用假实现替换。"""
        from app.rag import service
        import os.path
        r = service.retrieve_with_scores(question)
        top = []
        for c in (r.chunks or [])[:10]:
            top.append({
                "doc": os.path.basename(c.document or ""),
                "distance": c.distance,
                "content": c.content,
            })
        return {"outcome": r.outcome.value if r.outcome else None, "top": top}


# ---------------------------------------------------------------- 选择

# 组名（含③⑥符号）→ 实际编号前缀（ASCII）映射，支持 --only T-05-6-1 这类编号选择
GROUP_ID_PREFIX = {
    "T-03": "T-03-", "T-02b": "T-02b-", "T-06③": "T-06-",
    "T-05⑥": "T-05-6-", "T-12": "T-12-", "T-13": "T-13-", "T-14b": "T-14b-", "T-15": "T-15-",
}


def select_cases(groups, only=None, fr01=False):
    """only: None=全部 pass/fail 组；或逗号分隔的组名/编号前缀（如 T-03, T-02b-1）。
    返回 [(meta, fn_or_case)]。fr01=True 时追加 FR-01 诊断样本。"""
    from .cases import fr01_cases  # 延迟导入
    if only:
        keys = [k.strip() for k in only.split(",") if k.strip()]
        picked = [(g, fn, desc) for g, fn, desc in groups
                  if any(k == g or k.startswith(GROUP_ID_PREFIX.get(g, g + "-"))
                         for k in keys)]
    else:
        picked = list(groups)
    out = list(picked)
    if fr01:
        for c in fr01_cases():
            out.append((c["group"], c, "FR-01 诊断样本（不计通过/失败）"))
    return out


# ---------------------------------------------------------------- 执行

def _safe_record(rec):
    return json.loads(redact(json.dumps(rec, ensure_ascii=False, default=str)))


def run_report(only=None, fr01=False, out_path=None, client=None):
    """执行验收并落盘 JSON 报告。client 缺省时起真实 TestClient。
    返回报告 dict（调用方也可直接拿去做进一步汇总）。"""
    if client is None:
        from fastapi.testclient import TestClient
        from app.rag.main import app
        client = TestClient(app)

    ctx = ClientCtx(client)
    results = []
    api_error_seen = None
    id_keys = ([k.strip() for k in only.split(",") if "-" in k.strip()]
               if only else [])

    for g, fn_or_case, desc in select_cases(CASE_GENERATORS, only, fr01):
        if isinstance(fn_or_case, dict) and fn_or_case.get("kind") == "diagnostic":
            c = fn_or_case
            try:
                rec = c["fn"](ctx, c["question"], c["fr01_key"])
                rec = _safe_record(rec)
                rec.update(id=c["id"], group=c["group"], kind="diagnostic")
                rec.setdefault("status", "recorded")
            except Exception as e:  # noqa: BLE001
                rec = dict(id=c["id"], group=c["group"], kind="diagnostic",
                           question=c["question"], status="error",
                           error=f"{type(e).__name__}: {e}")
            results.append(rec)
            print(f"[DIAG] {rec['id']:10s} outcome={rec.get('outcome')}")
            continue

        for rec in fn_or_case(ctx):
            # 编号级选择（如 T-02b-1）：整组仍会执行（多轮用例有前置调用），
            # 但只记录编号匹配的条目
            if id_keys and not any(rec["id"] == k or rec["id"].startswith(k)
                                   for k in id_keys):
                continue
            if ctx.last_error:
                rec["_error"] = ctx.last_error
                ctx.last_error = None
            rec = _safe_record(rec)
            err = rec.get("_error")
            rec["status"] = "blocked" if err else ("pass" if rec.get("ok") else "fail")
            if err:
                rec["error"] = err           # 失败原因入记录（已脱敏）
                del rec["_error"]
                api_error_seen = api_error_seen or err
            results.append(rec)
            print(f"[{'PASS' if rec.get('ok') else 'FAIL':4s}] {rec['id']:24s} "
                  f"{str(rec.get('ms')):>7} outcome={rec.get('outcome')}"
                  + (f"  ERROR={rec.get('error')}" if rec.get("error") else ""))

    pf = [r for r in results if r.get("kind") != "diagnostic"]
    passed = sum(1 for r in pf if r.get("ok"))
    report = {
        "meta": {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "cwd": os.getcwd(),
            "only": only, "fr01": fr01,
            "api_key_configured": bool(os.environ.get("DEEPSEEK_API_KEY")),
        },
        "summary": {
            "total": len(results),
            "pass_fail_cases": len(pf),
            "passed": passed,
            "failed": [r["id"] for r in pf if not r.get("ok")],
            "diagnostic": [r["id"] for r in results if r.get("kind") == "diagnostic"],
            "api_blocked": api_error_seen is not None,
        },
        "results": results,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n== {passed}/{len(pf)} 通过；明细 {out_path}")
    return report
