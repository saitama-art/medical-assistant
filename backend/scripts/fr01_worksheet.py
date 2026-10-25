"""FR-01 人工抽查工作表生成器（AI 初筛版，供用户终审）。

输出：FR-01-4（句数最多）与 FR-01-8（含源站错字样本）的逐句核对表——
每句医学陈述附最可能的支持片段（n-gram 重合度最高的 chunk），
用户只需逐句打 ✅/❌/➖（非医学句）。支持率≥0.85 为 FR-01 通过线。

运行（cwd=backend/，真实 LLM 8 次回答调用）：
    G:\\annacoda\\envs\\ai_agent\\python.exe -I scripts/fr01_worksheet.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from acceptance.fr01 import FR01_SAMPLES  # noqa: E402
from acceptance.runner import ClientCtx, _login_acceptance_user  # noqa: E402
from acceptance.support import _ngrams  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from app.rag.main import app  # noqa: E402
from app.rag import service  # noqa: E402

TARGETS = (4, 8)  # FR-01-4 / FR-01-8


def split_sentences(answer: str) -> list[str]:
    return [s.strip() for s in re.split(r"[。！？!?；\n]", answer) if len(s.strip()) >= 12]


def best_chunk(sentence: str, chunks):
    s_ngrams = _ngrams(sentence, 8)
    if not s_ngrams:
        return None, 0.0
    best, best_score = None, 0.0
    for c in chunks:
        overlap = len(s_ngrams & _ngrams(c.content, 8)) / len(s_ngrams)
        if overlap > best_score:
            best, best_score = c, overlap
    return best, best_score


def main() -> None:
    client = TestClient(app, base_url="https://testserver")
    _login_acceptance_user(client)
    ctx = ClientCtx(client)
    lines = ["# FR-01 人工抽查工作表（AI 初筛版，终审人：用户本人）",
             f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}",
             "- 用法：逐句核对候选证据，打 ✅支持 / ❌不支持 / ➖非医学句（不计分母）。",
             "- 通过线：支持句数 ÷ 医学陈述句总数 ≥ 0.85。",
             "- 注意：候选证据是 AI 按 n-gram 重合度初筛的定位建议，**判定以你阅读为准**；"
             "FR-01-8 的源站错字（甲哨唑/克林霉素→克拉霉素）属源文错误，记支持并单独转评审。\n"]
    for idx, (question, _kw) in enumerate(FR01_SAMPLES, 1):
        if idx not in TARGETS:
            continue
        code, body, _ms = ctx.chat(f"fr01-ws-{idx}", question)
        answer = body.get("answer") or ""
        r = service.retrieve_with_scores(question)
        lines.append(f"\n---\n\n## FR-01-{idx}：{question}\n")
        lines.append(f"- outcome={body.get('outcome')}，检索 {len(r.chunks)} 段\n")
        lines.append("| # | 回答中的医学陈述句 | 候选证据（AI 初筛） | 判定 |")
        lines.append("|---|---|---|---|")
        for n, sent in enumerate(split_sentences(answer), 1):
            chunk, score = best_chunk(sent, r.chunks)
            if chunk is None:
                evidence = "（无匹配片段）"
            else:
                doc = os.path.basename(chunk.document)[:30]
                excerpt = chunk.content[:110].replace("\n", " ").replace("|", "／")
                evidence = f"{doc}（重合 {score:.0%}）：{excerpt}…"
            lines.append(f"| {n} | {sent} | {evidence} | ☐✅ ☐❌ ☐➖ |")
    out = BACKEND.parent / "Evaluation" / "reports" / datetime.now().strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fr01_worksheet.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("工作表已生成:", path)


if __name__ == "__main__":
    main()
