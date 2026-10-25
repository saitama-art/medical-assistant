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


def _keywords(sentence: str) -> set[str]:
    """从句中抽取候选检索关键词（连续中文/字母数字子串，2~6 字），剔除通用词。
    用于弥补纯 n-gram 字面重合的不足——FR-01 终审实证：「质子泵」「次水杨酸铋」
    「50%」这类关键词命中能定位到 n-gram 匹配不上的正确证据段。"""
    kws: set[str] = set()
    for tok in re.findall(r"[A-Za-z0-9%]+|[一-鿿]{2,}", sentence):
        if re.fullmatch(r"[A-Za-z0-9%]+", tok):
            if len(tok) >= 2:
                kws.add(tok)
        else:
            for n in (6, 4, 3, 2):
                for i in range(0, max(0, len(tok) - n + 1)):
                    kws.add(tok[i:i + n])
    stop = {"患者", "可以", "需要", "如果", "通常", "治疗", "应该", "注意", "一种",
            "可能", "哪些", "进行", "使用", "建议", "常见", "情况", "为主", "以上",
            "以下", "这个", "就是", "以及", "或者", "主要", "一般", "方面", "相关"}
    return {k for k in kws if k not in stop}


def best_chunk(sentence: str, chunks):
    """返回最多 2 个候选 (chunk, 说明) ——关键词命中数优先、n-gram 重合度次序。

    旧版只按 n-gram 字面重合选 1 个，匹配不上就写"无匹配片段"，会把用户误导成
    "无证据"。现在：关键词命中>0 的给 top-2；命中为 0 也不写"无证据"，而是
    提示"请人工核对 top-10 原文"。
    """
    s_ngrams = _ngrams(sentence, 8)
    kws = _keywords(sentence)
    # IDF-lite：出现在半数以上片段里的词（如疾病名本身）没有定位价值，剔除，
    # 只留"有区分度"的关键词——否则每个片段都命中几十个通用词，候选还是定位到
    # 文档开头而非答案段。
    df = {k: sum(1 for c in chunks if k in c.content) for k in kws}
    specific = {k for k in kws if df[k] <= max(1, len(chunks) // 2)}
    scored = []
    for c in chunks:
        hits = sum(1 for k in specific if k in c.content)
        overlap = (len(s_ngrams & _ngrams(c.content, 8)) / len(s_ngrams)) if s_ngrams else 0.0
        scored.append((hits, overlap, c))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = [(c, f"区分词{h}个/重合{ov:.0%}") for h, ov, c in scored[:2] if h > 0 or ov > 0]
    return top


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
            candidates = best_chunk(sent, r.chunks)
            if not candidates:
                evidence = "未定位直接片段，请人工核对 top-10 原文（不等于无证据）"
            else:
                parts = []
                for chunk, note in candidates:
                    doc = os.path.basename(chunk.document)[:26]
                    excerpt = chunk.content[:90].replace("\n", " ").replace("|", "／")
                    parts.append(f"{doc}（{note}）：{excerpt}…")
                evidence = " ／ ".join(parts)
            lines.append(f"| {n} | {sent} | {evidence} | ☐✅ ☐❌ ☐➖ |")
    out = BACKEND.parent / "Evaluation" / "reports" / datetime.now().strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fr01_worksheet.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("工作表已生成:", path)


if __name__ == "__main__":
    main()
