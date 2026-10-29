"""可选的 LLM 叙述生成与程序校验（任务书 §二.7）。

叙述只是给医生看的"导读"，结构化结果不依赖它：生成失败、输出非法、
校验不过都只导致 narrative 为空并记录降级，绝不影响事实段。

校验是程序性的、先于对模型的信任：
1. 必须是合法 JSON 且字段类型齐全；
2. referenced_ids 必须全部存在于本患者上下文（known_ids ∪ 关联 id），
   任何陌生 id 即视为编造——这也同时校验了患者归属（known_ids 只来自
   归属校验通过的 service 读取）；
3. 叙述文本中出现的每个 YYYY-MM-DD 日期必须出现在上下文允许日期集里
   （由真实发生/录入时间预先格式化而来），模型不得写出新日期；
4. 不得出现指令性表述（确诊/处方/停药建议等）或因果表述（导致/引起等），
   命中即整体作废。

已知校验边界（诚实声明，第二轮审查 M1）：程序无法可靠核对叙述中的
**数值是否与上下文一致**、**否定语义是否被反转**、**时间关系是否被改写**——
这三项靠提示词约束 + 上述兜底，不构成事实一致性证明。因此本模块的产出
一律视为**非权威补充**：不进入 answer_text 正式结果（render 只放确定性
内容），调用方若展示必须带非权威标注。
"""

from __future__ import annotations

import json
import re
import time

from ..medical_agent.state import NodeResult, NodeStatus, exception_result
from .context import AnalysisContext
from .outputs import BANNED_DIRECTIVE_PHRASES

# 因果表述词表：命中即整体作废。这只是兜底防线，不等于"事实一致性验证"
# （见模块 docstring 的校验边界声明）。
CAUSAL_PHRASES: tuple[str, ...] = (
    "导致", "引起", "所致", "诱发", "因此发病", "由此引发",
)

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MAX_NARRATIVE_CHARS = 4000

SYSTEM_PROMPT = """你是病例分析结果的整理员。给你一份某患者的结构化分析上下文（JSON），
请输出一段供医生快速浏览的叙述。

硬性要求：
1. 只输出 JSON 对象，不要解释、前后缀或 markdown 代码块；
2. 只能复述上下文中存在的事实；上下文中没有的信息一律不得补充；
3. 所有时间必须使用上下文中已出现的 YYYY-MM-DD 日期；时间为 null 的条目
   只能说"时间未知"，不得推测或编造日期；
4. 时间上的先后/接近不等于因果，不得使用"导致""引起""因此发病"等因果表述；
5. 不得给出诊断结论、处方或治疗建议；
6. 否定语义（"无/未/否认"）必须原样保留，不得反转；
7. 输出格式：{"summary": "一段总述", "highlights": ["要点1", "要点2"],
   "caveats": ["不确定性或缺失1"], "referenced_ids": ["叙述依据的条目id"]}"""

USER_PROMPT = """【患者结构化分析上下文】
{context}

请输出符合要求的 JSON。"""


def generate_narrative(
    context: AnalysisContext,
    comparisons,
    drafts,
    knowledge_refs,
    *,
    focus_question: str = "",
    llm,
) -> NodeResult[str]:
    """生成并校验叙述；任何一步不合格都返回非 OK 的 NodeResult。"""
    try:
        payload = _build_prompt_payload(
            context, comparisons, drafts, knowledge_refs, focus_question,
        )
        response = llm.invoke([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT.format(context=payload)),
        ])
    except Exception as exc:
        return exception_result(exc)

    content = getattr(response, "content", None)
    if not isinstance(content, str):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="content_not_text")
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="json_invalid")
    try:
        data = json.loads(match.group(0))
    except (TypeError, ValueError):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="json_invalid")

    return validate_narrative(data, context)


def validate_narrative(data, context: AnalysisContext,
                       extra_ids: frozenset[str] = frozenset()) -> NodeResult[str]:
    """程序校验。通过时 value 为渲染后的叙述纯文本。"""
    if not isinstance(data, dict):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="payload_not_object")
    summary = data.get("summary")
    highlights = data.get("highlights")
    caveats = data.get("caveats")
    ref_ids = data.get("referenced_ids")
    if not isinstance(summary, str) or not summary.strip():
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="summary_invalid")
    if not isinstance(highlights, list) or not all(
        isinstance(h, str) for h in highlights
    ):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="highlights_invalid")
    if not isinstance(caveats, list) or not all(
        isinstance(c, str) for c in caveats
    ):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="caveats_invalid")
    if not isinstance(ref_ids, list) or not all(
        isinstance(r, str) for r in ref_ids
    ):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="referenced_ids_invalid")

    allowed_ids = context.known_ids | extra_ids
    if any(r not in allowed_ids for r in ref_ids):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="unknown_reference")

    texts = [summary, *highlights, *caveats]
    joined = "\n".join(texts)
    if len(joined) > _MAX_NARRATIVE_CHARS:
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="narrative_too_long")
    for date in _DATE_RE.findall(joined):
        if date not in context.allowed_dates:
            return NodeResult(NodeStatus.INVALID_OUTPUT, detail="unknown_date")
    if any(p in joined for p in BANNED_DIRECTIVE_PHRASES):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="directive_phrase")
    if any(p in joined for p in CAUSAL_PHRASES):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="causal_phrase")

    parts = [summary.strip()]
    if highlights:
        parts.append("要点：\n" + "\n".join(f"- {h.strip()}" for h in highlights if h.strip()))
    if caveats:
        parts.append("不确定性与缺失：\n" + "\n".join(
            f"- {c.strip()}" for c in caveats if c.strip()
        ))
    return NodeResult(NodeStatus.OK, "\n".join(parts))


def _build_prompt_payload(
    context: AnalysisContext,
    comparisons,
    drafts,
    knowledge_refs,
    focus_question: str,
) -> str:
    """组装给模型的只读上下文：时间预格式化为日期或"未知"，模型永远不需要
    也不会换算 epoch——它写不出上下文之外的合法日期。"""
    def _d(ts: float | None) -> str:
        return (
            time.strftime("%Y-%m-%d", time.localtime(ts))
            if ts is not None else "未知"
        )

    payload = {
        "focus": focus_question or "",
        "facts": [
            {
                "id": f.fact_id, "type": f.fact_type.value, "name": f.display_name,
                "polarity": f.polarity.value, "certainty": f.certainty.value,
                "review_status": f.review_status.value,
                "onset": _d(f.onset_time), "end": _d(f.end_time),
                "recorded": _d(f.recorded_at),
            }
            for f in context.facts
        ],
        "medications": [
            {
                "id": m.medication_event_id, "name": m.medication_name,
                "event_type": m.event_type.value,
                "start": _d(m.start_time), "end": _d(m.end_time),
                "review_status": m.review_status.value,
            }
            for m in context.medications
        ],
        "comparisons": [
            {
                "indicator": c.indicator, "unit": c.unit,
                "comparable": c.comparable, "delta": c.delta,
                "reason": c.reason,
            }
            for c in comparisons
        ],
        "candidate_associations": [
            {
                "type": d.association_type.value,
                "temporal_relationship": d.temporal_relationship.value,
                "explanation": d.explanation,
                "fact_ids": list(d.fact_ids),
                "missing_information": list(d.missing_information),
            }
            for d in drafts
        ],
        "knowledge": [
            {"question": k.question, "answer": k.answer}
            for k in knowledge_refs if k.status == "ok"
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


__all__ = ["generate_narrative", "validate_narrative"]
