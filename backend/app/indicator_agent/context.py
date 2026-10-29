"""健康指标输入的标准化：结构化输入直通 + 自由文本 LLM 抽取（带校验）。

铁律（任务书 §三/§五）：
- 缺失的单位、参考范围、时间一律保持 None/空串，**不得由模型或代码猜测补齐**；
- 抽取结果必须经程序校验（类型/字段/数值范围），非法输出走既有 repair 一次；
- 原始表述（数值原文、检测条件原文）保留，渲染时逐字引用；
- 单位不做任何换算，跨单位比较一律判不可比。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..medical_agent.state import NodeResult, NodeStatus, exception_result
from ..rag.service import MAX_QUESTION_LEN


@dataclass(frozen=True)
class IndicatorInput:
    """一项指标的标准化输入。未知字段一律 None/空串，不猜。"""

    name: str
    value_text: str = ""
    value_numeric: float | None = None
    unit: str = ""
    reference_low: float | None = None
    reference_high: float | None = None
    sample_time: str = ""         # 检测时间原文（如"上周体检"），不解析
    note: str = ""                # 检测条件原文（如"空腹"）


@dataclass(frozen=True)
class IndicatorPayload:
    """一次解释请求的完整输入：结构化指标 + 用户追问（可为空）。"""

    indicators: tuple[IndicatorInput, ...]
    question: str = ""            # 用户的关注点原文（如"这个严重吗"）


# ==================== 自由文本抽取（LLM，一次 + repair 一次） ====================

EXTRACT_SYSTEM_PROMPT = """你是检验指标信息抽取器。从用户文本中抽取检验指标的结构化信息。

要求：
1. 只输出 JSON 对象，不要解释、前后缀或 markdown 代码块；
2. 只抽取文本中**明确写出**的信息；文本没有给出的字段一律输出 null 或空字符串，
   不得根据常识、经验或其他指标推测补齐（例如不得自行填参考范围）；
3. value_numeric 只在文本给出阿拉伯数字数值时填写；value_text 保留数值原文
   （含"++"、"阴性"等文本结果）；
4. reference_low/reference_high 只在文本明确给出参考范围时填写；
5. sample_time 保留时间原文（如"上周""3月份"），不换算；
6. question 抽取用户想问的问题（如有），没有则空字符串；
7. 输出格式：{"indicators": [{"name": "指标名", "value_text": "原文",
   "value_numeric": 数值或null, "unit": "单位或空串",
   "reference_low": 数值或null, "reference_high": 数值或null,
   "sample_time": "时间原文", "note": "检测条件原文"}], "question": ""}"""

EXTRACT_USER_PROMPT = "用户文本：{text}"

EXTRACT_REPAIR_PROMPT = """

上一次输出未通过结构校验。请只重新输出符合上述 Schema 的合法 JSON 对象，
不得解释，不得猜测文本中没有的信息。"""

_MAX_INDICATORS = 20
_NAME_MAX = 40
_TEXT_MAX = 200


def extract_indicators(text: str, *, llm, repair: bool = False) -> NodeResult[IndicatorPayload]:
    """从自由文本抽取指标；非法输出由调用方走一次 repair。"""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACT_SYSTEM_PROMPT + (EXTRACT_REPAIR_PROMPT if repair else "")),
        ("human", EXTRACT_USER_PROMPT),
    ])
    try:
        response = llm.invoke(prompt.format_messages(text=text[:MAX_QUESTION_LEN]))
    except Exception as exc:
        return exception_result(exc)

    content = getattr(response, "content", None)
    if not isinstance(content, str):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="content_not_text")
    try:
        payload = _parse_payload(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="extract_invalid")
    return NodeResult(NodeStatus.OK, payload)


def _parse_payload(text: str) -> IndicatorPayload:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no_json")
    raw = json.loads(match.group(0))
    if not isinstance(raw, dict):
        raise ValueError("not_object")
    items = raw.get("indicators")
    if not isinstance(items, list) or len(items) > _MAX_INDICATORS:
        raise ValueError("indicators_invalid")
    question = raw.get("question", "")
    if not isinstance(question, str):
        raise ValueError("question_invalid")

    indicators: list[IndicatorInput] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("indicator_not_object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > _NAME_MAX:
            raise ValueError("name_invalid")
        indicators.append(IndicatorInput(
            name=name.strip(),
            value_text=_opt_text(item.get("value_text")),
            value_numeric=_opt_num(item.get("value_numeric")),
            unit=_opt_text(item.get("unit")),
            reference_low=_opt_num(item.get("reference_low")),
            reference_high=_opt_num(item.get("reference_high")),
            sample_time=_opt_text(item.get("sample_time")),
            note=_opt_text(item.get("note")),
        ))
    return IndicatorPayload(tuple(indicators), question.strip())


def _opt_text(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > _TEXT_MAX:
        raise ValueError("text_field_invalid")
    return value.strip()


def _opt_num(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric_field_invalid")
    return float(value)


__all__ = ["IndicatorInput", "IndicatorPayload", "extract_indicators"]
