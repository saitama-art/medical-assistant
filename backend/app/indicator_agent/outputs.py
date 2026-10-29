"""健康指标解释 Agent 的输出模型与确定性文案渲染。

协议对齐 `case_analysis_agent.outputs`（2026-10-28 实读确认）：
- status 语义五类（completed/partial/needs_input/refused/processing_failed，
  另有 emergency_handled 透传安全门）；
- result 区分客观数据（指标判读）、医学知识（evidence，来自知识子 Agent）、
  不确定性（uncertainties）与待补充信息（follow_up_questions）；
- 渲染是确定性的，不含模型自由文本；知识回答原样引用并标注其性质。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..case_analysis_agent.outputs import KnowledgeReference
from ..medical_agent.state import Source, TraceRecord


class IndicatorStatus(str, Enum):
    """指标解释请求的最终状态。所有终止路径都必须落到其中一个值。"""

    COMPLETED = "completed"                      # 判读+证据完整
    PARTIAL = "partial"                          # 部分环节降级，结果仍可用
    NEEDS_INPUT = "needs_input"                  # 指标/单位/范围缺失，需用户补充
    REFUSED = "refused"                          # 输入超出本 Agent 职责边界
    EMERGENCY_HANDLED = "emergency_handled"      # 命中安全门，直接拦截
    BUDGET_EXHAUSTED = "budget_exhausted"
    PROCESSING_FAILED = "processing_failed"


class RangeFlag(str, Enum):
    """相对参考范围的位置。仅当用户给出的参考范围完整时才判定。"""

    BELOW = "below"
    WITHIN = "within"
    ABOVE = "above"
    UNKNOWN = "unknown"          # 参考范围缺失/不完整，不套用通用值


@dataclass(frozen=True)
class IndicatorReading:
    """一项指标的确定性判读（客观数据部分）。"""

    name: str
    value_text: str               # 原样保留用户输入的数值/文本
    value_numeric: float | None
    unit: str                     # 原样保留，不做换算
    reference_low: float | None
    reference_high: float | None
    flag: RangeFlag
    sample_time: str = ""         # 用户提供的检测时间原文（不解析猜测）
    note: str = ""                # 检测条件原文（如空腹/餐后），仅原样引用


@dataclass(frozen=True)
class TrendNote:
    """同一指标多次测量（来自调用方结构化输入）的确定性趋势描述。"""

    name: str
    unit: str
    points: tuple[str, ...]       # 形如 "2026-01: 7.2 mmol/L"
    delta_text: str               # 形如 "较上次 -0.7 mmol/L"；不可比时为空
    comparable: bool
    reason: str = ""              # 不可比原因（unit_mismatch/value_missing 等）


@dataclass
class IndicatorResult:
    """健康指标解释的完整结果。结构化字段优先于 answer_text 被消费。"""

    status: IndicatorStatus
    request_id: str

    # —— 客观数据判读（确定性，不含医学解释） ——
    readings: tuple[IndicatorReading, ...] = ()
    trends: tuple[TrendNote, ...] = ()

    # —— 医学证据（知识子 Agent 产出，通识知识，非该患者的事实） ——
    evidence: tuple[KnowledgeReference, ...] = ()

    # —— 不确定性与待补充 ——
    uncertainties: tuple[str, ...] = ()
    follow_up_questions: tuple[str, ...] = ()

    # —— 用户可读汇总（确定性组装） ——
    answer_text: str = ""

    # —— 降级与错误 ——
    degraded: tuple[str, ...] = ()
    error_code: str = ""
    trace: tuple[TraceRecord, ...] = ()


_FLAG_TEXT = {
    RangeFlag.BELOW: "低于参考范围下限",
    RangeFlag.WITHIN: "在参考范围内",
    RangeFlag.ABOVE: "高于参考范围上限",
    RangeFlag.UNKNOWN: "参考范围不完整，无法判断是否异常",
}


def render_answer_text(result: IndicatorResult) -> str:
    """把结构化结果确定性渲染为用户可读文本。

    分段固定：判读（客观数据）→ 趋势 → 证据（标注通识性质）→
    不确定性 → 待补充信息 → 边界声明。任何段落无内容则整段省略。
    """
    lines: list[str] = ["## 健康指标解读（不构成诊断）", ""]

    if result.readings:
        lines.append("### 指标判读")
        for r in result.readings:
            value = r.value_text
            unit = f" {r.unit}" if r.unit else ""
            ref = ""
            if r.reference_low is not None or r.reference_high is not None:
                low = r.reference_low if r.reference_low is not None else "—"
                high = r.reference_high if r.reference_high is not None else "—"
                ref = f"（参考 {low}~{high}）"
            suffix = f"；{r.note}" if r.note else ""
            lines.append(f"- {r.name}：{value}{unit}{ref} —— {_FLAG_TEXT[r.flag]}{suffix}")
        lines.append("")

    if result.trends:
        lines.append("### 变化趋势（仅按所给数据，不代表病情变化）")
        for t in result.trends:
            points = " → ".join(t.points)
            delta = f"；{t.delta_text}" if t.delta_text else ""
            note = delta if t.comparable else f"（不可直接比较：{t.reason}）"
            lines.append(f"- {t.name}：{points}{note}")
        lines.append("")

    if result.evidence:
        lines.append("### 医学知识参考（通识资料，非针对个人病情的结论）")
        for ref in result.evidence:
            if ref.answer:
                lines.append(ref.answer)
                if ref.sources:
                    src = "；".join(
                        f"{s.document} - 第{s.page}页" if s.page else s.document
                        for s in ref.sources[:3]
                    )
                    lines.append(f"（资料：{src}）")
            elif ref.detail:
                lines.append(f"（证据获取未完成：{ref.detail}）")
        lines.append("")

    if result.uncertainties:
        lines.append("### 需要注意的不确定性")
        for u in result.uncertainties:
            lines.append(f"- {u}")
        lines.append("")

    if result.follow_up_questions:
        lines.append("### 建议补充的信息")
        for q in result.follow_up_questions:
            lines.append(f"- {q}")
        lines.append("")

    lines.append("以上解读基于你提供的数据与已收录资料，不能替代医生的判断；"
                 "不同实验室的参考范围可能不同，请以你的检验报告单为准。")
    return "\n".join(lines)


__all__ = [
    "IndicatorReading",
    "IndicatorResult",
    "IndicatorStatus",
    "RangeFlag",
    "TrendNote",
    "render_answer_text",
]
