"""病例分析 Agent 的结构化输出模型与确定性文案组装。

结果模型明确区分四类内容（任务书 §二.8）：
- 观察到的事实（facts/timeline/observations/medications，原样来自 service）；
- 候选关联（associations，确定性草稿或落库记录，均非因果结论）；
- 知识依据（knowledge，知识子 Agent 的回答与来源，不是该患者的事实）；
- 待医生核实的问题（doctor_questions）。

本模块只做数据定义与确定性文案渲染，不做 IO、不调用 LLM。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..case_analysis.models import (
    AssociationDraft,
    CandidateAssociation,
    ComparisonEntry,
    TimelineEntry,
)
from ..medical_agent import respond
from ..medical_agent.state import Source, TraceRecord
from .context import AnalysisContext

# 输出中禁止出现的指令性表述：命中即视为模型越界（叙述校验使用，
# 也作为确定性文案组装的自检词表）。宁可误拦，不放行疑似医嘱。
BANNED_DIRECTIVE_PHRASES: tuple[str, ...] = (
    "确诊", "诊断为", "建议停药", "建议停用", "自行停药", "自行加量",
    "自行减量", "处方如下", "建议处方", "应该服用", "务必服用",
)


class CaseAnalysisStatus(str, Enum):
    """分析请求的最终状态。所有终止路径都必须落到其中一个值。"""

    COMPLETED = "completed"                    # 全部环节成功
    PARTIAL = "partial"                        # 部分环节降级，结构化结果仍可用
    NO_HISTORY = "no_history"                  # 患者存在但无任何历史数据
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # 有历史但无可分析内容
    ACCESS_DENIED = "access_denied"            # 患者不存在或不属于调用者（不区分）
    VALIDATION_FAILED = "validation_failed"    # 入参非法，未开始分析
    EMERGENCY_HANDLED = "emergency_handled"    # 关注点命中急症规则，直接拦截
    BUDGET_EXHAUSTED = "budget_exhausted"      # 全局预算耗尽
    PROCESSING_FAILED = "processing_failed"    # 存储/内部故障，无法完成


@dataclass(frozen=True)
class KnowledgeReference:
    """一条知识子 Agent 的回答。answer 是通识医学知识，不是该患者的事实。"""

    question: str
    status: str                                # ok / 机器可读失败标识
    answer: str = ""
    sources: tuple[Source, ...] = ()
    related_fact_ids: tuple[str, ...] = ()     # 触发该问题的事实/事件 id
    detail: str = ""                           # 失败时的机器标识


@dataclass(frozen=True)
class DoctorQuestion:
    """一条待医生核实的问题。全部由确定性规则推出，模型不得增删。"""

    topic: str
    detail: str
    related_ids: tuple[str, ...] = ()


@dataclass
class CaseAnalysisResult:
    """单次病例分析的完整结果。结构化字段优先于 answer_text 被消费。"""

    status: CaseAnalysisStatus
    request_id: str
    patient_id: str
    run_id: str | None = None                  # None = 运行登记失败（降级）
    time_range_start: float | None = None
    time_range_end: float | None = None

    # —— 观察到的事实（原样来自 service，未经模型改写） ——
    context: AnalysisContext | None = None
    timeline: tuple[TimelineEntry, ...] = ()
    comparisons: tuple[ComparisonEntry, ...] = ()

    # —— 候选关联（待核实线索，非因果） ——
    associations: tuple[CandidateAssociation, ...] = ()   # 已落库
    association_drafts: tuple[AssociationDraft, ...] = ()  # 未落库（降级时）

    # —— 知识依据与待核实问题 ——
    knowledge: tuple[KnowledgeReference, ...] = ()
    doctor_questions: tuple[DoctorQuestion, ...] = ()

    # —— 叙述（可选，LLM 生成且通过程序校验；非权威补充，不进 answer_text ——
    # —— 程序无法核对数值/否定/时间关系的一致性，见 narrative.py 校验边界声明） ——
    narrative: str = ""

    # —— 用户可读汇总（确定性组装，不依赖 LLM） ——
    answer_text: str = ""

    # —— 降级与错误 ——
    degraded: tuple[str, ...] = ()             # 机器可读的降级环节标识
    error_code: str = ""
    trace: tuple[TraceRecord, ...] = ()


def render_answer_text(result: CaseAnalysisResult) -> str:
    """把结构化结果确定性渲染为医生可读的汇总文本。

    全部内容来自结构化字段；渲染不含模型自由文本（narrative 单独成段并
    显式标注其性质）。任何渲染失败都不允许发生——数据缺失只表现为空段。
    """
    ctx = result.context
    lines: list[str] = [
        "## 病例纵向分析（辅助医生复核，不构成诊断）",
        "",
        f"- 状态：{result.status.value}"
        + (f"（降级环节：{'、'.join(result.degraded)}）" if result.degraded else ""),
        f"- 分析运行：{result.run_id or '未登记'}",
    ]

    if ctx is not None:
        lines.append(
            f"- 数据概览：就诊 {ctx.encounter_count} 次、事实 {ctx.fact_count} 条"
            f"（待审核 {ctx.pending_fact_count}、含否定语义 {ctx.negated_fact_count}、"
            f"不确定性 {ctx.uncertain_fact_count}）、用药 {ctx.medication_count} 条、"
            f"观察 {ctx.observation_count} 条；发生时间未知的条目 "
            f"{ctx.unknown_time_count} 条"
        )

    if result.comparisons:
        lines += ["", "### 指标变化（确定性比较）"]
        for comp in result.comparisons:
            if comp.comparable and comp.earlier and comp.later:
                lines.append(
                    f"- {comp.indicator}（{comp.unit or '无单位'}）："
                    f"{_fmt_point(comp.earlier)} → {_fmt_point(comp.later)}"
                    + (f"，变化 {comp.delta:+.4g}" if comp.delta is not None else "")
                )
            else:
                lines.append(
                    f"- {comp.indicator}：不可比较（{comp.reason or '原因未知'}）"
                )

    assoc = result.associations or ()
    drafts = result.association_drafts or ()
    if assoc or drafts:
        lines += ["", "### 候选关联（时间接近 ≠ 因果，仅供医生核实）"]
        for item in assoc:
            lines.append(
                f"- [{item.association_type.value}] {item.explanation or '无说明'}"
                f"（时间关系：{item.temporal_relationship.value}）"
            )
        for draft in drafts:
            lines.append(
                f"- [{draft.association_type.value}] {draft.explanation or '无说明'}"
                f"（时间关系：{draft.temporal_relationship.value}，未落库）"
            )

    if result.knowledge:
        lines += ["", "### 知识依据（通识医学知识，非该患者事实）"]
        for ref in result.knowledge:
            if ref.status == "ok":
                src = "；".join(
                    s.document for s in ref.sources if s.document
                )
                lines.append(f"- 问：{ref.question}")
                lines.append(f"  答：{ref.answer}")
                if src:
                    lines.append(f"  来源：{src}")
            else:
                lines.append(f"- 问：{ref.question}（未获解答：{ref.detail or ref.status}）")

    if result.doctor_questions:
        lines += ["", "### 待医生核实"]
        for q in result.doctor_questions:
            lines.append(f"- 【{q.topic}】{q.detail}")

    # 模型叙述不进入正式结果：程序无法可靠核对其中数值/否定/时间关系与
    # 事实的一致性（narrative.py 校验边界）。answer_text 保持纯确定性内容，
    # narrative 作为非权威补充单独存放在结果字段中，展示方须自行加标注。
    if result.narrative:
        lines.append("")
        lines.append(
            "附：本次分析另有模型生成的叙述文本（结果 narrative 字段），"
            "未完成逐项事实一致性核对，不纳入本汇总，仅供医生参考。"
        )

    if result.status is CaseAnalysisStatus.EMERGENCY_HANDLED:
        lines = [respond.EMERGENCY_ANSWER]
    elif result.status is CaseAnalysisStatus.NO_HISTORY:
        lines.append("")
        lines.append("该患者暂无任何就诊、事实、用药或观察记录，无法分析。")
    elif result.status is CaseAnalysisStatus.INSUFFICIENT_EVIDENCE:
        lines.append("")
        lines.append("已有历史记录，但没有可比较的指标序列或时间关联线索。")

    lines += ["", respond.DISCLAIMER]
    return "\n".join(lines)


def _fmt_point(point) -> str:
    value = (
        f"{point.value_numeric:g}{point.unit or ''}"
        if point.value_numeric is not None
        else (point.value_text or "未知值")
    )
    flag = point.abnormal_flag.value if point.abnormal_flag else "unknown"
    return f"{value}（{flag}）"


__all__ = [
    "BANNED_DIRECTIVE_PHRASES",
    "CaseAnalysisResult",
    "CaseAnalysisStatus",
    "DoctorQuestion",
    "KnowledgeReference",
    "render_answer_text",
]
