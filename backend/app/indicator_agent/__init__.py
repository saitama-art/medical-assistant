"""健康指标解释 Agent：检验指标的结构化判读与证据化解释（独立子 Agent）。

入口：

    from app.indicator_agent import run_indicator_explanation

    result = run_indicator_explanation(
        owner_user_id=1,
        raw_text="空腹血糖 7.2 mmol/L（参考 3.9-6.1），这个高吗？",
    )
    # 或结构化输入：
    result = run_indicator_explanation(
        owner_user_id=1,
        indicators=[IndicatorInput(name="空腹血糖", value_numeric=7.2,
                                   unit="mmol/L", reference_low=3.9,
                                   reference_high=6.1)],
        question="这个高吗",
    )

边界：不诊断、不推荐药物、不下因果结论；证据只来自知识子 Agent（task_spec
授权链）；参考范围缺失时明确说明，不套用通用值。
"""

from .agent import run_indicator_explanation
from .context import IndicatorInput, IndicatorPayload
from .outputs import (
    IndicatorReading,
    IndicatorResult,
    IndicatorStatus,
    RangeFlag,
    TrendNote,
)

__all__ = [
    "IndicatorInput",
    "IndicatorPayload",
    "IndicatorReading",
    "IndicatorResult",
    "IndicatorStatus",
    "RangeFlag",
    "TrendNote",
    "run_indicator_explanation",
]
