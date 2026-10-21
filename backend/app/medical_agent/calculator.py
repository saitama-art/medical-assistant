"""健康指标计算工具：BMI、标准体重、常用单位换算。

确定性计算模块（2026-10-21 功能扩展任务）：计算结果**全部来自本模块的真实函数
执行**，大模型只做意图路由（且路由规则可测、可查），不允许模型心算冒充工具结果。

接入方式：orchestrator 在 N03（紧急判定）之后、N04（LLM 分类）之前调用
`detect()` 做规则短路——
- 未命中计算意图 → 返回 None，走原有 LLM 分类链路（零影响）；
- 参数齐全 → 直接执行计算并组装回答（0 LLM、0 工具），组合问法可拆出
  「剩余知识问题」交给知识子 Agent；
- 参数缺失/异常 → 返回追问信息，由编排器给固定追问文案。

刻意不做的事：
- 不识别中文数字（「一米七五」外的写法）与复合单位（「5尺6寸」）——
  解析不了就交给 LLM 链路或追问，不猜参数；
- BMI 只给成人参考分类，不输出任何诊断结论。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ==================== 常量与参考标准 ====================

# 中国成人 BMI 分类（WS/T 428-2013 成人体重判定），仅成人参考口径。
BMI_CLASSES = (
    (18.5, "偏瘦"),
    (24.0, "正常范围"),
    (28.0, "超重"),
    (float("inf"), "肥胖"),
)

# 成人身高/体重的可接受输入范围（超出视为参数异常而非计算结果）
HEIGHT_CM_MIN, HEIGHT_CM_MAX = 80.0, 250.0
WEIGHT_KG_MIN, WEIGHT_KG_MAX = 2.0, 300.0

# 标准体重粗略公式（常用临床估算）：身高(cm) - 105，±10% 为参考区间
STANDARD_WEIGHT_OFFSET = 105
STANDARD_WEIGHT_TOLERANCE = 0.10

_UNIT_TABLE = {
    # 规范化单位 -> (类别, 转基准倍率)。基准：长度=cm，重量=kg
    "cm": ("length", 1.0), "厘米": ("length", 1.0), "公分": ("length", 1.0),
    "m": ("length", 100.0), "米": ("length", 100.0),
    "kg": ("weight", 1.0), "公斤": ("weight", 1.0), "千克": ("weight", 1.0),
    "斤": ("weight", 0.5),
    "lb": ("weight", 0.45359237), "磅": ("weight", 0.45359237),
    "g": ("weight", 0.001), "克": ("weight", 0.001),
}


# ==================== 计算函数（纯函数，可独立测试） ====================


@dataclass(frozen=True)
class CalcResult:
    """一次确定性计算的产物。error 非空表示参数问题，此时 value 无意义。"""

    kind: str                                   # bmi | standard_weight | convert
    formula: str                                # 展示用公式说明
    value: float | None = None
    unit: str = ""
    extra: dict = field(default_factory=dict)   # 分类、参考区间等
    error: str = ""                             # 机器可读错误标识


def calc_bmi(height_cm: float, weight_kg: float) -> CalcResult:
    """BMI = 体重(kg) / 身高(m)²。返回中国成人参考分类；不输出诊断结论。"""
    if not (HEIGHT_CM_MIN <= height_cm <= HEIGHT_CM_MAX):
        return CalcResult("bmi", "BMI = 体重(kg) ÷ 身高(m)²",
                          error="height_out_of_range")
    if not (WEIGHT_KG_MIN <= weight_kg <= WEIGHT_KG_MAX):
        return CalcResult("bmi", "BMI = 体重(kg) ÷ 身高(m)²",
                          error="weight_out_of_range")
    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m * height_m), 1)
    label = next(name for threshold, name in BMI_CLASSES if bmi < threshold)
    return CalcResult(
        "bmi", "BMI = 体重(kg) ÷ 身高(m)²",
        value=bmi, unit="kg/m²",
        extra={"label": label, "height_cm": height_cm, "weight_kg": weight_kg},
    )


def calc_standard_weight(height_cm: float, sex: str | None = None) -> CalcResult:
    """粗略标准体重估算：身高(cm) - 105（kg），±10% 参考区间。

    成人通用经验公式（非性别区分版）；sex 参数仅为展示说明保留。
    """
    if not (HEIGHT_CM_MIN <= height_cm <= HEIGHT_CM_MAX):
        return CalcResult("standard_weight", "标准体重(kg) ≈ 身高(cm) - 105",
                          error="height_out_of_range")
    base = round(height_cm - STANDARD_WEIGHT_OFFSET, 1)
    low = round(base * (1 - STANDARD_WEIGHT_TOLERANCE), 1)
    high = round(base * (1 + STANDARD_WEIGHT_TOLERANCE), 1)
    return CalcResult(
        "standard_weight", "标准体重(kg) ≈ 身高(cm) - 105",
        value=base, unit="kg",
        extra={"range": (low, high), "height_cm": height_cm,
               "sex": sex or "未指定"},
    )


def convert_unit(value: float, from_unit: str, to_unit: str) -> CalcResult:
    """常用身高/体重单位换算。不支持单位或跨类别换算返回明确错误。"""
    f = _UNIT_TABLE.get(from_unit)
    t = _UNIT_TABLE.get(to_unit)
    if f is None:
        return CalcResult("convert", "", error=f"unsupported_from_unit:{from_unit}")
    if t is None:
        return CalcResult("convert", "", error=f"unsupported_to_unit:{to_unit}")
    if f[0] is not t[0]:
        return CalcResult("convert", "", error="cross_category_convert")
    base = value * f[1]
    result = round(base / t[1], 2)
    return CalcResult(
        "convert", f"{value} {from_unit} × {f[1]} ÷ {t[1]}",
        value=result, unit=to_unit,
        extra={"from": (value, from_unit)},
    )


# ==================== 意图检测与参数解析（规则，不用 LLM） ====================


@dataclass(frozen=True)
class CalcPlan:
    """detect() 的产出。status: ready | missing | invalid | none。"""

    status: str
    kind: str = ""
    args: dict = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    invalid_msg: str = ""
    remaining: str = ""   # 组合问法中拆分出的知识问题（可为空）


_BMI_RE = re.compile(r"(bmi|体质指数)", re.I)
_STD_W_RE = re.compile(r"(标准体重|理想体重)")
_SEX_RE = re.compile(r"(男|女)")

_HEIGHT_PATTERNS = (
    re.compile(r"身高\s*(\d{3})\s*(?:厘米|cm|公分)?", re.I),
    re.compile(r"(\d{3})\s*(?:厘米|cm|公分)"),
    re.compile(r"1\s*米\s*(\d{1,2})"),
)
_WEIGHT_PATTERNS = (
    re.compile(r"体重\s*(\d{2,3}(?:\.\d+)?)\s*(kg|公斤|千克|斤)?", re.I),
    re.compile(r"(\d{2,3}(?:\.\d+)?)\s*(kg|公斤|千克|斤)"),
)

# 组合问法：计算之后还问了医学解释（剩余部分交知识链路）
_KW_REMAINS = re.compile(
    r"(正常吗|算胖吗|算瘦吗|有什么危害|有什么影响|要注意什么|是什么意思|"
    r"怎么改善|如何改善|怎么办|健康吗|超标吗|偏高|偏低)")

# 单位换算：「170 厘米等于多少米」「65kg 是多少斤」「130斤折合多少公斤」
_CONVERT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+|厘米|公分|米|公斤|千克|斤|磅|克)\s*"
    r"(?:等于|是|约合|折合|换算成?)?\s*(?:多少)?\s*"
    r"(米|厘米|公分|kg|公斤|千克|斤|磅|克|lb|cm|m)",
    re.I)


def _parse_height(text: str) -> float | None:
    for pat in _HEIGHT_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if pat is _HEIGHT_PATTERNS[-1]:  # 1米75
            return 100.0 + float(m.group(1))
        return float(m.group(1))
    return None


def _parse_weight(text: str) -> float | None:
    for pat in _WEIGHT_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        value = float(m.group(1))
        unit = (m.group(2) or "kg").lower()
        if unit == "斤":
            value *= 0.5
        return value
    return None


def _split_remaining(text: str) -> str:
    """从组合问法中拆出知识问题（命中关键词时），否则为空串。"""
    m = _KW_REMAINS.search(text)
    if not m:
        return ""
    tail = text[m.start():].strip("，。?？!！ ")
    return tail


def detect(text: str) -> CalcPlan | None:
    """规则检测计算意图并解析参数。未命中返回 None（走原有 LLM 链路）。"""
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()

    # ---- BMI ----
    if _BMI_RE.search(text):
        height = _parse_height(text)
        weight = _parse_weight(text)
        missing = tuple(k for k, v in (("身高", height), ("体重", weight)) if v is None)
        if missing:
            return CalcPlan("missing", "bmi", missing=missing)
        return CalcPlan("ready", "bmi", args={"height_cm": height, "weight_kg": weight},
                        remaining=_split_remaining(text))

    # ---- 标准体重 ----
    if _STD_W_RE.search(text):
        height = _parse_height(text)
        if height is None:
            return CalcPlan("missing", "standard_weight", missing=("身高",))
        sex_m = _SEX_RE.search(text)
        return CalcPlan("ready", "standard_weight",
                        args={"height_cm": height,
                              "sex": sex_m.group(1) if sex_m else None},
                        remaining=_split_remaining(text))

    # ---- 单位换算 ----
    conv = _CONVERT_RE.search(text)
    if conv and ("换算" in text or "等于" in text or "多少" in text or "约合" in text
                 or "折合" in text):
        value = float(conv.group(1))
        from_unit = conv.group(2).lower()
        to_unit = conv.group(3).lower()
        return CalcPlan("ready", "convert",
                        args={"value": value, "from_unit": from_unit,
                              "to_unit": to_unit})

    return None


__all__ = [
    "BMI_CLASSES",
    "CalcResult",
    "CalcPlan",
    "calc_bmi",
    "calc_standard_weight",
    "convert_unit",
    "detect",
]
