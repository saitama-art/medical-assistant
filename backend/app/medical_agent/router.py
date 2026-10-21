"""意图识别（N04）与条件路由（N05）。

意图识别：寒暄规则短路 → LLM 结构化分类 → 类别/置信校验。
任何失败（异常、超时、输出非法、置信不足、预算不足未启动）都归入
UNCLASSIFIABLE，**不得猜最接近的一类**（设计文档 4.1 第 5 条）。

条件路由：category + emergency + context_status → Route 的确定性映射表
（设计文档 2.2），无任何判断空间。
"""

from __future__ import annotations

import json
import re

from .state import (
    ContextStatus,
    EmergencyLevel,
    QuestionCategory,
    Route,
)

# ==================== 配置（设计文档 5.8，最终收敛 config.py） ====================

CATEGORY_CONFIDENCE_MIN = 0.6

# 追问指代标记：命中任一即认为本句依赖上文，需要上下文补全 / 重述判定
REFERENCE_WORDS: tuple[str, ...] = (
    "它", "这个", "那个", "这种", "该药", "这药", "哪个",
    "上面", "之前", "刚才", "上次", "前述",
)


def has_reference(text: str) -> bool:
    """文本是否含指代标记（用于 N02 条件触发与 N05 优先级 2）。"""
    return any(word in text for word in REFERENCE_WORDS)


# ==================== 寒暄短路（T-04：命中时 0 次 LLM、0 次工具） ====================

_GREETING_RE = re.compile(
    r"^(你好|您好|嗨|哈喽|HELLO|Hello|hello|HI|Hi|hi|在吗|在么|在不在|"
    r"早上好|上午好|下午好|晚上好|中午好|谢谢|感谢|辛苦了)"
    r"[啦呢吧啊呀哦哈~～!！。.好\s]*$",
)


def is_greeting(text: str) -> bool:
    """寒暄/客套话的规则短路。整句匹配，避免"你好，我想问二甲双胍"被误判。"""
    if not isinstance(text, str):
        return False
    return bool(_GREETING_RE.match(text.strip()))


# ==================== LLM 结构化分类 ====================

CLASSIFY_SYSTEM_PROMPT = """你是医疗问答的意图分类器。把用户问题分到且只分到以下一类：

    knowledge      查询医学知识：药品信息（适应症、用法用量、不良反应、成分、药理、
                   价格、医保报销等药品相关事实）、疾病或症状的科普与处理/调理/改善方法
                   （如「失眠怎么办」「胃痛如何缓解」）、检查指标含义、预防保健与日常注意事项
    triage         就医决策：要不要去医院/需不需要就医、挂什么科、去哪类医疗机构、
                   急救转运等就诊安排问题
    emergency      正在发生的、可能危及生命的急症求助
    chitchat       寒暄、闲聊、与医疗无关的客套话
    unclassifiable 无法确定归属以上任何一类

knowledge 与 triage 的分界（最常见混淆点）：
- 「某症状怎么办/如何改善/怎么调理/平时注意什么」→ knowledge（问的是医学知识与自助方法）
- 「要不要去医院/挂什么科/去哪看/严重吗需要就医吗」→ triage（问的是就诊决策）
- 仅描述症状而未问就诊安排（如「晚上睡不着」「最近总是头晕」）→ knowledge
- 问「某症状该吃什么药/用什么药好」而未指明具体药名 → triage（用药选择须医生判断）；
  指明具体药名问用法用量、不良反应 → knowledge
- 极短模糊的症状倾诉、无任何可定位信息（如「疼」「难受」「不舒服」）→ triage

参考示例（类别仅为示意格式）：
- 「失眠了怎么办」→ knowledge；「晚上睡不着」→ knowledge
- 「头疼要挂什么科」→ triage；「胸口疼要不要去医院」→ triage
- 「我有点头晕要吃什么药」→ triage；「二甲双胍怎么吃」→ knowledge

要求：
1. 只输出 JSON 对象，不要输出解释、前后缀或 markdown 代码块；
2. confidence 是 0 到 1 之间的数，表示你对分类的把握；不确定时给出低置信度；
3. 举棋不定时宁可输出 unclassifiable，也不要猜最接近的一类；
4. 输出格式：{{"category": "knowledge|triage|emergency|chitchat|unclassifiable", "confidence": 0.0~1.0}}"""

CLASSIFY_USER_PROMPT = "用户问题：{question}"


def classify(question: str, *, llm) -> tuple[QuestionCategory, float]:
    """LLM 结构化分类。

    Returns:
        (category, confidence)。任何失败都返回 (UNCLASSIFIABLE, 0.0)，
        调用方（编排器 N04）不捕获异常——本函数把异常全部吞掉。
    """
    if is_greeting(question):
        return QuestionCategory.CHITCHAT, 1.0
    try:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", CLASSIFY_SYSTEM_PROMPT),
            ("human", CLASSIFY_USER_PROMPT),
        ])
        messages = prompt.format_messages(question=question)
        response = llm.invoke(messages)
        content = getattr(response, "content", None)
        if not isinstance(content, str):
            return QuestionCategory.UNCLASSIFIABLE, 0.0

        payload = _parse_json_object(content)
        raw_category = str(payload.get("category", "")).strip().lower()
        try:
            category = QuestionCategory(raw_category)
        except ValueError:
            return QuestionCategory.UNCLASSIFIABLE, 0.0

        confidence = payload.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence = 0.0
        confidence = max(0.0, min(1.0, float(confidence)))

        if category is QuestionCategory.UNCLASSIFIABLE:
            return QuestionCategory.UNCLASSIFIABLE, confidence
        if confidence < CATEGORY_CONFIDENCE_MIN:
            return QuestionCategory.UNCLASSIFIABLE, confidence
        return category, confidence
    except Exception:
        # 超时、连接错误、输出无法解析……一律归入 unclassifiable（N04 失败语义）
        return QuestionCategory.UNCLASSIFIABLE, 0.0


def _parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("分类模型输出中没有 JSON 对象")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("分类模型输出的 JSON 不是对象")
    return payload


# ==================== 条件路由表（N05，设计文档 2.2） ====================


def decide_route(
    category: QuestionCategory | None,
    emergency_level: EmergencyLevel,
    context_status: ContextStatus,
    resolved_question: str,
) -> Route:
    """条件路由的确定性映射。优先级自上而下，先命中先返回。"""
    # 1. 强紧急：实际已在 N01/N03 拦截，此处是防御性冗余
    if emergency_level is EmergencyLevel.STRONG:
        return Route.EMERGENCY
    # 2. 上下文缺失且仍含未消解指代：不得猜测指代对象（FR-06 边界）
    if context_status is ContextStatus.MISSING and has_reference(resolved_question):
        return Route.CONTEXT_MISSING
    # 3~6. 按类别走表；不存在"默认进知识库问答"的分支
    if category is QuestionCategory.CHITCHAT:
        return Route.CHITCHAT
    if category in (QuestionCategory.UNCLASSIFIABLE, None):
        return Route.UNCLASSIFIABLE
    # 2.2 表没有 category == emergency 这一行：完整紧急处理只由上面优先级 1 的
    # emergency.level == strong 触发（N01/N03 已先行拦截）。分类器判 emergency 而
    # 规则未判 strong（如裸症状「胸痛」判 weak）时，不升级为紧急，落入下方兜底。
    if category is QuestionCategory.TRIAGE:
        return Route.TRIAGE
    if category is QuestionCategory.KNOWLEDGE:
        return Route.KNOWLEDGE
    # 兜底：以上均不匹配 → 无法处理话术
    return Route.UNCLASSIFIABLE


__all__ = [
    "CATEGORY_CONFIDENCE_MIN",
    "REFERENCE_WORDS",
    "has_reference",
    "is_greeting",
    "classify",
    "decide_route",
]
