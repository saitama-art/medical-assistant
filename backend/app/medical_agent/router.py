"""条件路由（N05）与追问触发判定。

意图识别本体已迁至 intent.py（2026-10-29 解耦）；本模块保留：
- 追问/指代触发（has_reference / has_followup_lead，N02 前置条件）；
- 条件路由：category + emergency + context_status → Route 的确定性映射表
  （设计文档 2.2），无任何判断空间。

意图相关的 classify/is_greeting 经底部 re-export 保持旧引用面不变。
"""

from __future__ import annotations

import re

from . import config as _cfg
from .intent import (
    CLASSIFY_REPAIR_PROMPT,
    CLASSIFY_SYSTEM_PROMPT,
    CLASSIFY_USER_PROMPT,
    classify,
    is_greeting,
)
from .state import (
    ContextStatus,
    EmergencyLevel,
    QuestionCategory,
    Route,
)

# ==================== 配置（设计文档 5.8，最终收敛 config.py） ====================

CATEGORY_CONFIDENCE_MIN = _cfg.CATEGORY_CONFIDENCE_MIN

# 追问指代标记：命中任一即认为本句依赖上文，需要上下文补全 / 重述判定
REFERENCE_WORDS: tuple[str, ...] = (
    "它", "这个", "那个", "这种", "该药", "这药", "哪个",
    "上面", "之前", "刚才", "上次", "前述",
)


def has_reference(text: str) -> bool:
    """文本是否含指代标记（用于 N02 条件触发与 N05 优先级 2）。"""
    return any(word in text for word in REFERENCE_WORDS)


# 隐性回指开头：「那什么情况要去医院？」——「那」起头的追问句式不含 REFERENCE_WORDS
# 任何词，但显然依赖上文。只匹配句首（自包含问题罕见以「那什么/那该/那要/那还」
# 开头），且 N02 另有"历史非空"前置条件，误触发代价最多一次改写调用。
# 注意：「那」已在 ^ 后消费，「那种」只需匹配剩余的「种」，写「种」即可。
FOLLOWUP_LEAD_RE = re.compile(r"^那(?:什么|哪种|种|该|要|还能|还)")

# 「那…」触发排除明确完结的陈述句：自包含句「那要么明天再说。」曾命中触发，
# 历史非空时白付一次改写调用、失败走重述（2026-10-25 S1-D1b 实测修复）。
# 无终止符的短句（「那要不要紧」）仍视为追问。

# 零代词省略追问（S1-D1）：短句（≤8 字）+ 问号收尾 + 含疑问标记。
# 「用量呢？」「几点吃？」「空腹吃好不好？」有历史时必须触发 N02——
# 基线实测（CS-17）不触发时残缺问句直送分类器落 unclassified。
# 误触发代价 ≤ 一次改写调用；漏触发代价 = 落 unclassified 的安全兜底。
_FRAGMENT_RE = re.compile(r"^[一-鿿A-Za-z0-9]{2,8}？$")
# 2026-10-28 实测补：「什么/办法」是省略追问的高频结尾——
# 「需要注意什么？」「吃什么？」「有什么办法？」此前不触发 N02，
# 裸问句直送分类器被判 unclassifiable（ux_probe S1-T2/T3、S3-T2）。
_FRAGMENT_MARKERS = ("呢", "吗", "吧", "几", "多", "怎", "什么", "办法", "好不好", "行行")


def has_followup_lead(text: str) -> bool:
    """文本是否以隐性追问开头（那什么/那该/那要…），或为零代词省略追问短句。

    用于 N02 触发补充。历史非空由 N02 前置条件保证，这里只做文本级判定。
    """
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if FOLLOWUP_LEAD_RE.match(s):
        # 「那…」分支：陈述句终止符（。/！）排除 + 长度 ≤10。过长的
        # 「那种保健品不用吃也行吧？」是自带主语的陈述（12 字），不触发；
        # 宁精勿滥，漏触发有 unclassified 的安全兜底。
        return len(s) <= 10 and not s.endswith(("。", "！", "!"))
    if _FRAGMENT_RE.match(s) and any(m in s for m in _FRAGMENT_MARKERS):
        return True
    return False


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
    # 2. 上下文不可用且仍是显式/隐性追问：不得猜测指代对象（FR-06 边界）
    if (
        context_status in (ContextStatus.MISSING, ContextStatus.DEGRADED_NO_HISTORY)
        and (has_reference(resolved_question) or has_followup_lead(resolved_question))
    ):
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
    if category is QuestionCategory.DAILY:
        return Route.DAILY
    if category is QuestionCategory.KNOWLEDGE:
        return Route.KNOWLEDGE
    # 兜底：以上均不匹配 → 无法处理话术
    return Route.UNCLASSIFIABLE


__all__ = [
    "CATEGORY_CONFIDENCE_MIN",
    "REFERENCE_WORDS",
    "has_reference",
    "has_followup_lead",
    "is_greeting",
    "classify",
    "decide_route",
]
