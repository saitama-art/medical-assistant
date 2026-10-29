"""意图识别（N04）：寒暄规则短路 → LLM 结构化分类 → 类别/置信校验。

从 router.py 拆出（2026-10-29 解耦）：本模块只负责"理解用户想要什么"，
不负责路由分发（decide_route 留在 router.py）与追问触发判定。
合法的业务未知、低置信度与模型执行失败分别表达；任何路径都不得猜最接近的一类。
"""

from __future__ import annotations

import json
import math
import re

from . import complex_handler
from .state import (
    IntentAssessment,
    NodeResult,
    NodeStatus,
    QuestionCategory,
    UnresolvedKind,
)

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
    triage         就医决策与就诊安排：问题的主要诉求是决定去哪看病——要不要去医院、
                   挂什么科、去哪类医疗机构、急救转运等就诊安排本身
    emergency      正在发生的、可能危及生命的急症求助
    daily          日常表达与生活类求助：不是医学知识查询，但用户期待一个自然、
                   有用的回应。包括生活状态表达（「饿了」「最近好累」「心情不太好」）、
                   日常生活建议（「晚上吃什么」「医生让我减肥」「熬夜怎么调整作息」）、
                   以及对本次对话内容的回顾（「今天我说了什么」「我们刚才聊了什么」）
    chitchat       纯寒暄客套：你好、谢谢、再见等，不含任何实际诉求
    unclassifiable 确实无法判断意图，或以上内容都不相关且无法回应

knowledge 与 triage 的分界（最常见混淆点，2026-10-21 按 72 题覆盖测试实测修正）：
- 「某症状怎么办/如何改善/怎么调理/平时注意什么」→ knowledge（问的是医学知识与自助方法）
- 症状、疾病或用药问题中附带就医判断的——「是否需要就医」「什么时候需要就医」
  「需不需要治疗」「能否自行停药」——核心是了解该症状/疾病/药物的知识，相关词条
  内通常本身含就医时机，归 knowledge。例如：
  「打鼾严重是否需要就医」→ knowledge（打鼾词条含就医指征）
  「咳嗽持续不愈，什么时候需要就医」→ knowledge（咳嗽词条含就医时机）
  「长期服药能否自行停药」→ knowledge（用药知识）
  「幽门螺杆菌感染需要治疗吗」→ knowledge（疾病知识）
  「焦虑到什么程度需要寻求专业帮助」→ knowledge（心理疾病知识）
  「什么时候应该去急诊」→ knowledge（急症识别知识）
- 只有当"就诊安排本身就是问题的主要诉求"时才归 triage：「要不要去医院」
  「挂什么科」「去哪看」「需要去急诊吗（询问就诊去向而非急症知识）」。例如：
  「胸口疼要不要去医院」→ triage（决定去不去医院是主要诉求）
  「持续性头痛应该挂什么科」→ triage（选择科室是主要诉求）
- 「（某病/某症状）出现什么情况要去医院」「什么情况需要就医」问的是该疾病的
  就医指征/警示征象知识 → knowledge。例如：
  「腹泻出现什么情况要去医院」→ knowledge（腹泻的警示征象知识）
- 仅描述症状而未问就诊安排（如「晚上睡不着」「最近总是头晕」）→ knowledge
- 问「某症状该吃什么药/用什么药好」而未指明具体药名 → triage（用药选择须医生判断）；
  指明具体药名问用法用量、不良反应 → knowledge
- 极短模糊的症状倾诉、无任何可定位信息（如「疼」「难受」「不舒服」）→ triage

daily 与其他类别的分界（2026-10-28 实测修正，核心原则：口语化短表达不等于无法分类）：
- 不要因为输入短、没有问号、没有医学术语就判 unclassifiable。先判断用户
  最可能想要一个什么性质的回应：
  「饿了」→ daily（生活状态表达，期待自然回应）
  「最近好累」→ daily（状态表达）
  「医生让我减肥」→ daily（生活健康目标，期待一般性建议）
  「晚上吃什么」→ daily（生活建议）
  「熬夜了怎么调整」→ daily（作息生活建议）
  「被狗咬了」→ knowledge（明确的医疗场景，需要规范处置知识，不是日常闲聊）
- daily 只用于"不需要查医学资料也能自然回应"的内容；涉及症状、疾病、
  药品、指标的具体医学问题时，即使很短也归 knowledge 或 triage。
- 对先前对话的回顾（「今天我说了什么」「刚才我们聊了什么」）→ daily，
  不是 chitchat，也不是 unclassifiable。
- 假设、否认或创作场景中的医疗事件**不是正在发生的事件**：
  「我没有被狗咬，只是在写文章」→ daily 或 unclassifiable，绝不能归 emergency。
- missing_inputs 只在"缺失信息会实质改变回答"时才使用；用户只是描述得简短，
  但已能给出一般性回应时，不要为了凑字段输出 missing_input。

参考示例（类别仅为示意格式）：
- 「失眠了怎么办」→ knowledge；「晚上睡不着」→ knowledge
- 「头疼要挂什么科」→ triage；「胸口疼要不要去医院」→ triage
- 「我有点头晕要吃什么药」→ triage；「二甲双胍怎么吃」→ knowledge
- 「饿了」→ daily；「医生让我减肥」→ daily；「今天我说了什么」→ daily
- 「你好」→ chitchat

要求：
1. 只输出 JSON 对象，不要输出解释、前后缀或 markdown 代码块；
2. confidence 是 0 到 1 之间的数，表示你对分类的把握；不确定时给出低置信度；
   confidence 只是辅助信号，低置信度不必然意味着必须澄清；
3. 举棋不定时宁可输出 unclassifiable，也不要猜最接近的一类；
4. unresolved 只能取 none、missing_input、context_required、intent_conflict：
   信息完整为 none；必要信息缺失为 missing_input；必须依赖未提供的上下文为
   context_required；多个意图冲突且不能确定主任务为 intent_conflict；
5. missing_inputs 仅列出缺失参数名，不得猜测或编造参数值；其他情况输出空数组；
6. 输出格式：{{"category": "knowledge|triage|emergency|daily|chitchat|unclassifiable",
   "confidence": 0.0~1.0, "unresolved": "none", "missing_inputs": []}}"""

CLASSIFY_USER_PROMPT = "用户问题：{question}"
CLASSIFY_REPAIR_PROMPT = """

上一次输出未通过结构或字段校验。请只重新输出符合上述 Schema 的合法 JSON 对象，
不要解释错误，不要增加字段，不要猜测缺失参数。"""


def classify(
    question: str, *, llm, repair: bool = False,
) -> NodeResult[IntentAssessment]:
    """执行一次 N04 分类；业务判断与调用/解析失败使用不同状态。"""
    if is_greeting(question):
        return NodeResult(
            NodeStatus.OK,
            IntentAssessment(QuestionCategory.CHITCHAT, 1.0),
        )

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", CLASSIFY_SYSTEM_PROMPT + (CLASSIFY_REPAIR_PROMPT if repair else "")),
        ("human", CLASSIFY_USER_PROMPT),
    ])
    messages = prompt.format_messages(question=question)
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        return complex_handler.exception_result(exc)

    content = getattr(response, "content", None)
    if not isinstance(content, str):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="content_not_text")
    try:
        payload = _parse_json_object(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="json_invalid")

    raw_category = payload.get("category")
    if not isinstance(raw_category, str):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="category_invalid")
    try:
        category = QuestionCategory(raw_category.strip().lower())
    except ValueError:
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="category_invalid")

    confidence = payload.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="confidence_invalid")

    raw_unresolved = payload.get("unresolved", UnresolvedKind.NONE.value)
    if not isinstance(raw_unresolved, str):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="unresolved_invalid")
    try:
        unresolved = UnresolvedKind(raw_unresolved.strip().lower())
    except ValueError:
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="unresolved_invalid")

    raw_missing = payload.get("missing_inputs", [])
    if not isinstance(raw_missing, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_missing
    ):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="missing_inputs_invalid")
    missing_inputs = tuple(dict.fromkeys(item.strip() for item in raw_missing))
    return NodeResult(
        NodeStatus.OK,
        IntentAssessment(category, float(confidence), unresolved, missing_inputs),
    )


def _parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("分类模型输出中没有 JSON 对象")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("分类模型输出的 JSON 不是对象")
    return payload


__all__ = [
    "CLASSIFY_REPAIR_PROMPT",
    "CLASSIFY_SYSTEM_PROMPT",
    "CLASSIFY_USER_PROMPT",
    "classify",
    "is_greeting",
]
