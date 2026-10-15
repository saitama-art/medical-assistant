"""固定响应文案。

所有用户可见的提示都必须取自本模块的常量，禁止在各节点内联字符串，禁止拼接异常对象
（str(e) 会把内部信息暴露给用户）。

硬约束：
1. 除最终回答正文外，本模块的文案不得包含任何具体的医学结论、病因判断或用药建议；
2. 不得出现堆栈、密钥、服务器路径、模型配置。
"""

from __future__ import annotations

from .state import TerminationReason


# ==================== 默认声明 ====================

DISCLAIMER = "本内容仅供健康科普参考，不构成诊断、处方或治疗建议，不能替代医生面诊。"


# ==================== 紧急路径 ====================

# 只给行动指引，不给病因判断、不给用药建议。
EMERGENCY_ANSWER = (
    "您描述的情况可能属于需要立即处理的情况，请不要等待在线回复。\n"
    "请立即拨打 120 急救电话，或马上前往最近的医院急诊科。\n"
    "如身边有人，请让对方陪同并协助说明症状出现的时间和表现。"
)

# 紧急规则自身异常时的保守提示：不继续常规问答，也不作任何判断。
CONSERVATIVE_MEDICAL_NOTICE = (
    "为了安全起见，本次无法为您提供在线判断。\n"
    "您描述的情况需要当面评估，建议尽快前往 医院就诊；如症状明显或正在加重，请立即前往急诊。"
)

# 紧急词出现在非紧急语境（如询问说明书内容）时，附加在正常回答顶部。
RISK_NOTICE = (
    "提示：如果您或身边的人正在出现类似的急性症状，请不要依赖在线查询，"
    "请立即拨打 120 或前往急诊。"
)


# ==================== 知识问答路径 ====================

# 生成前的证据判定未通过。不得包含任何具体医学结论。
EVIDENCE_INSUFFICIENT = (
    "知识库中暂未找到足够依据来回答这个问题，因此不能给出结论，以免提供不准确的信息。\n"
    "建议您咨询线下医生或药师，获取针对性的意见。"
)

# 检索成功但没有返回任何内容（与"检索失败"是两回事，措辞必须不同）。
EMPTY_RESULT = "知识库中暂无相关资料。"

# 检索环节执行失败或超时。
RETRIEVAL_ERROR = "知识库检索服务暂时不可用，请稍后重试。"

# 证据已判定充分，但生成环节失败或超时。
GENERATION_ERROR = "回答生成服务暂时不可用，请稍后重试。"

# 未预期的异常兜底。
SERVICE_UNAVAILABLE = "服务暂时不可用，请稍后重试。"


# ==================== 其余路径 ====================

CHITCHAT_REPLY = (
    "您好，我是医疗健康知识助手。我可以依据已收录的医学资料回答药品和健康知识方面的问题，"
    "也可以提示哪些情况需要尽快就医。请问您想了解什么？"
)

# 分类失败/超时/输出非法时使用。不得声称知道用户想问什么。
UNCLASSIFIABLE_REPLY = (
    "抱歉，我暂时无法判断您的问题属于健康知识查询还是就医指引。\n"
    "请补充说明您想了解的内容，例如具体的药名、症状或想查的资料，我再试着回答。"
)

# 就医指引：只给通用方向，不指定具体科室（见需求待确认 9）。
TRIAGE_GUIDANCE = (
    "这里只能给出一般性的就医方向，不能替代医生判断。\n"
    "如果症状持续、反复或影响到日常生活，建议尽快到正规医院就诊，"
    "由分诊台或全科医生结合实际情况安排相应的检查与科室。"
)

# 上下文已丢弃时使用。不得猜测用户指代的对象。
CONTEXT_MISSING_REPLY = (
    "较早的对话内容已不再保留，我无法确认您指的是哪一项。\n"
    "请把问题重新完整描述一次，例如直接写明药名或症状。"
)


# ==================== 执行控制类 ====================

INVALID_ARGS_REPLY = "本次查询提交的信息不符合要求，请重新描述您的问题。"

# 达到轮数上限、重复调用被拦截，或工具失败后无法继续时使用。
ROUNDS_EXCEEDED_REPLY = "本次查询未能获得更多有效信息，请补充或修改问题后重试。"

REPEATED_CALL_REPLY = "本次查询未能获得更多信息，请补充或修改问题后重试。"

BUDGET_EXHAUSTED_REPLY = "当前请求的处理时间已超出上限，请稍后重试，或把问题描述得更简短一些。"

# 已产出可用结果、但流程未完整走完时，附加在回答末尾。
PARTIAL_RESULT_NOTICE = "（注意：本次查询未能完整完成，以上内容可能不完整。）"


# ==================== 组装用 ====================

SOURCE_HEADER = "\n\n【参考来源】："


# ==================== 终止原因 → 用户可见文案 ====================

TERMINATION_TEXTS: dict[TerminationReason, str] = {
    TerminationReason.EMERGENCY_HANDLED: EMERGENCY_ANSWER,
    TerminationReason.EMERGENCY_RULE_FAULT: CONSERVATIVE_MEDICAL_NOTICE,
    TerminationReason.INSUFFICIENT_EVIDENCE: EVIDENCE_INSUFFICIENT,
    TerminationReason.EMPTY_KNOWLEDGE_BASE: EMPTY_RESULT,
    TerminationReason.RETRIEVAL_FAILED: RETRIEVAL_ERROR,
    TerminationReason.GENERATION_FAILED: GENERATION_ERROR,
    TerminationReason.UNCLASSIFIED: UNCLASSIFIABLE_REPLY,
    TerminationReason.CONTEXT_MISSING: CONTEXT_MISSING_REPLY,
    TerminationReason.ARGUMENT_INVALID: INVALID_ARGS_REPLY,
    TerminationReason.MAX_ROUNDS_REACHED: ROUNDS_EXCEEDED_REPLY,
    TerminationReason.REPEATED_CALL_BLOCKED: REPEATED_CALL_REPLY,
    TerminationReason.BUDGET_EXHAUSTED: BUDGET_EXHAUSTED_REPLY,
}
# COMPLETED 不在本表中：它返回的是生成的回答正文，没有固定文案。
