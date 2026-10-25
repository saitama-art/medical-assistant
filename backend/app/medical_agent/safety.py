"""紧急风险识别。

纯规则实现：无 LLM、无 IO、无状态。同一输入必定得到同一结果，可复现、可审计。

规则优先级（自上而下，先命中先返回）：

    1. 未命中急症词                        → NONE   (no_acute_term)
    2. 命中但全部被否定                     → NONE   (all_negated, hit_negated=True)
    3. 出现求助主体（第一/第三人称）        → STRONG (help_subject)
    4. 出现当前急症语境词或求助语           → STRONG (acute_context)
    5. 出现说明书等元问题标记               → WEAK   (meta_question)
    6. 其余（裸症状词）                     → BARE_SYMPTOM_LEVEL

第 3、4 条排在元问题标记之前，因此"我吃了这个药突然胸痛"这类既有元问题词又有真实
急症信号的说法，会按 STRONG 处理，不会被"说明书/这个药"压成 WEAK。

第 2 条的否定判定**只在同一分句内**向前回溯（见 CLAUSE_BREAK）：分句标点之后的症状
不受前一分句否定词的影响。"我的一侧手脚无力，口角歪斜"曾被前一分句的"无"整条否掉，
漏拦了一次卒中征象。

保守性：宁可误拦，不可漏拦。需求 9.3.5 要求"判定不确定但存在风险可能时，按紧急处理"。
"""

from __future__ import annotations

from .state import EmergencyHit, EmergencyLevel


class EmergencyRuleError(RuntimeError):
    """紧急规则自身异常（词表为空、输入类型非法）。

    调用方（编排器）必须捕获本异常并保守降级为就医提示，
    不得因为规则不可用就继续常规问答。
    """


# ==================== 词表 ====================
# 说明：词表为模块级常量。若后续需要纳入配置管理，整体搬移即可，规则逻辑不变。

# 急症词：每组是同一含义的不同表述，命中组内任一即算命中该组。共 9 组。
ACUTE_TERMS: tuple[tuple[str, ...], ...] = (
    ("胸痛", "胸口疼", "胸口痛", "心口疼", "心口痛", "胸闷痛", "胸骨后疼", "压榨样痛", "压榨样"),
    ("呼吸困难", "喘不上气", "喘不过气", "窒息", "呼吸急促", "气促", "憋气"),
    ("意识障碍", "意识不清", "昏迷", "抽搐", "晕厥", "昏倒", "叫不醒"),
    ("大出血", "大量出血", "呕血", "咯血", "吐血", "便血", "血便", "止不住血"),
    ("剧烈头痛", "爆炸样头痛", "头痛得厉害", "头疼得厉害"),
    # 卒中征象组。「一侧无力」是既有条目，但它不是「一侧手脚无力」的子串——部位词
    # 插在「一侧」与「无力」之间就会漏检，而这类写法正是最常见的口语表述。故按
    # 「偏侧 + 部位 + 无力」补齐同义表述；只补带偏侧标记的说法，不收裸的「无力」/
    # 「肢体无力」（双侧慢性乏力会被误拦为急症）。
    (
        "言语不清", "说话不清楚", "口齿不清",
        "一侧无力", "单侧无力", "一侧肢体无力", "单侧肢体无力",
        "一侧手脚无力", "单侧手脚无力",
        "半边身子", "口角歪斜", "偏瘫", "面瘫",
    ),
    ("误服", "吃错药", "吃多了药", "过量服药"),
    ("高热惊厥",),
    # 咽喉梗阻组（2026-10-20 真实验收补）：与「呼吸困难」组分开计组——两组同时
    # 命中会触发叠加征象规则（multiple_groups → STRONG）。实测漏拦案例：
    # 「喉咙发紧肿起来了，呼吸困难，喘不过气」只有语境词在表内时只有单组命中。
    ("喉头水肿", "喉咙发紧", "喉咙肿", "咽喉肿痛", "异物卡喉"),
)

# 第一人称：说话人自己正在出现症状。
FIRST_PERSON: tuple[str, ...] = ("我", "本人", "自己")

# 第三人称求助主体：替家人/他人求助。
# 刻意不收裸的"他/她"——"阿托伐他汀"里含"他"，会把药品说明书类问题误判为紧急。
# 这类说法（如"他胸痛怎么办"）会由 HELP_SEEKING 兜住，不会漏。
THIRD_PERSON_SUBJECTS: tuple[str, ...] = (
    "我妈", "我爸", "妈妈", "爸爸", "母亲", "父亲", "家长",
    "孩子", "小孩", "宝宝", "婴儿", "幼儿", "儿子", "女儿",
    "家人", "家里人", "家属", "老人", "老年人", "爷爷", "奶奶",
    "姥姥", "姥爷", "外公", "外婆", "老公", "老婆", "妻子", "丈夫",
    "爱人", "对象", "朋友", "同事", "病人", "患者", "室友", "亲戚",
)

# 当前急症语境：症状正在发生或正在加重。
CURRENT_CONTEXT: tuple[str, ...] = (
    "现在", "此刻", "这会儿", "刚刚", "刚才", "突然", "一直", "正在",
    "目前", "眼下", "越来越", "加重", "持续", "急诊", "马上", "立刻",
    # 2026-10-20 真实验收补：出血控制不住的表述本身就是急症语境（按压不止仍出血），
    # 与「持续/加重」同档。只对已命中急症词的句子升级，裸句不受影响。
    "止不住", "按压不止",
    # 2026-10-25 场景集 CS-13 暴露：「服药后出现皮疹且呼吸困难」单组命中急症词但
    # 无语境词/求助语未升级 STRONG，落 unclassified（T-03 同类题靠「怎么办」兜底，
    # 本句无求助语即漏）。「服药后/用药后」=用药行为后紧接着出现症状，保守优先按
    # 急症语境处理。同样只对已命中急症词的句子升级——「服药后血压稳定」无急症词，
    # 不受影响。
    "服药后", "用药后",
)

# 求助语：没有主语也能表明是真实求助，而不是在问资料。
HELP_SEEKING: tuple[str, ...] = (
    "怎么办", "咋办", "如何是好", "怎么处理", "怎么急救", "救命", "急救",
    "要不要去医院", "该不该去医院", "要去医院吗", "需要去医院吗", "需要急诊吗",
    "挂什么科", "挂哪个科", "该挂哪个科", "哪个科",
    "严重吗", "危险吗", "有危险吗", "会不会有事", "吃什么药",
)

# 否定词：出现在急症词之前的小窗口内即视为否定。
NEGATION: tuple[str, ...] = ("不", "没", "无", "未", "别")

# 含否定字但不是否定语义的搭配，避免误判为"没有症状"。
NEGATION_EXCEPTIONS: tuple[str, ...] = ("不止", "不仅", "不但", "不光是", "不排除")

# 元问题标记：用户是在问资料里怎么写的，不是在描述自身症状。
META_QUESTION: tuple[str, ...] = (
    "说明书", "是否提到", "有没有提到", "提到过", "副作用", "不良反应",
    "禁忌", "适应症", "药理", "成分", "用法用量", "这个药", "这种药",
)

# 否定词的检测窗口：急症词起点前多少个字符。
NEGATION_WINDOW = 3

# 分句标点：否定只在**同一分句内**生效，窗口回溯到此为止。
# 反例（本次修复的漏拦根因）：「我的一侧手脚无力，口角歪斜」——「口角歪斜」前 3 字符的
# 候选窗口是「无力，」，其中的「无」是**上一个分句**里「无力」的一部分，并不是对
# 「口角歪斜」的否定。窗口跨过标点，就会把阳性症状误判成被否定，进而整条漏拦。
CLAUSE_BREAK = "，。；、！？：,.;!?:…"

# 只命中症状词、既无求助主体也无急症语境时的等级（可调整档位）。
BARE_SYMPTOM_LEVEL: EmergencyLevel = EmergencyLevel.WEAK


# ==================== 对外接口 ====================


def check_emergency(text: str) -> EmergencyHit:
    """判定一段文本的紧急风险等级。

    Args:
        text: 用户文本（可以是本轮原始输入，也可以是补全上下文后的问题）。

    Returns:
        EmergencyHit：等级、命中的规则标识与命中词。level 为 STRONG 时调用方
        必须直接返回紧急应答并终止流程，不得继续检索或生成。

    Raises:
        EmergencyRuleError: 词表不可用或 text 类型非法。调用方应保守降级为就医提示。
    """
    _validate_tables()
    normalized = _normalize(text)
    if not normalized:
        return EmergencyHit(level=EmergencyLevel.NONE, rule_id="empty_input")

    hit_terms = [term for term in _iter_terms() if term in normalized]
    if not hit_terms:
        return EmergencyHit(level=EmergencyLevel.NONE, rule_id="no_acute_term")

    active, negated = _split_by_negation(normalized, hit_terms)
    if not active:
        return EmergencyHit(
            level=EmergencyLevel.NONE,
            rule_id="all_negated",
            terms=tuple(negated),
            hit_negated=True,
        )

    terms = tuple(active)
    has_negated = bool(negated)

    if _contains_any(normalized, FIRST_PERSON) or _contains_any(normalized, THIRD_PERSON_SUBJECTS):
        return EmergencyHit(EmergencyLevel.STRONG, "help_subject", terms, has_negated)

    if _contains_any(normalized, CURRENT_CONTEXT) or _contains_any(normalized, HELP_SEEKING):
        return EmergencyHit(EmergencyLevel.STRONG, "acute_context", terms, has_negated)

    if _contains_any(normalized, META_QUESTION):
        return EmergencyHit(EmergencyLevel.WEAK, "meta_question", terms, has_negated)

    # 叠加征象规则（2026-10-20 真实验收补）：单个裸症状词保持 WEAK（2.2 路由表约定），
    # 但**两个及以上不同急症组同时命中**时，即使没有求助主体/语境词也按 STRONG 处理。
    # 依据：多组征象并存本身就是急症语境（胸痛+大汗、呼吸困难+喉头肿），且符合
    # 「宁可误拦、不可漏拦」（需求 9.3.5）。实测漏拦案例：
    # 「喉咙发紧肿起来了，呼吸困难，喘不过气」「胸口压榨样疼痛，大汗淋漓，持续不缓解」。
    # 刻意不放宽单组裸词——「胸痛」单独出现仍走 WEAK + 风险提示，不回退 2d7c9e6 约定。
    active_groups = sum(
        1 for group in ACUTE_TERMS
        if any(term in normalized for term in group)
        # 该组至少要有一个未被否定的命中词，与 active 语义一致
        and any(term in active for term in group)
    )
    if active_groups >= 2:
        return EmergencyHit(EmergencyLevel.STRONG, "multiple_groups", terms, has_negated)

    return EmergencyHit(BARE_SYMPTOM_LEVEL, "bare_symptom", terms, has_negated)


# ==================== 内部实现 ====================


def _iter_terms():
    for group in ACUTE_TERMS:
        for term in group:
            yield term


def _validate_tables() -> None:
    if not ACUTE_TERMS or not any(ACUTE_TERMS):
        raise EmergencyRuleError("急症词表为空，紧急风险检查不可用")


def _normalize(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        raise EmergencyRuleError(f"紧急风险检查只接受字符串，收到 {type(text).__name__}")
    # 去掉所有空白，避免"胸 痛"这类写法漏检，也让否定窗口更稳定。
    return "".join(text.split())


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _negation_window(text: str, index: int) -> str:
    """取急症词起点前的候选窗口，并在分句标点处截断。

    截断方向是保守的：窗口变小只会让"被否定"更难成立，即更容易判为阳性症状
    （宁可误拦，不可漏拦，需求 9.3.5）。

    影响范围可控：只有当**否定字与被否定的急症词之间夹了标点**时才改变结果；
    「没有胸痛」「无明显胸痛」「不排除胸痛」这些写法窗口内本来就没有标点，行为不变。
    """
    window = text[max(0, index - NEGATION_WINDOW):index]
    cut = max((window.rfind(ch) for ch in CLAUSE_BREAK), default=-1)
    return window[cut + 1:]


def _is_negated(text: str, index: int) -> bool:
    """判断从 index 处开始的急症词是否被否定。"""
    window = _negation_window(text, index)
    return any(ch in window for ch in NEGATION) and not any(
        exc in window for exc in NEGATION_EXCEPTIONS
    )


def _occurrences(text: str, term: str) -> list[int]:
    positions: list[int] = []
    start = text.find(term)
    while start != -1:
        positions.append(start)
        start = text.find(term, start + 1)
    return positions


def _split_by_negation(text: str, terms: list[str]) -> tuple[list[str], list[str]]:
    """按词逐个判断是否被否定。

    逐个判断而不是整句判断，是为了正确处理
    "我没有胸痛但现在突然喘不上气"这种部分否定的情况：
    "胸痛"被否定丢弃，"喘不上气"仍然有效并触发 STRONG。

    否定只在同一分句内生效，见 _negation_window。

    同一个词可能出现多次，只要**存在任一处未被否定**就算有效命中：
    "没有口角歪斜，但现在突然口角歪斜"里真正危险的是后一处，只查首次出现会漏掉它。
    """
    active: list[str] = []
    negated: list[str] = []
    for term in terms:
        if any(not _is_negated(text, i) for i in _occurrences(text, term)):
            active.append(term)
        else:
            negated.append(term)
    return active, negated
