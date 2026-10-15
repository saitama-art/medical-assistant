"""RAG 组合服务层：检索、证据判定、回答生成、来源提取。

本模块只提供能力，**不含编排逻辑**：不决定分支走向、不写 trace、不写终止原因。
调用方向严格单向：`tools.py → service.py → chain.py`，`state.py` 只提供数据类。

失败语义（与 N07 / N08 / N10 的分工一一对应）：

    retrieve_with_scores   不因检索/模型失败而抛异常，归一为 outcome = RETRIEVAL_ERROR / EMPTY_RESULT
    judge_evidence         不抛异常，任何异常一律判为 insufficient（保守降级，设计文档 5.5 / 4.1）
    generate_answer        会抛 RagServiceError，由调用方映射为 GENERATION_ERROR

配置项暂放在本模块顶部。设计文档 5.8 要求它们最终收敛到 `backend/app/config.py`；
本阶段不新增配置文件，因此先就地定义，取值与 5.8 一致。
"""

from __future__ import annotations

import json
import re
import time

from ..medical_agent.state import (
    EvidenceJudgement,
    RetrievalOutcome,
    RetrievalResult,
    RetrievedChunk,
    Source,
)
from .chain import get_llm, get_medical_prompt, get_vector_store


class RagServiceError(RuntimeError):
    """生成环节失败。调用方据此产出 GENERATION_ERROR 分支的固定文案。

    消息里只放机器可读的短标识（如 generate_failed:TimeoutError），
    不得把异常原文交给用户——用户可见文案一律取自 respond.py 的常量表。
    """


# ==================== 配置（设计文档 5.4 / 5.5 / 5.8） ====================

# 距离粗筛阈值。**G-1 门禁：Step 0b 校准完成前必须保持 None**，
# 此时第 1 段不参与判定（仅记录 top1_distance 供观测），
# 不允许为了"先跑通流程"填一个拍脑袋的数值（设计文档 5.4）。
EVIDENCE_THRESHOLD: float | None = None

# 阈值对应的距离度量（cosine / l2）。与 collection 的 hnsw:space 不符时第 1 段自动禁用。
EVIDENCE_THRESHOLD_METRIC: str | None = None

# threshold / llm / both
EVIDENCE_JUDGE: str = "both"

# 第 4 段（生成后引用支撑性核验）。默认 off；门禁 T-02b 不通过时必须改 on（设计文档 5.5）。
GROUNDING_CHECK: str = "off"

# 检索条数。N06 的参数校验用 MIN_TOP_K / MAX_TOP_K 作为 k 的合法区间。
DEFAULT_TOP_K = 3
MIN_TOP_K = 1
MAX_TOP_K = 10

# 问题长度上限，与 N06 校验口径一致（测试 T-07）。
MAX_QUESTION_LEN = 500


# ==================== 检索（N07） ====================


def retrieve_with_scores(question: str, k: int = DEFAULT_TOP_K) -> RetrievalResult:
    """带距离与度量元信息的检索。

    Args:
        question: 已补全的独立问题。
        k: 返回条数。

    Returns:
        RetrievalResult。outcome 的取值语义：

            EMPTY_RESULT / RETRIEVAL_ERROR  检索失败或返回 0 条，此时调用方直接走降级分支
            None                            检索成功且返回了片段，**证据充分性尚未判定**，
                                            由 judge_evidence() 决定（N08）

        注意 outcome 不用 SUFFICIENT 表示"检索成功"：本方案的立论之一就是
        「检索到相关内容 ≠ 证据充分」（设计文档 1.4 候选二）。从检索层直接给出
        SUFFICIENT，等于把最需要拦住的那类误判放行，因此成功时留 None 而非 SUFFICIENT。

    Raises:
        ValueError: question 非法。调用方（N06/N07）必须先完成参数校验再调用，
            这里抛出是为了让契约违反显式暴露，而不是被静默当成"检索无结果"。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("retrieve_with_scores 需要非空字符串 question")
    if len(question) > MAX_QUESTION_LEN:
        raise ValueError("retrieve_with_scores 的 question 超长")
    if not isinstance(k, int) or isinstance(k, bool) or not (MIN_TOP_K <= k <= MAX_TOP_K):
        raise ValueError(f"retrieve_with_scores 的 k 必须在 {MIN_TOP_K}~{MAX_TOP_K} 之间")

    started = time.monotonic()

    try:
        vector_store = get_vector_store()
    except Exception as exc:
        return _failed_retrieval(started, exc)

    # collection 的真实距离度量，随结果一并带出，供第 1 段判定做一致性核对。
    metric = _collection_distance_metric(vector_store)

    try:
        pairs = vector_store.similarity_search_with_score(question.strip(), k=k)
    except Exception as exc:
        return _failed_retrieval(started, exc, metric)

    chunks = [_to_chunk(doc, score) for doc, score in pairs if doc is not None]
    return RetrievalResult(
        outcome=RetrievalOutcome.EMPTY_RESULT if not chunks else None,
        chunks=chunks,
        distance_metric=metric,
        latency_ms=_elapsed_ms(started),
    )


def has_usable_chunks(retrieval: RetrievalResult) -> bool:
    """检索是否成功返回了片段（尚未涉及证据充分性）。"""
    return retrieval.outcome is None and bool(retrieval.chunks)


def _failed_retrieval(started: float, exc: BaseException, metric: str | None = None) -> RetrievalResult:
    """把检索期异常归一为 RETRIEVAL_ERROR。

    error 只记录异常类型名：trace 与响应体都不得出现堆栈、盘符路径、密钥，
    异常原文在真实故障里往往同时包含这三者（设计文档 4.1「不泄露内部信息」）。
    """
    return RetrievalResult(
        outcome=RetrievalOutcome.RETRIEVAL_ERROR,
        distance_metric=metric,
        latency_ms=_elapsed_ms(started),
        error=type(exc).__name__,
    )


def _to_chunk(doc, score) -> RetrievedChunk:
    """把 langchain Document + 距离转成状态层片段，不原样持有 Document。"""
    metadata = getattr(doc, "metadata", None) or {}
    raw_source = metadata.get("source")
    return RetrievedChunk(
        content=getattr(doc, "page_content", "") or "",
        document=str(raw_source) if raw_source else "",
        page=_normalize_page(metadata),
        distance=float(score) if score is not None else None,
    )


def _normalize_page(metadata: dict) -> int | None:
    """把 loader 的 0 基页码转成 1 基展示页码；元数据缺 page 时返回 None。

    这里的 None 与 chain.extract_sources() 的行为**不同**：后者在缺 page 时
    回退成 `0 + 1`，等于显示"第1页"，那是推测出来的页码。主 Agent 面向用户展示
    来源，按测试 T-14 要求，缺页码时只显示文档名，不推测。
    """
    raw = metadata.get("page")
    if raw is None:
        return None
    try:
        return int(raw) + 1
    except (TypeError, ValueError):
        return None


def _collection_distance_metric(vector_store) -> str | None:
    """读取 collection 的真实距离度量（hnsw:space）。取不到时返回 None。

    读不到不视为故障：度量未知时第 1 段会自动禁用（见 _resolve_gate1），
    判定退回第 3 段，而不是拿一个来路不明的刻度去比阈值。
    """
    try:
        metadata = vector_store._collection.metadata or {}
        space = metadata.get("hnsw:space")
        return str(space) if space else None
    except Exception:
        return None


# ==================== 证据判定（N08） ====================


def judge_evidence(
    question: str,
    retrieval: RetrievalResult,
    *,
    llm=None,
    threshold: float | None = None,
    metric: str | None = None,
    judge: str | None = None,
) -> EvidenceJudgement:
    """三段证据判定。**不抛异常**：任何异常一律判为 insufficient。

    段序为「第 1 段 → 第 3 段 → 第 2 段」：第 3 段（模型判定）产出 supporting_quote，
    第 2 段（逐字校验）才有东西可校验。第 1 段先跑，不过就不调模型，省一次 LLM 调用。

    最终判定：第 1 段（若启用）∧ 第 2 段 ∧ 第 3 段（若 judge 含 llm）全部通过才 sufficient。

    Args:
        question: 用户问题（补全后的独立问题）。
        retrieval: retrieve_with_scores() 的结果。
        llm: 可注入，便于测试。默认取 chain.get_llm()。
        threshold / metric / judge: 覆盖模块级配置，仅供测试与后续 config.py 接入。

    Returns:
        EvidenceJudgement。reason 只含机器可读标识，不含用户原文。
    """
    threshold = EVIDENCE_THRESHOLD if threshold is None else threshold
    metric = EVIDENCE_THRESHOLD_METRIC if metric is None else metric
    judge = EVIDENCE_JUDGE if judge is None else judge

    judgement = EvidenceJudgement(distance_metric=retrieval.distance_metric)

    try:
        if not has_usable_chunks(retrieval):
            judgement.reason = f"no_usable_chunks:{retrieval.outcome.value if retrieval.outcome else 'none'}"
            return judgement

        chunks = retrieval.chunks
        judgement.top1_distance = chunks[0].distance

        # —— 第 1 段：距离粗筛（未校准时禁用） ——
        gate1_enabled, gate1_note = _resolve_gate1(threshold, metric, retrieval.distance_metric)
        if gate1_enabled:
            # 启用却拿不到距离，属于异常情形，保守判不通过。
            judgement.stage1_gate1 = (
                judgement.top1_distance is not None and judgement.top1_distance <= threshold
            )
            if not judgement.stage1_gate1:
                judgement.reason = f"stage1_rejected:{gate1_note}:distance={judgement.top1_distance}"
                return judgement
        else:
            judgement.stage1_gate1 = None  # None = 本段未启用，不是"通过"

        # judge = threshold 时第 1 段即最终判定，不再调模型
        if judge == "threshold":
            if not gate1_enabled:
                # 两段都不可用：无判定能力，保守判不足。
                # 该组合本应在启动自检时就被拒绝，见 assert_evidence_judge_available()。
                judgement.reason = f"no_usable_judge:{gate1_note}"
                return judgement
            judgement.sufficient = True
            judgement.reason = f"stage1_passed:gate1={gate1_note}"
            return judgement

        # —— 第 3 段：模型判定"片段是否包含回答所问的依据" ——
        answerable, quote = _judge_by_llm(question, chunks, llm)
        judgement.stage3_answerable = answerable
        judgement.supporting_quote = quote
        if not answerable:
            judgement.reason = "stage3_not_answerable"
            return judgement

        # —— 第 2 段：引用真实性（逐字校验） ——
        judgement.stage2_quote_verified = _verify_quote(quote, chunks)
        if not judgement.stage2_quote_verified:
            judgement.reason = "stage2_quote_not_found"
            return judgement

        judgement.sufficient = True
        judgement.reason = (
            f"stage1_{gate1_note}" if not gate1_enabled else f"all_stages_passed:gate1={gate1_note}"
        )
        return judgement

    except Exception as exc:
        # 判定环节任何异常 → 一律 insufficient（设计文档 4.1、5.5）。
        # 宁可判"证据不足"给出固定文案，也不能放行到 N10 去生成。
        judgement.sufficient = False
        judgement.stage2_quote_verified = False
        judgement.supporting_quote = None
        judgement.reason = f"judge_error:{type(exc).__name__}"
        return judgement


def _resolve_gate1(threshold, metric, collection_metric) -> tuple[bool, str]:
    """第 1 段是否启用。

    未校准或度量不符一律禁用，绝不静默使用未经确认的数值（设计文档 5.4）。

    实测（2026-10-15）：本机 collection 的 `hnsw:space` 为 **None**，即库本身
    没有声明距离度量。此时不禁用第 1 段——阈值本身来自 Step 0b 的实验结论，
    collection 元数据只是交叉核对手段——但必须把"未核对"写进 reason，
    不能让一个未经核对的状态看起来像已确认。
    """
    if threshold is None:
        return False, "disabled_uncalibrated"
    if metric is None:
        return False, "disabled_no_metric_declared"
    if collection_metric is None:
        return True, "enabled_metric_unverified"
    if collection_metric != metric:
        return False, f"disabled_metric_mismatch_{collection_metric}_vs_{metric}"
    return True, "enabled"


JUDGE_SYSTEM_PROMPT = """你是医疗知识库的证据充分性判定器。你只判断给定的参考资料是否足以回答用户问题，不回答用户问题本身。

判定要求：
1. 只有当参考资料中确实包含回答问题所必需的依据时，才输出 "answerable": true。
2. 资料只涉及相关但不相同的主题，或只给出别的信息（如剂量、规格、贮藏、药理、不良反应）而问题问的是另一件事时，必须输出 false。
3. 把握不准、资料内容不完整、需要靠常识补充才能回答时，必须输出 false。
4. answerable 为 true 时，supporting_quote 必须是参考资料中足以支撑回答的那段原文，**逐字复制**：不得改写、不得拼接不相邻的句子、不得加入资料中没有的文字。
5. answerable 为 false 时，supporting_quote 必须为空字符串。
6. 只输出 JSON 对象，不要输出解释、前后缀或 markdown 代码块。"""

JUDGE_USER_PROMPT = """【用户问题】
{question}

【参考资料】
{context}

请输出：{{"answerable": true 或 false, "supporting_quote": "..."}}"""


def _judge_by_llm(question: str, chunks: list[RetrievedChunk], llm=None) -> tuple[bool, str | None]:
    """调用模型做答案支撑性判定。

    Returns:
        (answerable, supporting_quote)

    Raises:
        Exception: 调用失败或输出无法解析。由 judge_evidence 统一兜成 insufficient。
    """
    from langchain_core.prompts import ChatPromptTemplate

    llm = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", JUDGE_USER_PROMPT),
    ])
    messages = prompt.format_messages(
        question=question,
        context=_format_chunks(chunks),
    )
    response = llm.invoke(messages)
    content = getattr(response, "content", None)
    if not isinstance(content, str):
        raise RuntimeError("判定模型返回内容不是文本")

    payload = _parse_json_object(content)
    answerable = payload.get("answerable")
    if not isinstance(answerable, bool):
        raise RuntimeError("判定模型未返回布尔型 answerable")

    quote = payload.get("supporting_quote")
    if not isinstance(quote, str):
        quote = ""
    return answerable, (quote.strip() or None)


def _parse_json_object(text: str) -> dict:
    """从模型输出里取出第一个 JSON 对象。取不到就抛异常（→ insufficient）。"""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("判定模型输出中没有 JSON 对象")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("判定模型输出的 JSON 不是对象")
    return payload


def _verify_quote(quote: str | None, chunks: list[RetrievedChunk]) -> bool:
    """第 2 段：引用是否真实存在于检索片段中。

    先做逐字包含；失败后再做一次「去掉全部空白后的逐字包含」。去掉空白是因为
    模型常在中英文之间增删空格，这属于排版差异；比较的仍是**字符及其顺序**，
    因此不会放过任何编造的或改写过的文字——"引用是不是编的"这一形式化保证不受影响。
    """
    if not quote:
        return False
    stripped = _strip_whitespace(quote)
    if not stripped:
        return False
    for chunk in chunks:
        content = chunk.content or ""
        if quote in content:
            return True
        if stripped in _strip_whitespace(content):
            return True
    return False


def _strip_whitespace(text: str) -> str:
    return "".join(text.split())


# ==================== 回答生成（N10） ====================


def generate_answer(question: str, chunks: list[RetrievedChunk], *, llm=None) -> str:
    """以证据为唯一事实来源生成回答正文。

    只返回正文，不拼来源、不加免责声明：来源与提示由 N11 按固定顺序组装，
    避免来源与正文在两条路径上各拼一次而出现不一致。

    Args:
        question: 补全后的独立问题。
        chunks: 已判定充分的检索片段。
        llm: 可注入，便于测试。

    Returns:
        回答正文。

    Raises:
        RagServiceError: 无片段、模型调用失败或返回空正文。调用方据此走
            GENERATION_ERROR 分支的固定文案，不得把异常内容透给用户。
    """
    if not chunks:
        raise RagServiceError("generate_failed:no_chunks")

    try:
        llm = llm or get_llm()
        messages = get_medical_prompt().format_messages(
            context=_format_chunks(chunks),
            question=question,
        )
        response = llm.invoke(messages)
    except RagServiceError:
        raise
    except Exception as exc:
        raise RagServiceError(f"generate_failed:{type(exc).__name__}") from exc

    content = getattr(response, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RagServiceError("generate_failed:empty_answer")
    return content.strip()


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    """拼接片段作为 Prompt 的 {context}。

    语义与 chain.format_docs() 相同（双换行连接）。这里无法直接复用后者，
    因为它接收的是带 page_content 的 Document，而状态层刻意不持有 Document
    （见 state.RetrievedChunk 的注释）。
    """
    return "\n\n".join(chunk.content for chunk in chunks if chunk.content)


# ==================== 来源提取 ====================


def extract_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    """从检索片段提取面向用户展示的来源。**不经过模型**，全部取自元数据。

    元数据里没有文档名时跳过该片段：宁可不显示来源，也不能编一个文档名出来
    （测试 T-14①：检索置空后回答中不出现任何文档名）。
    """
    sources: list[Source] = []
    seen: set[tuple[str, int | None]] = set()
    for chunk in chunks:
        if not chunk.document:
            continue
        key = (chunk.document, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(Source(document=chunk.document, page=chunk.page, quote=None))
    return sources


# ==================== 启动自检 ====================


def assert_evidence_judge_available() -> None:
    """两段判定都不可用时直接报错，禁止在无判定能力的情况下提供服务。

    对应设计文档 5.4 fail-safe 表的最后一行。**main.py 目前尚未调用它**
    （本阶段不修改 main.py），Step 4 接入 /agent/chat 时必须在启动阶段调用一次。
    """
    gate1_enabled, note = _resolve_gate1(EVIDENCE_THRESHOLD, EVIDENCE_THRESHOLD_METRIC, None)
    if not gate1_enabled and EVIDENCE_JUDGE == "threshold":
        raise RagServiceError(f"evidence_judge_unavailable:{note}")


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


__all__ = [
    "RagServiceError",
    "EVIDENCE_THRESHOLD",
    "EVIDENCE_THRESHOLD_METRIC",
    "EVIDENCE_JUDGE",
    "GROUNDING_CHECK",
    "DEFAULT_TOP_K",
    "MIN_TOP_K",
    "MAX_TOP_K",
    "MAX_QUESTION_LEN",
    "retrieve_with_scores",
    "has_usable_chunks",
    "judge_evidence",
    "generate_answer",
    "extract_sources",
    "assert_evidence_judge_available",
]
