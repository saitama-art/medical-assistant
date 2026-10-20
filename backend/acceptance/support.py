"""验收 harness 共用小工具（无第三方依赖，可被单测直接引用）。"""

import re

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}"),          # DeepSeek 等 OpenAI 兼容 key
    re.compile(r"(?i)(api[_-]?key|token|secret)([\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9_\-]{8,}"),
]


def redact(text):
    """把任何字符串里的密钥样式内容替换为 ***REDACTED***。不落盘密钥的兜底。"""
    if not isinstance(text, str):
        return text
    for pat in _SECRET_PATTERNS:
        text = pat.sub(lambda m: (m.group(1) + m.group(2) + "***REDACTED***")
                       if m.lastindex and m.lastindex >= 2 else "***REDACTED***", text)
    return text


def _ngrams(text, n=8):
    text = re.sub(r"\s+", "", text)
    return {text[i:i + n] for i in range(len(text) - n + 1)} if len(text) >= n else {text} if text else set()


def char_ngram_support(answer, source_texts, n=8):
    """T-16-lite 自动指标：回答逐句检查是否存在于任一来源片段的字符 n-gram 集合中。
    返回 {sentences, supported, ratio}。仅供参考，不等于人工核对引用支持率。"""
    sents = [s.strip() for s in re.split(r"[。！？!?；\n]", answer) if len(s.strip()) >= 12]
    if not sents:
        return {"sentences": 0, "supported": 0, "ratio": None}
    corpus = set().union(*(_ngrams(t, n) for t in source_texts)) if source_texts else set()
    supported = sum(1 for s in sents if _ngrams(s, n) & corpus)
    return {"sentences": len(sents), "supported": supported,
            "ratio": round(supported / len(sents), 3)}
