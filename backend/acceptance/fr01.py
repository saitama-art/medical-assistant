"""FR-01 引用支持率诊断样本与记录逻辑。

题稿与版本：根目录 testing/datasets/fr01.json。

口径说明（对齐设计 T-16 / 开发进度 2026-10-16 条目）：
- 官方 FR-01 验收 = 人工核对，引用支持率 ≥0.85；本模块产出的是**诊断数据**，
  自动指标（8-gram 字符重合，T-16-lite）只用于定位问题，不表述为真实正确率。
- 每样本记录调用链各环节事实：检索 top-k（文档/距离）→ 子 Agent outcome →
  来源集合 → 回答 → 自动支持率，失败样本据此可区分：
  「未召回 / 召回但判定拒答 / 生成但引用不支持」三类原因。
"""

from .datasets import load_dataset
from .support import char_ngram_support

_DATASET = load_dataset("fr01")
# 保留既有 (question, expect_keyword) 元组接口，调用方无需修改。
FR01_SAMPLES = [(case["question"], case["expect_keyword"])
                for case in _DATASET["cases"]]


def run_fr01_sample(ctx, question, expect_kw):
    """ctx 提供 .chat(sid,q) 与 .retrieve(q)；返回诊断记录 dict。"""
    rec = {"question": question, "expect_keyword": expect_kw}

    # 环节 1：纯检索诊断（真实索引、零 LLM）
    contents = []
    try:
        r = ctx.retrieve(question)
        rec["retrieval"] = {
            "outcome": r.get("outcome"),
            "top": [{"doc": t["doc"], "distance": t["distance"]} for t in r.get("top", [])],
            "expect_doc_hit_rank": next(
                (i + 1 for i, t in enumerate(r.get("top", [])) if expect_kw in t["doc"]),
                None),
        }
        contents = [t.get("content", "") for t in r.get("top", [])]
    except Exception as e:  # noqa: BLE001 —— 诊断不允许抛
        rec["retrieval"] = {"error": type(e).__name__}

    # 环节 2：完整编排链路（真实 LLM）
    code, body, ms = ctx.chat("fr01", question)
    rec.update(ms=ms, http_status=code, outcome=body.get("outcome"),
               n_sources=len(body.get("sources", []) or []),
               sources=[s.get("document", "") for s in body.get("sources", []) or []],
               answer=body.get("answer", ""))

    # 环节 3：自动引用支持率（T-16-lite，仅诊断用；语料=本样本检索 top-k 正文）
    if body.get("answer") and contents:
        try:
            rec["auto_support_t16lite"] = char_ngram_support(body["answer"], contents)
        except Exception as e:  # noqa: BLE001
            rec["auto_support_t16lite"] = {"error": type(e).__name__}
    return rec
