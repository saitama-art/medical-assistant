"""O2 归因：误拒哨兵 CS-09（孕期常见不适）/ CS-10（长期熬夜）。

问题：两题在场景集里持续判 insufficient_evidence（哨兵观察）。
目标：区分 ①证据偏题（检索片段确实不答所问）②判定口径过严（有答案段仍被拒）。
方法：零 LLM 检索诊断（top-10 文档+距离+前 3 条片段正文）+ 真实 judge_evidence
（第 3 段判定理由与 supporting_quote）。判定层调用 2 次 LLM/题，其余零成本。
运行（cwd=backend/）：G:\\annacoda\\envs\\ai_agent\\python.exe -I scripts/diag_sentinel.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.rag import service  # noqa: E402

QUESTIONS = [
    ("CS-09", "孕期常见不适有哪些一般处理原则？", ["孕期自我保健", "孕期母体变化", "孕期用药"]),
    ("CS-10", "长期熬夜可能产生哪些影响？", ["昼夜节律", "睡眠概述", "失眠"]),
]


def main() -> None:
    for cid, q, expects in QUESTIONS:
        print(f"\n================ {cid}: {q}")
        r = service.retrieve_with_scores(q)
        print("retrieval outcome:", r.outcome)
        for i, c in enumerate(r.chunks[:10], 1):
            doc = os.path.basename(c.document)
            mark = "★期望" if any(e in doc for e in expects) else "  "
            print(f"  [{i:2d}] {mark} {doc[:46]}  d={c.distance:.3f}")
        for i, c in enumerate(r.chunks[:3], 1):
            print(f"  --- top{i} 正文前 300 字：{c.content[:300]!r}")
        result = service.judge_evidence(q, r)
        print("  判定节点状态:", result.status.value)
        j = result.value
        print("  判定：sufficient=", j.sufficient,
              " stage3=", j.stage3_answerable,
              " stage2=", j.stage2_quote_verified,
              " reason=", j.reason)
        print("  supporting_quote:", (j.supporting_quote or "")[:160])


if __name__ == "__main__":
    main()
