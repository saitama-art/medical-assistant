"""校验 golden_seed 并生成用户人工审核页。

用法（仓库根目录）：python -I testing/tools/golden_seed.py
审核前它是候选集，不是已经确认的医学金标准；执行覆盖率也不由本工具推断。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROUTES = {"knowledge", "triage", "emergency"}
OUTCOMES = {
    "completed", "insufficient_evidence", "emergency_handled",
    "unclassified", "context_missing",
}
EVIDENCE_STATES = {
    "retrieval_candidate_pending_review", "partial_requires_label_review",
    "expected_absent_or_insufficient", "not_required_for_fixed_emergency_response",
    "external_guideline_required", "not_required_for_triage",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def validate(root: Path) -> tuple[dict, list[str], list[str]]:
    path = root / "testing" / "datasets" / "golden_seed.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors, warnings = [], []
    if payload.get("name") != "golden_seed":
        errors.append("name 必须为 golden_seed")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return payload, ["cases 必须是数组"], warnings
    if payload.get("case_count") != len(cases):
        errors.append("case_count 与实际用例数不一致")
    snapshot = payload.get("knowledge_base_snapshot") or {}
    if not snapshot.get("document_count") or not snapshot.get("chunk_count"):
        errors.append("缺少知识库文档/chunk 快照")

    seen = set()
    for case in cases:
        cid = case.get("id", "<missing-id>")
        if cid in seen:
            errors.append(f"{cid}: ID 重复")
        seen.add(cid)
        if not case.get("question") or not case.get("turns"):
            errors.append(f"{cid}: question/turns 不能为空")
        expected = case.get("expected") or {}
        if expected.get("route") not in ROUTES:
            errors.append(f"{cid}: route 非法")
        if expected.get("outcome") not in OUTCOMES:
            errors.append(f"{cid}: outcome 非法")
        if case.get("risk") not in {"normal", "high", "critical"}:
            errors.append(f"{cid}: risk 非法")
        if case.get("review_priority") not in {"P0", "P1", "P2"}:
            errors.append(f"{cid}: review_priority 非法")
        rubric = case.get("rubric") or {}
        if not rubric.get("required_points") or not rubric.get("forbidden_points"):
            errors.append(f"{cid}: 必答点和禁止点均不能为空")
        review = case.get("review") or {}
        if review.get("status") not in payload.get("review_statuses", []):
            errors.append(f"{cid}: review.status 非法")
        if review.get("status") == "approved" and (
                not review.get("reviewer") or not review.get("reviewed_at")):
            errors.append(f"{cid}: approved 必须记录审核人和日期")

        state = case.get("evidence_status")
        if state not in EVIDENCE_STATES:
            errors.append(f"{cid}: evidence_status 非法")
        evidence = case.get("evidence") or []
        if state in {"retrieval_candidate_pending_review", "partial_requires_label_review"}:
            if not evidence:
                errors.append(f"{cid}: 当前证据状态要求至少一条证据")
        if state == "external_guideline_required":
            warnings.append(f"{cid}: 尚需权威外部指南")
        for item in evidence:
            quote = item.get("quote") or ""
            if len(quote) < 20:
                errors.append(f"{cid}: 证据摘录过短")
            if hashlib.sha256(quote.encode("utf-8")).hexdigest() != item.get("quote_sha256"):
                errors.append(f"{cid}: quote_sha256 不匹配")
            if not HEX64.fullmatch(item.get("chunk_sha256", "")):
                errors.append(f"{cid}: chunk_sha256 格式非法")
            if not isinstance(item.get("page"), int) or item["page"] < 1:
                errors.append(f"{cid}: 页码必须从 1 开始")
            if not isinstance(item.get("retrieval_rank"), int) or item["retrieval_rank"] < 1:
                errors.append(f"{cid}: retrieval_rank 非法")
            if not (root / "backend" / "data" / item.get("document", "")).is_file():
                errors.append(f"{cid}: 证据文档不存在：{item.get('document')}")
    return payload, errors, warnings


def _bullets(values: list[str]) -> list[str]:
    return [f"  - {value}" for value in values]


def render_review(payload: dict) -> str:
    cases = payload["cases"]
    lines = [
        "# 黄金种子集人工审核页",
        "",
        f"> 版本：`{payload['version']}` ｜ 共 {len(cases)} 条 ｜ "
        "当前全部待用户终审，不是已批准医学金标准。",
        "",
        "## 审核方法",
        "",
        "按 P0 → P1 → P2 顺序审核。重点看：期望路由是否合理、必答点是否被证据支持、",
        "禁止点是否覆盖危险错误、证据摘录是否准确且足够。不要只审参考答案措辞。",
        "",
        "- `retrieval_candidate_pending_review`：来自当前 Chroma 的候选证据，仍需你确认。",
        "- `partial_requires_label_review`：证据不完整，必须补完整说明书/原文。",
        "- `external_guideline_required`：当前知识库没有足够资料，必须补权威指南。",
        "- 勾选页用于人工记录；最终需在 JSON 写回 `approved/changes_requested/retired`、审核人和日期。",
        "",
    ]
    for priority in ("P0", "P1", "P2"):
        group = [case for case in cases if case["review_priority"] == priority]
        lines += [f"## {priority}（{len(group)} 条）", ""]
        for case in group:
            lines += [
                f"### {case['id']} · {case['group']} · {case['risk']}",
                "",
                f"- **问题/上下文**：{' → '.join(case['turns'])}",
                f"- **期望**：{case['expected']['route']} / {case['expected']['outcome']}",
                "- **必答点**：",
                *_bullets(case["rubric"]["required_points"]),
                "- **禁止点**：",
                *_bullets(case["rubric"]["forbidden_points"]),
            ]
            if case["rubric"]["acceptable_uncertainty"]:
                lines += ["- **可接受的不确定性**：", *_bullets(case["rubric"]["acceptable_uncertainty"])]
            lines.append(f"- **证据状态**：`{case['evidence_status']}`")
            if case["evidence"]:
                lines.append("- **证据候选**：")
                for item in case["evidence"]:
                    lines.append(
                        f"  - `{item['document']}` 第 {item['page']} 页；检索 rank {item['retrieval_rank']}；"
                        f"distance={item['distance']}"
                    )
                    lines.extend("    > " + line for line in item["quote"].splitlines())
            else:
                lines.append("- **证据候选**：无（原因见上方证据状态）。")
            if case["notes"]:
                lines.append(f"- **起草备注**：{case['notes']}")
            lines += [
                "- **审核**：[ ] 通过　[ ] 修改　[ ] 删除　审核人：______　日期：______",
                "- **审核备注**：____________________________________________________________",
                "",
            ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="校验黄金种子集并生成人工审核页")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload, errors, warnings = validate(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2
    output = args.output or (args.root / "testing" / "golden_review.md")
    output.write_text(render_review(payload), encoding="utf-8", newline="\n")
    print(f"审核页：{output}")
    print(f"用例：{len(payload['cases'])}；证据：{sum(len(c['evidence']) for c in payload['cases'])}；"
          f"待补外部指南提示：{len(warnings)}；全部仍为待审核状态")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
