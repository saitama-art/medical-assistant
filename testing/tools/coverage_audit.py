"""从 testing/coverage_matrix.json 生成可审计的覆盖与缺口报告。

只计算测试设计资产，不把 draft 当稳定资产，也不在没有运行清单时声称已执行或通过。
零第三方依赖；在仓库根目录执行：python testing/tools/coverage_audit.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRAFT_STATES = {
    "draft", "unreviewed_candidates", "pending", "pending_user",
    "draft_pending_user_review", "changes_requested", "retired",
}


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: 顶层必须是对象")
    return payload


def _dataset_index(root: Path) -> tuple[dict, dict]:
    """返回 {(dataset, case_id): case} 与每条用例的稳定/草稿状态。"""
    index, states = {}, {}
    for path in sorted((root / "testing" / "datasets").glob("*.json")):
        payload = _read(path)
        name = payload["name"]
        for number, case in enumerate(payload["cases"], 1):
            case_id = case.get("id")
            if not case_id and name == "fr01":
                case_id = f"FR-01-{number}"
            if not case_id:
                raise ValueError(f"{path}: 第 {number} 条缺少 id")
            key = (name, case_id)
            if key in index:
                raise ValueError(f"重复用例引用：{name}:{case_id}")
            index[key] = case
            review_state = case.get("review_status", payload.get("review_status"))
            states[key] = "draft" if review_state in DRAFT_STATES else "stable"
    return index, states


def _summary(rows: list[dict]) -> dict:
    total = len(rows)
    stable = sum(row["stable_covered"] for row in rows)
    design = sum(row["design_covered"] for row in rows)
    return {
        "total": total,
        "stable_covered": stable,
        "design_covered": design,
        "stable_rate": stable / total if total else None,
        "design_rate": design / total if total else None,
    }


def audit(root: Path) -> dict:
    root = root.resolve()
    matrix = _read(root / "testing" / "coverage_matrix.json")
    index, states = _dataset_index(root)
    errors, rows = [], []
    slot_ids = set()

    for slot in matrix["slots"]:
        if slot["id"] in slot_ids:
            errors.append(f"重复槽位：{slot['id']}")
        slot_ids.add(slot["id"])
        references = slot["cases"] + slot["unit_tests"]
        if len(references) != len(set(references)):
            errors.append(f"{slot['id']}: 同一资产重复引用，不得重复计数")
        stable_assets, draft_assets = 0, 0
        for ref in slot["cases"]:
            name, sep, case_id = ref.partition(":")
            if not sep or (name, case_id) not in index:
                errors.append(f"{slot['id']}: 用例引用不存在：{ref}")
                continue
            if states[(name, case_id)] == "draft":
                draft_assets += 1
            else:
                stable_assets += 1
        for relative in slot["unit_tests"]:
            file_ref, separator, test_name = relative.partition("::")
            test_file = root / file_ref
            if not test_file.is_file():
                errors.append(f"{slot['id']}: 测试文件不存在：{file_ref}")
                continue
            if separator and test_name not in test_file.read_text(encoding="utf-8"):
                errors.append(f"{slot['id']}: 测试名称不存在：{relative}")
                continue
            stable_assets += 1

        minimum = slot["min_cases"]
        stable_covered = stable_assets >= minimum
        design_covered = stable_assets + draft_assets >= minimum
        rows.append({
            **slot,
            "stable_assets": stable_assets,
            "draft_assets": draft_assets,
            "stable_covered": stable_covered,
            "design_covered": design_covered,
            "status": "covered" if stable_covered else ("draft_only" if design_covered else "gap"),
        })

    candidate_cases = 0
    for path in (root / "testing" / "candidates").glob("**/*.json"):
        candidate_cases += len(_read(path).get("cases", []))
    return {
        "matrix_name": matrix["name"],
        "version": matrix["version"],
        "policy": matrix["policy"],
        "slots": rows,
        "overall": _summary(rows),
        "critical": _summary([row for row in rows if row["risk"] == "critical"]),
        "layers": {
            layer: _summary([row for row in rows if row["layer"] == layer])
            for layer in sorted({row["layer"] for row in rows})
        },
        "candidate_cases": candidate_cases,
        "errors": errors,
    }


def _rate(summary: dict, key: str) -> str:
    covered = summary["stable_covered"] if key == "stable_rate" else summary["design_covered"]
    value = summary[key]
    return f"{covered}/{summary['total']}（{value:.1%}）" if value is not None else "N/A"


def render_markdown(result: dict) -> str:
    policy = result["policy"]
    overall, critical = result["overall"], result["critical"]
    stable_target = policy["stable_design_target"]
    critical_target = policy["critical_design_target"]
    lines = [
        "# 测试覆盖矩阵与缺口报告",
        "",
        f"> 矩阵：`{result['matrix_name']}` ｜ 版本：`{result['version']}` ｜ 自动生成，请勿手改本报告。",
        "",
        "## 一、结论",
        "",
        "本报告衡量**测试设计覆盖**，不是代码分支覆盖率、测试通过率或医学知识完整率。",
        "没有运行清单时，执行覆盖率与通过率一律记 N/A。",
        "",
        "| 口径 | 覆盖 | 目标 | 是否达标 |",
        "|---|---:|---:|---|",
    ]
    for title, summary, key, target in [
        ("现有稳定资产设计覆盖", overall, "stable_rate", stable_target),
        ("含草稿的设计覆盖（仅展示潜力）", overall, "design_rate", stable_target),
        ("安全关键·稳定资产设计覆盖", critical, "stable_rate", critical_target),
        ("安全关键·含草稿设计覆盖", critical, "design_rate", critical_target),
    ]:
        value = summary[key]
        lines.append(f"| {title} | {_rate(summary, key)} | {target:.0%} | "
                     f"{'是' if value is not None and value >= target else '否'} |")
    lines += [
        "| 实际执行覆盖率 / 通过率 | N/A | 不提供运行清单不计算 | 不判定 |",
        "",
        f"**当前缺口：稳定资产还有 {overall['total'] - overall['stable_covered']} 个槽位未覆盖；"
        f"其中 {sum(row['status'] == 'draft_only' for row in result['slots'])} 个只有草稿，"
        f"{sum(row['status'] == 'gap' for row in result['slots'])} 个连最低设计数量也未达到。**",
        "",
        "## 二、分层覆盖",
        "",
        "| 分层 | 稳定设计覆盖 | 含草稿设计覆盖 |",
        "|---|---:|---:|",
    ]
    for layer, summary in result["layers"].items():
        title = "医疗领域/真实事件" if layer == "medical_domain" else "系统能力"
        lines.append(f"| {title} | {_rate(summary, 'stable_rate')} | {_rate(summary, 'design_rate')} |")
    lines += [
        "",
        "## 三、未达稳定覆盖的槽位",
        "",
        "“仅草稿”不等于已验收；“缺口”表示连最低设计数量都未达到。",
        "",
        "| ID | 场景 | 风险 | 稳定/草稿/最低 | 状态 | 下一步 |",
        "|---|---|---|---:|---|---|",
    ]
    risk_names = {"critical": "关键", "high": "高", "normal": "普通"}
    status_names = {"draft_only": "仅草稿", "gap": "缺口"}
    for row in result["slots"]:
        if row["stable_covered"]:
            continue
        lines.append(
            f"| {row['id']} | {row['title']} | {risk_names[row['risk']]} | "
            f"{row['stable_assets']}/{row['draft_assets']}/{row['min_cases']} | "
            f"{status_names[row['status']]} | {row['next_action']} |"
        )
    lines += [
        "",
        "## 四、公开候选数据与执行边界",
        "",
        f"- HealthBench 等待审核候选共 **{result['candidate_cases']} 条**，位于 `testing/candidates/`；不计入上述任何覆盖率。",
        "- CBLUE、IMCS-21 等数据来源在许可未确认前不导入正式测试集。",
        "- `real_world_events.json` 是项目自有的真实事件路由草稿，只含路由/禁止结果，未冒充医学标准答案。",
        "- `scenarios` 中的 watch 题可算设计资产，但没有硬断言与运行结果时不算验收通过。",
        "- 下一轮先做安全关键草稿终审与执行，再补 MED-18、SYS-19、SYS-20；不得为凑比例删除槽位。",
        "",
        "## 五、复现命令",
        "",
        "```bash",
        "python testing/tools/coverage_audit.py",
        "cd backend && python -m unittest discover -s tests -p \"test_*.py\"",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成测试覆盖矩阵与缺口报告")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit(args.root)
    if result["errors"]:
        for error in result["errors"]:
            print(f"ERROR: {error}")
        return 2
    output = args.output or (args.root / "testing" / "coverage_report.md")
    output.write_text(render_markdown(result), encoding="utf-8", newline="\n")
    print(f"报告：{output}")
    print(f"稳定设计覆盖：{_rate(result['overall'], 'stable_rate')}；"
          f"含草稿：{_rate(result['overall'], 'design_rate')}；执行/通过：N/A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
