"""trend.py：汇总历次评分卡/报告 JSON → 趋势数据 + 单文件离线 HTML 面板（S5）。

定位（方案 V2 §四呈现第 3 层，可选项）：**只读已有产物**，不执行任何测试、
不调 LLM。输入是 S4b 的 scorecard_*.json 与 S4c 的 report_*.json，输出：
- trends JSON（日期×维度分矩阵的数据源，面板与人工都可读）；
- 单文件 HTML（内嵌数据 + 内联 SVG 折线图 + 矩阵表，无外部依赖，双击可开）。

file:// 下浏览器不能自动读目录，故面板另提供「选择 JSON 文件」入口追加数据；
常规用法是每次跑完报告后重新执行本脚本生成最新面板。

诚实口径（与报告一致）：
- N/A 维度在图上是**断点**，不补 0、不插值；总分缺失（安全否决）同样不画点；
- 输入套件不同 / 条件指纹未采集的 run 之间不做自动 diff，只并列展示并给警告；
- 无法识别的 JSON 进 skipped 清单并说明原因，不静默丢弃。

用法（backend/ 下）：
    python -m acceptance.trend                      # 默认读 ../Evaluation/reports
    python -m acceptance.trend ../Evaluation/reports/20261025 --out ../Evaluation/reports/dashboard.html
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SERIES = ("total", "retrieval", "answer", "refusal", "latency", "cost")
SERIES_NAMES = {"total": "总分", "retrieval": "检索", "answer": "答案质量",
                "refusal": "拒答健康", "latency": "时延", "cost": "成本"}
_DIR_DATE = re.compile(r"^(20\d{6})$")


def extract_run(path: str | Path) -> dict:
    """从单个 scorecard/report JSON 提取面板所需字段。

    两种产物同构处是 scorecard 块（S4b/S4c 约定）；缺 generated_at 时退到
    父目录名（YYYYmmdd，Evaluation/reports/20261025 约定），并标 time_basis。
    无法识别（无 scorecard.dimensions）→ ValueError，由调用方记入 skipped。
    """
    p = Path(path)
    rep = json.loads(p.read_text(encoding="utf-8"))
    sc = rep.get("scorecard")
    if not isinstance(sc, dict) or not isinstance(sc.get("dimensions"), dict):
        raise ValueError("无 scorecard.dimensions，不是评分卡/报告产物")
    generated = rep.get("generated_at")
    time_basis = "run"
    if not generated:
        parent = p.parent.name
        if _DIR_DATE.match(parent):
            generated = f"{parent[:4]}-{parent[4:6]}-{parent[6:8]}"
            time_basis = "dir_date"
        else:
            time_basis = "unknown"
    sources = rep.get("sources") or []
    suites = sorted({s.get("suite", "?") for s in sources}) or ["未记录"]
    promptmeta = rep.get("promptmeta")
    return {
        "id": p.stem,
        "file": p.name,
        "generated_at": generated,
        "time_basis": time_basis,
        "total": sc.get("total"),
        "valid": sc.get("valid", True),
        "partial": sc.get("partial"),
        "note": rep.get("note") or sc.get("note"),
        "dimensions": {k: (v or {}).get("score") for k, v in sc["dimensions"].items()},
        "suites": suites,
        "sources": [{"suite": s.get("suite"), "file": s.get("file"), "n": s.get("n")}
                    for s in sources],
        "promptmeta_status": ("已采集" if promptmeta and any(promptmeta.values())
                              else "未采集"),
    }


def collect_runs(paths: list[str | Path]) -> dict:
    """读多个产物 → {runs（按时间+文件名排序）, skipped}。坏文件不中断。"""
    runs, skipped = [], []
    for path in paths:
        try:
            runs.append(extract_run(path))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            skipped.append({"file": Path(path).name, "reason": str(e)})
    runs.sort(key=lambda r: (r["generated_at"] is None, r["generated_at"] or "",
                             r["file"]))
    return {"runs": runs, "skipped": skipped}


def build_trends(runs: list[dict]) -> dict:
    """趋势矩阵 + 可比性警告。不同输入套件/缺指纹只警告，不擅自剔除任何 run。"""
    matrix = []
    for r in runs:
        row = {"id": r["id"], "generated_at": r["generated_at"]}
        for key in SERIES:
            row[key] = r["total"] if key == "total" else r["dimensions"].get(key)
        matrix.append(row)
    warnings = []
    signatures = {"|".join(r["suites"]) for r in runs}
    if len(signatures) > 1:
        warnings.append("各 run 输入套件不一致（" + "；".join(
            f"{r['id']}={'/'.join(r['suites'])}" for r in runs)
            + "）：分数差异可能来自口径而非系统变化，不能直接当回归/提升。")
    if any(r["promptmeta_status"] == "未采集" for r in runs):
        warnings.append("部分 run 无条件指纹（promptmeta 未采集，多为 S4a 前的历史产物）："
                        "逐维度条件校验需 S4d，面板不做自动 diff。")
    if any(not r["valid"] for r in runs):
        warnings.append("存在安全否决/数据缺失的 run：其总分记 N/A，不参与趋势连线。")
    return {"series": [{"key": k, "name": SERIES_NAMES[k]} for k in SERIES],
            "matrix": matrix, "warnings": warnings}


def discover(root: str | Path) -> list[Path]:
    """目录输入：递归取 *.json（trends 自身产物除外），按路径排序保证可复现。"""
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.json") if not p.name.startswith("trends"))


def render_html(payload: dict, template_path: str | Path) -> str:
    """模板注入：仅替换 __DATA__ 占位。`</` 转义防 JSON 内文本截断 script。"""
    template = Path(template_path).read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return template.replace("__DATA__", data)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m acceptance.trend",
                                 description="评分卡/报告 JSON → 趋势数据 + 离线 HTML 面板")
    ap.add_argument("paths", nargs="*", default=["../Evaluation/reports"],
                    help="产物文件或目录（默认 ../Evaluation/reports）")
    ap.add_argument("--template", default="../Evaluation/dashboard.template.html")
    ap.add_argument("--out", default="../Evaluation/reports/dashboard.html")
    ap.add_argument("--trends-json", default=None, help="趋势数据输出路径（默认同 out 目录 trends.json）")
    args = ap.parse_args(argv)

    files = [p for path in args.paths for p in discover(path)]
    if not files:
        ap.error("未找到任何 JSON 产物")
    collected = collect_runs(files)
    trends = build_trends(collected["runs"])
    payload = {"runs": collected["runs"], "skipped": collected["skipped"], **trends}

    trends_path = Path(args.trends_json) if args.trends_json else Path(args.out).parent / "trends.json"
    trends_path.parent.mkdir(parents=True, exist_ok=True)
    trends_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload, args.template), encoding="utf-8")

    print(f"runs={len(collected['runs'])} skipped={len(collected['skipped'])}")
    for s in collected["skipped"]:
        print(f"  跳过 {s['file']}: {s['reason']}")
    for w in trends["warnings"]:
        print(f"WARNING: {w}")
    print(f"面板：{out}\n趋势数据：{trends_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
