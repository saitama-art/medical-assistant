"""验收 harness CLI。

    python -m acceptance --list
    python -m acceptance --only T-03
    python -m acceptance --only T-02b-1,T-12-二甲双胍
    python -m acceptance --fr01 --out fr01_report.json
    python -m acceptance --out acceptance_results/run.json
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .cases import CASE_GENERATORS
from .runner import run_report


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m acceptance",
        description="真实 LLM 验收 harness（TestClient + 真实 /agent/chat 编排链路）")
    ap.add_argument("--list", action="store_true", help="列出全部验收项后退出")
    ap.add_argument("--only", help="逗号分隔的组名或编号（如 T-03 / T-02b-1 / T-12-二甲双胍）")
    ap.add_argument("--fr01", action="store_true",
                    help="追加 FR-01 诊断样本（diagnostic，不计通过/失败）")
    ap.add_argument("--out", help="JSON 报告路径（默认 acceptance_results/<时间戳>.json）")
    args = ap.parse_args(argv)

    if args.list:
        for g, _fn, desc in CASE_GENERATORS:
            print(f"{g:8s} {desc}")
        print("FR-01    诊断样本 8 条（--fr01 追加；引用支持率自动指标仅诊断用，"
              "官方口径为人工核对 ≥0.85）")
        return 0

    out = args.out
    if not out:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("acceptance_results")
        out_dir.mkdir(exist_ok=True)
        out = str(out_dir / f"run_{ts}.json")

    run_report(only=args.only, fr01=args.fr01, out_path=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
