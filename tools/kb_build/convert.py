"""把 kb_src/ 抓取成功的文本逐字转成 PDF，写入 backend/data/ 供 ingest 建库。

依据《知识库扩充-盘点与规划.md》第三节：
- 原始抓取内容保留在 kb_src/（本脚本只读不改）；
- 用 fpdf2 把逐字文本转成 PDF 放入 backend/data/——ingest 流程、切分参数、检索逻辑零改动；
- 每份 PDF 首页附来源信息块：标题、来源 URL、来源类型、获取日期（取 .txt 文件 mtime，即实际抓取日）；
- 可重复执行：已存在的目标 PDF 直接覆盖。

用法（ai_agent 环境，仓库根目录）：

    python tools/kb_build/convert.py            # 转换全部 ok 条目
    python tools/kb_build/convert.py --dry-run  # 只打印计划不落盘
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

from fpdf import FPDF
from fontTools.ttLib import TTFont

# ---- 路径（相对本文件解析：kb_build/ -> tools/ -> 仓库根） ----
_ROOT = Path(__file__).resolve().parents[2]
KB_SRC = _ROOT / "kb_src"
DATA_DIR = _ROOT / "backend" / "data"
REPORT = KB_SRC / "fetch_report.json"

# ---- CJK 字体：优先黑体（纯 TTF，无 TTC 索引问题） ----
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
]
FONT_SIZE = 11

# 来源类型标注（与盘点文档 1.5 节来源表一致）
SOURCE_TYPE = {
    "msd": "默沙东诊疗手册·大众版（病人教育资料，逐字转载）",
    "dayi": "中国医药信息查询平台（经审核认证词条，逐字转载）",
    "who": "世界卫生组织（WHO 中文实况报道，逐字转载）",
    "cnsoc": "中国营养学会《中国居民膳食指南（2022）》（逐字转载）",
    "bwjw": "北京市卫生健康委员会（科普转载，逐字转载）",
}
# 文件名里的来源短标签（同时是引用展示的一部分，用中文可读名）
SOURCE_LABEL = {
    "msd": "默沙东诊疗手册",
    "dayi": "中国医药信息查询平台",
    "who": "WHO",
    "cnsoc": "中国居民膳食指南",
    "bwjw": "北京市卫健委",
}

_INVALID_FILENAME = re.compile(r'[\\/:*?"<>|]')


def _pick_font_path() -> str:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    raise SystemExit("未找到可用 CJK 字体（检查了 C:\\Windows\\Fonts 黑体/雅黑/宋体）")


def _load_supported_chars(font_path: str) -> set:
    """读字体 cmap，返回支持的字符集；不支持的字符落盘前剔除，防止渲染异常。"""
    font = TTFont(font_path, fontNumber=0)
    cmap = font.getBestCmap()
    font.close()
    return set(cmap.keys()) if cmap else set()


def _sanitize_filename(name: str) -> str:
    return _INVALID_FILENAME.sub("_", name).strip()


class _KbPdf(FPDF):
    def header(self):
        pass  # 首页信息块由正文绘制，不每页重复页眉

    def footer(self):
        self.set_y(-12)
        self.set_font("cjk", size=8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 8, f"第 {self.page_no()} 页", align="C")
        self.set_text_color(0, 0, 0)


def _write_wrapped(pdf: FPDF, text: str) -> None:
    """按实际字宽贪心折行写一段文本（CJK 无空格，multi_cell 自动折行不可靠）。"""
    pdf.set_font("cjk", size=FONT_SIZE)
    line = ""
    for ch in text:
        if ch == "\t":
            ch = " "
        candidate = line + ch
        if pdf.get_string_width(candidate) <= pdf.w - pdf.l_margin - pdf.r_margin:
            line = candidate
        else:
            pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
            line = "" if ch == " " else ch
    if line:
        pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")


def convert_one(entry: dict, supported: set, font_path: str, dry_run: bool) -> Path:
    key, site = entry["key"], entry["site"]
    txt_path = KB_SRC / site / f"{key}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"缺少文本文件: {txt_path}")

    title = entry["title"]
    out_name = _sanitize_filename(f"{title}_{SOURCE_LABEL[site]}.pdf")
    out_path = DATA_DIR / out_name

    if dry_run:
        print(f"[dry-run] {key} -> {out_name} ({txt_path.stat().st_size} bytes)")
        return out_path

    raw = txt_path.read_text(encoding="utf-8")
    # 剔除字体不支持的字符（如个别 emoji/罕见符号），保留 \n
    text = "".join(c for c in raw if c == "\n" or ord(c) in supported).strip()

    fetch_date = datetime.date.fromtimestamp(txt_path.stat().st_mtime).isoformat()

    pdf = _KbPdf(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(18, 18, 18)
    pdf.add_font("cjk", "", font_path)
    pdf.add_page()

    # ---- 首页来源信息块 ----
    pdf.set_font("cjk", size=14)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("cjk", size=FONT_SIZE)
    info_lines = [
        f"知识类别：{entry['category']}",
        f"资料来源：{SOURCE_TYPE[site]}",
        f"来源 URL：{entry['url']}",
        f"获取日期：{fetch_date}",
        "",
        "以下为逐字转载的原文内容：",
    ]
    for line in info_lines:
        if line:
            _write_wrapped(pdf, line)
        else:
            pdf.ln(4)
    pdf.ln(2)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)

    # ---- 逐字正文 ----
    for para in text.split("\n"):
        if para.strip():
            _write_wrapped(pdf, para)
        else:
            pdf.ln(3)

    pdf.output(str(out_path))
    print(f"[ok] {key} -> {out_name} ({out_path.stat().st_size // 1024} KB, 正文 {len(text)} 字)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="kb_src 文本 -> backend/data PDF（首页含来源块）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划不生成")
    args = parser.parse_args()

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    ok_entries = [e for e in report if e.get("ok")]
    print(f"报告共 {len(report)} 条，抓取成功 {len(ok_entries)} 条")

    font_path = _pick_font_path()
    print(f"字体: {font_path}")
    supported = _load_supported_chars(font_path)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for entry in ok_entries:
        outputs.append(convert_one(entry, supported, font_path, args.dry_run))
    print(f"完成：{len(outputs)} 份" + ("（dry-run，未落盘）" if args.dry_run else ""))


if __name__ == "__main__":
    sys.exit(main())
