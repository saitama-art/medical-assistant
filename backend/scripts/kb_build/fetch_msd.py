"""默沙东诊疗手册·大众版词条批量入库流水线（kb_build 第 10 轮扩源）。

流程：sitemap 候选 → 排除已收录（按现有 PDF 页眉 URL 去重）→ 分类轮转选样
→ 抓取正文（div.TopicMainContent）→ reportlab/SimHei 生成 PDF（页眉格式与
既有文档逐字一致）→ data/ 落盘 + manifest.json 记录。

只生成 PDF，不重建索引；索引重建走 ingest.py（需先停服务，README 第 6 节）。

用法（backend/ 目录，-X utf8 运行）：
    python -X utf8 scripts/kb_build/fetch_msd.py --target 360
    python -X utf8 scripts/kb_build/fetch_msd.py --target 360 --dry-run   # 只列计划
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]          # backend/
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
SITEMAP = HERE / "msd_sitemap.xml"
MANIFEST = HERE / "manifest.json"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
FETCH_INTERVAL = 1.2          # 抓取间隔（秒），礼貌限速
MIN_CONTENT_CHARS = 400       # 正文不足视为多媒体/空页，跳过
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"

_SKIP_PREFIX = (
    "resourcespages", "pages-with-widgets", "pronunciations", "healthy-living",
    "search", "multimedia", "about", "professional", "home",
)


def entry_urls() -> list[str]:
    xml = SITEMAP.read_text(encoding="utf-8")
    urls = re.findall(r"<loc>([^<]+)</loc>", xml)
    out = []
    for u in urls:
        if "/home/" not in u:
            continue
        parts = u.split("/home/", 1)[1].strip("/").split("/")
        if len(parts) < 3 or parts[0] in _SKIP_PREFIX:
            continue
        out.append(u)
    return out


def existing_urls() -> set[str]:
    """现有 PDF 页眉的 来源 URL（去重基准）。"""
    from pypdf import PdfReader
    urls = set()
    for p in DATA.glob("*.pdf"):
        try:
            text = PdfReader(str(p)).pages[0].extract_text() or ""
        except Exception:
            continue
        m = re.search(r"来源 URL：\s*(https?://\S+?)(?:\s|$)", text.replace("\n", " "))
        if m:
            urls.add(m.group(1).rstrip("，。"))
    return urls


def round_robin(urls: list[str], target: int) -> list[str]:
    """按顶层分类轮转选样，保证主题广度而不是某一领域堆满。"""
    groups: dict[str, list[str]] = defaultdict(list)
    for u in urls:
        cat = u.split("/home/", 1)[1].split("/", 1)[0]
        groups[cat].append(u)
    picked: list[str] = []
    cats = sorted(groups)
    idx = {c: 0 for c in cats}
    while len(picked) < target:
        progressed = False
        for c in cats:
            if idx[c] < len(groups[c]):
                picked.append(groups[c][idx[c]])
                idx[c] += 1
                progressed = True
                if len(picked) >= target:
                    break
        if not progressed:
            break
    return picked


def fetch(client: httpx.Client, url: str) -> tuple[str, str] | None:
    """抓取并解析词条；返回 (标题, 正文) 或 None。"""
    for attempt in range(3):
        try:
            r = client.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            break
        except (httpx.HTTPError, httpx.TimeoutException):
            if attempt == 2:
                raise
            time.sleep(3)
    soup = BeautifulSoup(r.text, "html.parser")
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""
    body = soup.select_one('div[class*="TopicMainContent_content"]')
    if not title or body is None:
        return None
    # 结构化提取：标题行独立成行；跳过祖先已被选中的元素，避免
    # li 与其内部 p 重复输出同一行（旧文档无此重复，需保持一致）。
    lines: list[str] = []
    selected = body.find_all(["h2", "h3", "h4", "p", "li", "tr"])
    selected_set = set(selected)
    for el in selected:
        if any(parent in selected_set for parent in el.parents):
            continue
        txt = el.get_text(" ", strip=True)
        if txt:
            lines.append(txt)
    content = "\n".join(lines)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if len(content) < MIN_CONTENT_CHARS:
        return None
    return title, content


def safe_name(title: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\s]+', "", title)
    return name[:40] or "未命名"


def make_pdf(title: str, url: str, content: str, out: Path, fetch_date: str) -> None:
    """按既有文档页眉格式生成 PDF（SimHei，与旧文档一致）。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(TTFont("SimHei", FONT_PATH))
    c = canvas.Canvas(str(out), pagesize=A4)
    width, height = A4
    left, top, leading = 18 * mm, height - 20 * mm, 5.6 * mm
    max_chars = 44                       # SimHei 10.5pt 在 A4 宽度下约 44 个中文字

    y = top

    def line(text: str, size: float = 10.5) -> None:
        nonlocal y
        if y < 20 * mm:
            c.showPage()
            y = top
        c.setFont("SimHei", size)
        c.drawString(left, y, text)
        y -= leading * (size / 10.5)

    def para(text: str, size: float = 10.5) -> None:
        # 简单按字符数折行（中文等宽，足够复刻旧文档观感）
        buf = text
        while buf:
            line(buf[:max_chars], size)
            buf = buf[max_chars:]

    para(title, 14)
    y -= 2 * mm
    para("知识类别：疾病与健康知识")
    para("资料来源：默沙东诊疗手册·大众版（病人教育资料，逐字转载）")
    para(f"来源 URL：{url}")
    para(f"获取日期：{fetch_date}")
    para("以下为逐字转载的原文内容：")
    y -= 3 * mm
    for block in content.split("\n"):
        para(block) if block.strip() else line("")
    c.save()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=360, help="新增文档数目标")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    urls = entry_urls()
    done = existing_urls()
    candidates = [u for u in dict.fromkeys(urls) if u not in done]
    print(f"sitemap 词条 {len(urls)}，已收录 {len(done)}，候选 {len(candidates)}",
          flush=True)
    plan = round_robin(candidates, args.target + 40)   # 多取缓冲，补失败损耗
    print(f"本轮计划抓取 {len(plan)}（目标新增 {args.target}）", flush=True)
    if args.dry_run:
        for u in plan[:20]:
            print(" ", u)
        return 0

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) \
        if MANIFEST.exists() else {}
    fetch_date = time.strftime("%Y-%m-%d")
    ok = fail = 0
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True,
                      trust_env=False) as client:
        for i, url in enumerate(plan, 1):
            if ok >= args.target:
                break
            slug = url.rstrip("/").rsplit("/", 1)[-1]
            if slug in manifest:
                continue
            try:
                result = fetch(client, url)
            except Exception as exc:
                manifest[slug] = {"url": url, "status": f"error:{type(exc).__name__}"}
                fail += 1
                print(f"[{i}] FAIL {slug} {type(exc).__name__}", flush=True)
                time.sleep(FETCH_INTERVAL)
                continue
            if result is None:
                manifest[slug] = {"url": url, "status": "skipped"}
                fail += 1
                print(f"[{i}] skip {slug}", flush=True)
                time.sleep(FETCH_INTERVAL)
                continue
            title, content = result
            out = DATA / f"{safe_name(title)}_默沙东诊疗手册.pdf"
            if out.exists():
                manifest[slug] = {"url": url, "status": "duplicate_name",
                                  "file": out.name}
                print(f"[{i}] dup {title}", flush=True)
                time.sleep(FETCH_INTERVAL)
                continue
            make_pdf(title, url, content, out, fetch_date)
            manifest[slug] = {"url": url, "title": title, "file": out.name,
                              "status": "ok", "chars": len(content)}
            ok += 1
            print(f"[{i}] ok {ok}/{args.target} {title} ({len(content)}字)",
                  flush=True)
            if i % 20 == 0:
                MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False,
                                               indent=1), encoding="utf-8")
            time.sleep(FETCH_INTERVAL)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"完成：新增 {ok}，跳过/失败 {fail}，data/ 现有 {len(list(DATA.glob('*.pdf')))} 份",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
