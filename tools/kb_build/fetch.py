"""抓取清单中的外部资料，逐字保存原始 HTML 与正文文本到 kb_src/。

- kb_src/<site>/<key>.html  原始 HTML（可追溯原件）
- kb_src/<site>/<key>.txt   正文提取（逐字，按块换行）
- 失败条目写入报告 dict 并打印，不冒充实测。

用法（repo 根目录，ai_agent 环境）：
    python -X utf8 tools/kb_build/fetch.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
from manifest import ALL_MSD, DAYI_KNOWN, DAYI_WANTED, MSD_BASE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KB_SRC = ROOT / "kb_src"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
MIN_TEXT_LEN = 250  # 正文过短视为抽取异常。2026-10-20 由 800 下调：实测 MSD 部分
# 主题页（如"肾功能测定"）正文本身仅约 300 字，800 的门槛把真实短页误判为
# 抽取失败；250 仍能拦住抽取异常（异常时正文接近 0）。


def extract_msd(soup: BeautifulSoup) -> str:
    """默沙东大众版：正文容器 class 前缀 Topic_topicContainer__，取其内文字块。"""
    container = None
    for div in soup.find_all("div"):
        cls = " ".join(div.get("class") or [])
        if cls.startswith("Topic_topicContainer__"):
            container = div
            break
    if container is None:
        raise ValueError("未找到 Topic_topicContainer 正文容器")
    blocks = []
    for el in container.find_all(["h1", "h2", "h3", "p", "li", "td", "th"]):
        t = el.get_text(" ", strip=True)
        if t:
            blocks.append(t)
    return "\n".join(blocks)


def extract_dayi(soup: BeautifulSoup) -> str:
    """中国医药信息查询平台：article.article-detail-container 字段块。"""
    art = soup.select_one("article.article-detail-container")
    if art is None:
        raise ValueError("未找到 article-detail-container 正文容器")
    blocks = []
    for el in art.find_all(["h1", "h2", "h3", "dt", "dd", "p", "li"]):
        t = el.get_text(" ", strip=True)
        if t:
            blocks.append(t)
    return "\n".join(blocks)


def fetch_one(url: str, site_dir: Path, key: str, extractor) -> dict:
    rec = {"key": key, "url": url}
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        rec["status"] = r.status_code
        r.raise_for_status()
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / f"{key}.html").write_text(r.text, encoding="utf-8")
        text = extractor(BeautifulSoup(r.text, "html.parser"))
        if len(text) < MIN_TEXT_LEN:
            raise ValueError(f"正文过短（{len(text)} 字符），疑似抽取异常")
        (site_dir / f"{key}.txt").write_text(text, encoding="utf-8")
        rec["chars"] = len(text)
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = str(e)
    return rec


def resolve_dayi_urls() -> list[tuple[str, str, str]]:
    """在 dayi 药品列表分页中解析 WANTED 药品的详情页 id。"""
    resolved = list(DAYI_KNOWN)
    found: dict[str, str] = {}
    for page in range(1, 30):
        if all(w in found for w in DAYI_WANTED):
            break
        url = f"https://www.dayi.org.cn/list/4_{page}.html" if page > 1 else "https://www.dayi.org.cn/list/4.html"
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                break
        except requests.RequestException:
            break
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a"):
            href = a.get("href") or ""
            name = a.get_text(strip=True)
            m = re.fullmatch(r"/drug/(\d+)\.html", href)
            if not m:
                continue
            for w in DAYI_WANTED:
                if w in name and w not in found:
                    found[w] = "https://www.dayi.org.cn" + href
        time.sleep(0.5)
    for w in DAYI_WANTED:
        if w in found:
            resolved.append((f"drug-{w}", found[w], w))
        else:
            print(f"[dayi] 未在列表中解析到：{w}（记录为未获取）")
    return resolved


def main() -> None:
    results = []
    msd_dir = KB_SRC / "msd"
    for key, path, title, cat in ALL_MSD:
        rec = fetch_one(MSD_BASE + path, msd_dir, key, extract_msd)
        rec.update(title=title, category=cat, site="msd")
        results.append(rec)
        print(("OK " if rec["ok"] else "FAIL"), key, rec.get("chars", rec.get("error", "")))
        time.sleep(0.5)

    dayi_dir = KB_SRC / "dayi"
    for key, url, title in resolve_dayi_urls():
        rec = fetch_one(url, dayi_dir, key, extract_dayi)
        rec.update(title=title, category="药品说明书", site="dayi")
        results.append(rec)
        print(("OK " if rec["ok"] else "FAIL"), key, rec.get("chars", rec.get("error", "")))
        time.sleep(0.5)

    report = ROOT / "kb_src" / "fetch_report.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in results if r["ok"])
    print(f"\n抓取完成：{ok}/{len(results)} 成功，报告见 {report}")


if __name__ == "__main__":
    main()
