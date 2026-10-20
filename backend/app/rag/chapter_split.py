"""章节结构切分（FR-01）：对带章节标记的文档按章节边界切分。

背景（2026-10-20 真实验收诊断）：说明书类文档用固定 chunk_size=800 硬切，
【用法用量】等章节正文被截断、后续 chunk 以【抬头】等无关词开头且未进 top-k，
导致判定器如实拒答（FR-01 / T-13-1）。本模块把切分对齐到章节边界：

- 含【...】章节标记的文档（国药准字说明书）：按章节切。
- 平台词条（中国医药信息查询平台）：无【】，章节标题是独占一行白名单词
  （适应症/规格/用法用量/...），同样按章节切；目录区（连续标题行）自动识别并跳过，
  chunk 内标题保持原文形态（不加【】，尊重逐字转载）。
- 两种形态共享同一套规则：
  - 每个 chunk 以「文档标题 + 章节标题」开头，保留检索关键词上下文；
  - 连续小章节合并到接近 chunk_size 为止（避免 50 字的小节单独成 chunk）；
  - 超过 chunk_size 的章节（如含表格的用法用量）退回 RecursiveCharacterTextSplitter
    细切，仍保留「文档标题+章节标题」前缀。
- 无章节标记的文档（默沙东诊疗手册等）：维持原有 RecursiveCharacterTextSplitter
  行为不变，避免引入回归。

ingest.py 是唯一调用方；查询侧无感知（chunk 只是向量库中的文档）。
"""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 【药品名称】~【生产企业】等说明书章节标题
SECTION_RE = re.compile(r"【[^】]{1,12}】")

# 平台词条（dayi）的章节标题白名单：独占一行、与词表完全相等才算标题。
# 用白名单而非「短行即标题」启发式，避免正文里的短句被误判为章节边界。
DAYI_HEADINGS = frozenset([
    "药品名称", "成份", "性状", "适应症", "规格", "用法用量", "不良反应",
    "禁忌", "注意事项", "孕妇及哺乳期妇女用药", "儿童用药", "老年用药",
    "药物相互作用", "药物过量", "药理毒理", "药代动力学", "贮藏", "包装",
    "有效期", "执行标准", "批准文号", "生产企业",
])

# 章节合并的粒度上限：超过才细切，小章节尽量合并成接近一整块
_MERGE_TARGET = 800


def _clean_lines(page_texts):
    """页文本拼接 + 页眉页脚行剔除（国药 PDF 有「第 N 页」等，平台 PDF 无）。"""
    return [
        ln for ln in "\n".join(t.strip("\n") for t in page_texts).splitlines()
        if not re.fullmatch(r"第\s*\d+\s*页.*", ln.strip())
        and not re.fullmatch(r"(核准|修订)日期：.*", ln.strip())
    ]


def _find_bracket_sections(lines):
    """国药准字形态：按【标题】分段。返回 (sections, hit_count)。"""
    full = "\n".join(lines)
    matches = list(SECTION_RE.finditer(full))
    if len(matches) < 3:
        return None, len(matches)
    sections: list[str] = []
    if matches[0].start() > 0:
        head = full[: matches[0].start()].strip()  # 文档大标题（说明书名称）
        if head:
            sections.append(head)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full)
        sections.append(full[m.start():end].strip())
    return sections, len(matches)


def _find_dayi_sections(lines):
    """平台词条形态：白名单独占行作标题。

    目录抑制：平台词条开头有一遍目录（标题行连续出现、行间无正文），
    只有「下一行不是标题」的标题才算真正章节起点，目录行自然被跳过。
    """
    heads = []  # (行号, 标题)
    for i, ln in enumerate(lines):
        if ln.strip() in DAYI_HEADINGS:
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt not in DAYI_HEADINGS:
                heads.append((i, ln.strip()))
    if len(heads) < 3:
        return None, len(heads)
    sections: list[str] = []
    if heads[0][0] > 0:
        head = "\n".join(lines[: heads[0][0]]).strip()  # 词条名+来源信息块
        if head:
            sections.append(head)
    for j, (i, _h) in enumerate(heads):
        end = heads[j + 1][0] if j + 1 < len(heads) else len(lines)
        sections.append("\n".join(lines[i:end]).strip())
    return sections, len(heads)


def _section_heading(sec: str) -> str:
    """章节标题 = 首行（若它像标题：【...】 或 dayi 白名单词），否则无标题。

    【】形态兼容标题与正文同一行（【注意事项】正文……）：用 match 取开头的
    【...】即可，不必要求独占一行。
    """
    first = sec.splitlines()[0].strip() if sec else ""
    if SECTION_RE.fullmatch(first) or first in DAYI_HEADINGS:
        return first
    m = SECTION_RE.match(sec)
    if m:
        return m.group(0)
    return ""


def split_chapter_document(page_texts, title: str, chunk_size: int, chunk_overlap: int):
    """把一份文档的各页文本按章节切成 chunks（纯文本列表，含标题前缀）。

    识别顺序：【】形态优先，其次平台白名单形态；都不是则返回 None（调用方回退
    到 RecursiveCharacterTextSplitter）。
    """
    lines = _clean_lines(page_texts)
    sections, _ = _find_bracket_sections(lines)
    if sections is None:
        sections, _ = _find_dayi_sections(lines)
    if sections is None:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    chunks: list[str] = []
    bare_prefix = f"文档：{title}" if title else ""
    buf = bare_prefix

    def flush():
        nonlocal buf
        if buf and buf != bare_prefix:  # 裸前缀不成 chunk（无正文）
            chunks.append(buf)
        buf = bare_prefix

    for sec in sections:
        if len(sec) > chunk_size:
            # 大章节：先冲刷缓冲，再带「文档标题+章节标题」前缀细切。
            # 文档标题前缀不可省：子块正文多为数字/表格时，均值池化向量由
            # 药品名等标识词主导，缺失前缀的子块在「药名+章节」类查询下
            # 排名会跌到同文档其他章节之后（2026-10-20 实测：缺前缀的
            # 用法用量子块在自家文档内排第 11/12 位）。
            # 冲刷后缓冲复位为「文档：标题」，保证后续小章节续入时前缀不断
            # （复位为裸空串会让下一个 chunk 丢失文档标识，实测不良反应段中招）。
            flush()
            heading = _section_heading(sec)
            prefix = f"文档：{title}\n{heading}" if (title and heading) else (title or heading)
            for piece in splitter.split_text(sec):
                # 首个子块本身已含章节标题，剥掉再统一拼前缀，避免标题重复
                body = piece[len(heading):].lstrip("\n") if (heading and piece.startswith(heading)) else piece
                chunks.append(f"{prefix}\n{body}" if prefix else body)
            continue
        candidate = f"{buf}\n{sec}" if buf else sec
        if len(candidate) <= _MERGE_TARGET:
            buf = candidate
        else:
            flush()
            buf = f"文档：{title}\n{sec}" if title else sec
    flush()
    return [c.strip() for c in chunks if c.strip()]


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    """对 Document 列表切分：含章节标记的走章节切分，其余维持原递归切分。

    返回 Document 列表（复用输入对象的 metadata，source 不变）。
    """
    from langchain_core.documents import Document

    fallback = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    out: list[Document] = []
    # PyPDFLoader 每页一个 Document；必须按 source 聚合成整份文档再切，
    # 否则跨页的同一章节被页边界切碎，章节切分失去意义。
    by_source: dict[str, list[Document]] = {}
    order: list[str] = []
    for doc in documents:
        src = doc.metadata.get("source", "")
        if src not in by_source:
            by_source[src] = []
            order.append(src)
        by_source[src].append(doc)

    for src in order:
        docs = by_source[src]
        title = src.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        texts = split_chapter_document(
            [d.page_content for d in docs], title, chunk_size, chunk_overlap
        )
        if texts is None:
            out.extend(fallback.split_documents(docs))
        else:
            meta = docs[0].metadata
            out.extend(Document(page_content=t, metadata=meta) for t in texts)
    return out
