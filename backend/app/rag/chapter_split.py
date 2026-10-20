"""章节结构切分（FR-01）：对带【标题】章节标记的文档按章节边界切分。

背景（2026-10-20 真实验收诊断）：说明书类文档用固定 chunk_size=800 硬切，
【用法用量】等章节正文被截断、后续 chunk 以【抬头】等无关词开头且未进 top-k，
导致判定器如实拒答（FR-01 / T-13-1）。本模块把切分对齐到章节边界：

- 含【...】章节标记的文档（国药准字说明书、中国医药信息查询平台）：按章节切。
  - 每个 chunk 以「文档标题 + 章节标题」开头，保留检索关键词上下文；
  - 连续小章节合并到接近 chunk_size 为止（避免 50 字的小节单独成 chunk）；
  - 超过 chunk_size 的章节（如含表格的【用法用量】）退回 RecursiveCharacterTextSplitter
    细切，仍保留章节标题作前缀。
- 无章节标记的文档（默沙东诊疗手册等）：维持原有 RecursiveCharacterTextSplitter
  行为不变，避免引入回归。

ingest.py 是唯一调用方；查询侧无感知（chunk 只是向量库中的文档）。
"""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 【药品名称】~【生产企业】等说明书章节标题
SECTION_RE = re.compile(r"【[^】]{1,12}】")

# 章节合并的粒度上限：超过才细切，小章节尽量合并成接近一整块
_MERGE_TARGET = 800


def _has_chapter_markers(text: str) -> bool:
    return len(SECTION_RE.findall(text)) >= 3


def split_chapter_document(page_texts, title: str, chunk_size: int, chunk_overlap: int):
    """把一份文档的各页文本按章节切成 chunks（纯文本列表，含标题前缀）。"""
    full = "\n".join(page_texts)
    if not _has_chapter_markers(full):
        return None  # 回退：调用方用 RecursiveCharacterTextSplitter

    # 去掉页眉页脚行（如「第 1 页/共 10 页」「核准日期：」「修订日期：」），避免混入 chunk；
    # 各页文本两端 \n 去掉后以单个 \n 连接，避免页边界出现空行（页首/页尾的换行都剥掉）
    lines = [
        ln for ln in "\n".join(t.strip("\n") for t in page_texts).splitlines()
        if not re.fullmatch(r"第\s*\d+\s*页.*", ln.strip())
        and not re.fullmatch(r"(核准|修订)日期：.*", ln.strip())
    ]
    full = "\n".join(lines)

    # 按章节标题分段；标题保留在段首
    matches = list(SECTION_RE.finditer(full))
    sections: list[str] = []
    if matches and matches[0].start() > 0:
        head = full[: matches[0].start()].strip()  # 文档大标题（说明书名称）
        if head:
            sections.append(head)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full)
        sections.append(full[m.start():end].strip())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    chunks: list[str] = []
    buf = f"文档：{title}" if title else ""
    for sec in sections:
        if len(sec) > chunk_size:
            # 大章节：先冲刷缓冲，再带「文档标题+章节标题」前缀细切。
            # 文档标题前缀不可省：子块正文多为数字/表格时，均值池化向量由
            # 药品名等标识词主导，缺失前缀的子块在「药名+章节」类查询下
            # 排名会跌到同文档其他章节之后（2026-10-20 实测：缺前缀的
            # 【用法用量】子块在自家文档内排第 11/12 位）。
            # 冲刷后缓冲复位为「文档：标题」，保证后续小章节续入时前缀不断
            # （复位为裸空串会让下一个 chunk 丢失文档标识，实测不良反应段中招）。
            if buf:
                chunks.append(buf)
            buf = f"文档：{title}" if title else ""
            heading = sec.splitlines()[0] if sec.startswith("【") else ""
            prefix = f"文档：{title}\n{heading}" if (title and heading) else (title or heading)
            for piece in splitter.split_text(sec):
                # 首个子块本身已含章节标题，剥掉再统一拼前缀，避免【标题】重复
                body = piece[len(heading):].lstrip("\n") if (heading and piece.startswith(heading)) else piece
                chunks.append(f"{prefix}\n{body}" if prefix else body)
            continue
        candidate = f"{buf}\n{sec}" if buf else sec
        if len(candidate) <= _MERGE_TARGET:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            buf = f"文档：{title}\n{sec}" if title else sec
    if buf:
        chunks.append(buf)
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
