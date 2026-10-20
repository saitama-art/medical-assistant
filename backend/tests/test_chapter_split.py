"""chapter_split 确定性单测：合成文本验证章节边界切分语义（无 PDF/embedding/网络）。"""

import unittest

from langchain_core.documents import Document

from app.rag.chapter_split import split_chapter_document, split_documents

CHUNK_SIZE = 800
OVERLAP = 100


def leaflet(pages=None):
    """合成一份说明书：大标题 + 若干章节，可按页拆分。"""
    text = (
        "盐酸二甲双胍片说明书\n请仔细阅读说明书并在医师指导下使用\n"
        "【药品名称】\n通用名称：盐酸二甲双胍片\n"
        "【成份】\n本品主要成份为盐酸二甲双胍。\n"
        "【适应症】\n用于单纯饮食控制不满意的2型糖尿病。\n"
        "【用法用量】\n口服，成人开始一次0.25g，一日2~3次，随餐服用。\n"
        "【不良反应】\n常见恶心、呕吐、腹泻。\n"
    )
    if pages is None:
        return [text]
    # 模拟页边界：从中间切开
    mid = len(text) // 2
    return [text[:mid], text[mid:]]


class TestChapterSplit(unittest.TestCase):
    def test_sections_merged_with_title_prefix(self):
        chunks = split_chapter_document(leaflet(), "二甲双胍片", CHUNK_SIZE, OVERLAP)
        self.assertIsNotNone(chunks)
        # 小章节应合并成少量 chunk，且每个 chunk 带文档标题前缀
        self.assertLessEqual(len(chunks), 3)
        self.assertTrue(all(c.startswith("文档：二甲双胍片") for c in chunks))
        # 章节标题不应被截断丢头：【用法用量】必须完整出现在某 chunk 的标题行
        joined = "\n".join(chunks)
        self.assertIn("【用法用量】", joined)
        self.assertIn("口服，成人开始一次0.25g", joined)

    def test_page_boundary_does_not_split_sections(self):
        # 跨两页的同一份说明书：切分结果应与单页一致（页边界不切碎章节）
        one = split_chapter_document(leaflet(), "二甲双胍片", CHUNK_SIZE, OVERLAP)
        two = split_chapter_document(leaflet(pages=2), "二甲双胍片", CHUNK_SIZE, OVERLAP)
        self.assertEqual(one, two)

    def test_oversized_section_subsplit_keeps_heading(self):
        big_section = "【注意事项】" + "用药期间应定期检查肾功能。" * 200  # ~2800 字
        text = "X说明书\n【适应症】\n用于2型糖尿病。\n" + big_section + "\n【禁忌】\n肾功能不全禁用。\n"
        chunks = split_chapter_document([text], "X", CHUNK_SIZE, OVERLAP)
        self.assertIsNotNone(chunks)
        # 大章节被细切为多块，每块都带【注意事项】前缀
        note_chunks = [c for c in chunks if "定期检查肾功能" in c]
        self.assertGreater(len(note_chunks), 1)
        # 子块带「文档标题+章节标题」前缀（缺失标题前缀会导致向量检索排名异常，见 chapter_split.py 注释）
        self.assertTrue(all(c.startswith("文档：X\n【注意事项】") for c in note_chunks))

    def test_page_header_lines_removed(self):
        text = ("X说明书\n第 1 页/共 3 页\n核准日期：2020年1月1日\n【药品名称】\nX片\n"
                "【适应症】\n用于2型糖尿病。\n【禁忌】\n肾功能不全禁用。\n第 2 页/共 3 页\n")
        chunks = split_chapter_document([text], "X", CHUNK_SIZE, OVERLAP)
        joined = "\n".join(chunks)
        self.assertNotIn("第 1 页", joined)
        self.assertNotIn("核准日期", joined)

    def test_no_markers_returns_none(self):
        text = "高血压是指动脉内的压力持续偏高。症状包括头痛。" * 50
        self.assertIsNone(split_chapter_document([text], "高血压", CHUNK_SIZE, OVERLAP))

    def test_split_documents_groups_by_source(self):
        # 两份文档、每份两页：同 source 的页先聚合再切
        docs = []
        for src in ("a.pdf", "b.pdf"):
            for page in leaflet(pages=2):
                docs.append(Document(page_content=page, metadata={"source": f"data/{src}"}))
        out = split_documents(docs, CHUNK_SIZE, OVERLAP)
        self.assertTrue(all(d.metadata["source"].endswith(".pdf") for d in out))
        a_chunks = [d for d in out if d.metadata["source"] == "data/a.pdf"]
        a_one = split_chapter_document(leaflet(), "a", CHUNK_SIZE, OVERLAP)
        self.assertEqual([d.page_content for d in a_chunks], a_one)

    def test_section_after_oversized_keeps_doc_prefix(self):
        # 回归：大章节细切后，其后的普通章节不得丢失「文档：」前缀
        big = "【用法用量】" + "口服，一次0.25g，一日三次。" * 60  # >800 字
        text = "X说明书\n" + big + "\n【不良反应】\n常见恶心、腹泻。\n【禁忌】\n肾功能不全禁用。\n"
        chunks = split_chapter_document([text], "X", CHUNK_SIZE, OVERLAP)
        joined = "\n".join(chunks)
        self.assertIn("文档：X\n【不良反应】", joined)
        # 【禁忌】与【不良反应】合并进同一 chunk 时，该 chunk 必须带文档前缀
        self.assertTrue(all(
            c.startswith("文档：X") for c in chunks if "【禁忌】" in c
        ))

    def test_no_duplicate_heading_in_subpieces(self):
        big = "【注意事项】" + "用药期间应定期检查肾功能。" * 200
        text = ("X说明书\n【适应症】\n用于2型糖尿病。\n" + big +
                "\n【禁忌】\n肾功能不全禁用。\n")
        chunks = split_chapter_document([text], "X", CHUNK_SIZE, OVERLAP)
        for c in chunks:
            self.assertNotIn("【注意事项】\n【注意事项】", c)

    def test_fallback_for_non_leaflet(self):        # 无【】标记的文档走原递归切分，行数与原 splitter 一致
        docs = [Document(page_content="高血压是指动脉压力持续偏高。" * 100,
                          metadata={"source": "data/高血压.pdf"})]
        out = split_documents(docs, CHUNK_SIZE, OVERLAP)
        self.assertGreater(len(out), 1)
        self.assertTrue(all("【" not in d.page_content[:5] for d in out))


if __name__ == "__main__":
    unittest.main()
