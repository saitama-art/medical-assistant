"""promptmeta（S4a：5 份 Prompt + 安全词表短哈希）确定性单测。

锁定口径：
- 恰好 5 份 Prompt（classify/resolve/judge/generate/rewrite）+ 安全词表整体哈希；
- AST 静态提取（不 import 生产模块）；f-string 重建为 {expr} 占位模板；
- 换行归一 LF 后哈希（CRLF/LF 检出指纹一致）；
- 常量缺失/非字面量 → 显式抛错，不静默产出空指纹；
- coverage / health72 报告带 promptmeta 块（打桩 client，零 LLM）。
"""

import re
import unittest

from acceptance.promptmeta import (HASH_LEN, extract_literal,
                                   extract_template, prompt_hashes,
                                   prompt_meta, safety_wordlist_hash,
                                   short_hash)

HEX12 = re.compile(r"^[0-9a-f]{12}$")


class TestShortHash(unittest.TestCase):
    def test_crlf_lf_equivalent(self):
        self.assertEqual(short_hash("a\r\nb"), short_hash("a\nb"))
        self.assertEqual(short_hash("a\rb"), short_hash("a\nb"))

    def test_deterministic_and_sensitive(self):
        self.assertEqual(short_hash("你是分类器"), short_hash("你是分类器"))
        self.assertNotEqual(short_hash("你是分类器"), short_hash("你是分类器。"))
        self.assertEqual(len(short_hash("x")), HASH_LEN)


class TestExtraction(unittest.TestCase):
    def test_plain_and_annotated_assignment(self):
        src = 'A = """你好\n{question}"""\nB: tuple[str, ...] = ("不", "没")\n'
        self.assertEqual(extract_template(src, "A"), "你好\n{question}")
        self.assertEqual(extract_literal(src, "B"), ("不", "没"))

    def test_fstring_becomes_placeholder_template(self):
        src = 'def f(q):\n    prompt = f"""前缀{q}后缀"""\n    return prompt\n'
        self.assertEqual(extract_template(src, "prompt"), "前缀{q}后缀")

    def test_implicit_concat_is_single_constant(self):
        src = 'A = ("甲"\n     "乙")\n'
        self.assertEqual(extract_template(src, "A"), "甲乙")

    def test_missing_constant_raises(self):
        with self.assertRaises(KeyError):
            extract_template("A = 'x'\n", "NOPE")

    def test_non_literal_prompt_raises(self):
        # 动态拼接的 Prompt 不得静默哈希——显式报错要求人工处理
        with self.assertRaises(TypeError):
            extract_template("A = build_prompt()\n", "A")


class TestRealSources(unittest.TestCase):
    def test_five_prompts_plus_safety(self):
        hashes = prompt_hashes()
        self.assertEqual(set(hashes),
                         {"classify", "resolve", "judge", "generate", "rewrite"})
        for name, h in hashes.items():
            self.assertRegex(h, HEX12, name)
        self.assertRegex(safety_wordlist_hash(), HEX12)

    def test_meta_shape_and_determinism(self):
        m1, m2 = prompt_meta(), prompt_meta()
        self.assertEqual(m1, m2)
        self.assertEqual(m1["hash_alg"], "sha256[:12]")
        self.assertEqual(set(m1), {"hash_alg", "prompt_hashes",
                                   "safety_wordlist_hash"})
        self.assertEqual(m1["prompt_hashes"], prompt_hashes())

    def test_prompts_are_real_content(self):
        # 防"提取到空串/错常量"：每份模板须含其身份关键词
        from pathlib import Path
        base = Path(__file__).resolve().parents[1]
        cases = [
            ("app/medical_agent/intent.py", "CLASSIFY_SYSTEM_PROMPT", "意图分类器"),
            ("app/medical_agent/resolver.py", "RESOLVE_SYSTEM_PROMPT",
             "上下文整理器"),
            ("app/rag/service.py", "JUDGE_SYSTEM_PROMPT", "证据充分性判定器"),
            ("app/rag/chain.py", "MEDICAL_SYSTEM_PROMPT", "参考文档"),
            ("app/rag/store.py", "prompt", "检索优化助手"),
        ]
        for rel, const, marker in cases:
            text = extract_template((base / rel).read_text(encoding="utf-8"), const)
            self.assertIn(marker, text, f"{rel}:{const}")
            self.assertGreater(len(text), 10)

    def test_safety_wordlists_complete(self):
        # 8 个词表全部纳入指纹；任一增删都会改变哈希
        from pathlib import Path
        base = Path(__file__).resolve().parents[1]
        src = (base / "app/medical_agent/safety.py").read_text(encoding="utf-8")
        terms = extract_literal(src, "ACUTE_TERMS")
        self.assertEqual(len(terms), 9)  # 9 组急症词
        self.assertIn("不", extract_literal(src, "NEGATION"))


class TestReportWiring(unittest.TestCase):
    def test_coverage_report_has_promptmeta(self):
        from acceptance.coverage import run_coverage

        class _Fake:
            def retrieve(self, q):
                return {"outcome": None, "top": [{"doc": "高血压", "distance": 0.1,
                                                  "content": "c"}]}

            def chat(self, sid, q):
                return 200, {"outcome": "completed"}, 1

        rep = run_coverage(do_chat=False, group="p0",
                           client_factory=lambda: _Fake())
        self.assertEqual(rep["promptmeta"], prompt_meta())

    def test_health72_report_has_promptmeta(self):
        # health72.run 内部自建 TestClient，打桩其依赖而非真实 App（零 LLM）
        from unittest.mock import patch
        from acceptance import health72

        class _FakeClientCtx:
            def __init__(self, *a, **kw):
                pass

            def retrieve(self, q):
                return {"outcome": None, "top": []}

            def chat(self, sid, q):
                return 200, {"outcome": "completed"}, 1

        with patch("acceptance.runner.ClientCtx", _FakeClientCtx), \
                patch("fastapi.testclient.TestClient", lambda *a, **kw: object()), \
                patch("app.rag.main.app", object()):
            rep = health72.run(do_chat=False)
        self.assertEqual(rep["promptmeta"], prompt_meta())
        self.assertEqual(rep["n"], 72)


if __name__ == "__main__":
    unittest.main()
