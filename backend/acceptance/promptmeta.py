"""promptmeta：5 份 Prompt + 安全词表的 SHA-256 短哈希采集（评估体系 S4a）。

用途（实施方案 V2 §四）：每次 run 的报告带 `promptmeta` 块，记录本次运行实际
使用的 Prompt/安全词表指纹——**改 Prompt/词表 = 改口径**，跨 run 比分数前
先比哈希；哈希不同 → 差异可能来自 Prompt 而非检索/判定逻辑。变更须登记
`docs/prompts-CHANGELOG.md`。

实现取舍：
- **AST 静态提取，不 import 生产模块**：只解析源码取常量值，避免为取几段
  文本拉起 langchain/chroma 等重依赖，也不会触发生产模块的模块级副作用；
- 提取失败（常量改名/删除/改成非字面量）**直接抛错**，不静默降级——指纹
  缺失必须显式暴露，不能产出"看着正常"的空哈希；
- f-string Prompt（store.rewrite_query 的局部 prompt）重建为带 `{expr}`
  占位的模板文本再哈希；
- 哈希前统一换行为 LF：Windows CRLF 检出与 Linux LF 检出指纹一致；
- 短哈希 = SHA-256 前 12 位十六进制（48 bit，指纹/对账用途足够，非安全用途）。

用法（backend/ 下）：
    python -m acceptance.promptmeta           # 表格：名称/哈希/来源
    python -m acceptance.promptmeta --json    # 报告 meta 块原样输出
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]

HASH_ALG = "sha256[:12]"
HASH_LEN = 12

# 5 份 Prompt（实施方案 V2 §四）：(指纹名, 源码相对 backend/ 路径, 常量名)。
# system+user 两段用 "\n\n" 连接后一起哈希——任一段变化指纹都变。
PROMPT_SPECS = (
    ("classify", "app/medical_agent/router.py",
     ("CLASSIFY_SYSTEM_PROMPT", "CLASSIFY_USER_PROMPT")),   # N01 意图分类
    ("resolve", "app/medical_agent/resolver.py",
     ("RESOLVE_SYSTEM_PROMPT", "RESOLVE_USER_PROMPT")),     # N02 上下文补全
    ("judge", "app/rag/service.py",
     ("JUDGE_SYSTEM_PROMPT", "JUDGE_USER_PROMPT")),         # 证据充分性判定
    ("generate", "app/rag/chain.py",
     ("MEDICAL_SYSTEM_PROMPT", "MEDICAL_HUMAN_PROMPT")),    # 知识问答生成
    ("rewrite", "app/rag/store.py", ("prompt",)),           # 检索改写（函数内 f-string）
)

# 安全词表（safety.py 模块级词表常量）：急症拦截规则的全部词汇输入。
SAFETY_WORDLIST_FILE = "app/medical_agent/safety.py"
SAFETY_WORDLIST_NAMES = (
    "ACUTE_TERMS",            # 急症词（9 组）
    "FIRST_PERSON",           # 第一人称
    "THIRD_PERSON_SUBJECTS",  # 第三人称求助主体
    "CURRENT_CONTEXT",        # 当前急症语境
    "HELP_SEEKING",           # 求助语
    "NEGATION",               # 否定词
    "NEGATION_EXCEPTIONS",    # 否定例外搭配
    "META_QUESTION",          # 元问题标记
)


def short_hash(text: str) -> str:
    """换行归一为 LF 后取 SHA-256 前 12 位——跨平台检出指纹一致。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:HASH_LEN]


def _assigned_value(source: str, name: str) -> ast.AST:
    """在模块/函数任意作用域找 name 的赋值表达式（支持 Annotated 赋值）。"""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return node.value
        elif isinstance(node, ast.AnnAssign):
            if (isinstance(node.target, ast.Name) and node.target.id == name
                    and node.value is not None):
                return node.value
    raise KeyError(f"未找到常量 {name}（被改名/删除？promptmeta 需同步更新）")


def _template_text(node: ast.AST) -> str:
    """字面量字符串 → 原文；f-string → 文本 + {expr} 占位模板。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                parts.append("{" + ast.unparse(v.value) + "}")
            else:
                raise TypeError("f-string 含无法静态提取的部分")
        return "".join(parts)
    raise TypeError(f"Prompt 不是字面量/f-string（{type(node).__name__}），"
                    "请改为字面量或更新 promptmeta 提取逻辑")


def extract_template(source: str, name: str) -> str:
    return _template_text(_assigned_value(source, name))


def extract_literal(source: str, name: str):
    return ast.literal_eval(_assigned_value(source, name))


def _read(backend_dir: Path, rel: str) -> str:
    return (backend_dir / rel).read_text(encoding="utf-8")


def prompt_hashes(backend_dir: Path | None = None) -> dict[str, str]:
    """{classify/resolve/judge/generate/rewrite: 短哈希}。"""
    base = backend_dir or _BACKEND_DIR
    out = {}
    for name, rel, consts in PROMPT_SPECS:
        source = _read(base, rel)
        text = "\n\n".join(extract_template(source, c) for c in consts)
        out[name] = short_hash(text)
    return out


def safety_wordlist_hash(backend_dir: Path | None = None) -> str:
    """8 个安全词表常量的规范化 JSON（sort_keys）整体哈希。"""
    base = backend_dir or _BACKEND_DIR
    source = _read(base, SAFETY_WORDLIST_FILE)
    data = {name: extract_literal(source, name) for name in SAFETY_WORDLIST_NAMES}
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return short_hash(canonical)


def prompt_meta(backend_dir: Path | None = None) -> dict:
    """报告 meta 块：{hash_alg, prompt_hashes, safety_wordlist_hash}。"""
    return {
        "hash_alg": HASH_ALG,
        "prompt_hashes": prompt_hashes(backend_dir),
        "safety_wordlist_hash": safety_wordlist_hash(backend_dir),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m acceptance.promptmeta",
                                 description="5 份 Prompt + 安全词表短哈希（零 LLM）")
    ap.add_argument("--json", action="store_true", help="输出报告 meta 块 JSON")
    args = ap.parse_args(argv)
    meta = prompt_meta()
    if args.json:
        print(json.dumps({"promptmeta": meta}, ensure_ascii=False, indent=2))
        return 0
    print(f"hash_alg: {HASH_ALG}")
    for name, rel, _consts in PROMPT_SPECS:
        print(f"{name:10s} {meta['prompt_hashes'][name]}  {rel}")
    print(f"{'safety':10s} {meta['safety_wordlist_hash']}  {SAFETY_WORDLIST_FILE}"
          f"（{len(SAFETY_WORDLIST_NAMES)} 个词表常量）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
