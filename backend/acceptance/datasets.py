"""根目录 testing/datasets 的只读加载器。

测试执行代码仍留在 backend/acceptance；版本化题目与标签独立存放，便于查看、
评审和复现。数据集内容变化必须同步提升其 version，供运行条件指纹比较。
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_DIR / "testing" / "datasets"


def load_dataset(name: str, dataset_dir: Path | None = None) -> dict:
    """读取并校验数据集公共结构；具体字段由各测试套件解释。"""
    path = (dataset_dir or DATASET_DIR) / f"{name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: 数据集顶层必须是对象")
    if payload.get("name") != name:
        raise ValueError(f"{path}: name 必须为 {name!r}")
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"{path}: version 必须是非空字符串")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not all(isinstance(c, dict) for c in cases):
        raise ValueError(f"{path}: cases 必须是对象数组")
    return payload


def dataset_version(name: str) -> str:
    """供报告和运行条件指纹使用，避免在代码里重复硬编码版本。"""
    return load_dataset(name)["version"]
