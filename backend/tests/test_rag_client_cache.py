"""N07 检索环节的客户端生命周期：embedding 与 vectorstore 必须是进程内单例。

回归目标（门禁 G-3「整体上限真实生效」的根因之一）：

    设计文档 5.6 规定「embedding 模型与 vectorstore 在**启动时预热**，避免首次
    请求承担模型加载耗时（可达数十秒），这部分耗时**不计入请求预算**」，并以此
    作为「检索不引入线程池」的前提。该前提只有在**启动预热的对象与请求路径用的
    是同一个对象**时才成立。

    修改前的实现每次调用都新建 `HuggingFaceEmbeddings`（构造函数即加载模型权重），
    预热对象被立即丢弃，请求路径照旧重新加载：实测 N07 单次检索 9.7~15.6s，而
    N07 的配置预算是 5s（设计 5.6 预算表），整体请求实测 50.1s > 45s。

本测试**不加载真实模型**，保持本套件「确定性单测、无网络、无模型集成」的性质：
向 sys.modules 注入伪造的 langchain_huggingface / langchain_chroma 并统计构造次数。

在 backend/ 目录下运行：

    python -m unittest tests.test_rag_client_cache -v
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from app.rag import chain


class _CountingEmbeddings:
    """伪造 HuggingFaceEmbeddings：构造即计数，模拟"构造=加载模型权重"这一事实。"""

    instances: list["_CountingEmbeddings"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _CountingEmbeddings.instances.append(self)


class _CountingChroma:
    """伪造 Chroma：构造即计数。"""

    instances: list["_CountingChroma"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _CountingChroma.instances.append(self)


def _fake_modules():
    """构造两个伪造模块，供 `from X import Y` 在函数内部解析。"""
    hf = types.ModuleType("langchain_huggingface")
    hf.HuggingFaceEmbeddings = _CountingEmbeddings
    chroma = types.ModuleType("langchain_chroma")
    chroma.Chroma = _CountingChroma
    return {"langchain_huggingface": hf, "langchain_chroma": chroma}


class RagClientCacheTest(unittest.TestCase):
    """embedding / vectorstore 的单例性质。"""

    def setUp(self):
        # 保存并清空模块级缓存，保证用例之间互不影响、也不依赖执行顺序。
        # 这里刻意用 getattr 容忍"缓存属性不存在"的实现：把 chain.py 还原成修改前
        # 的版本时，失败必须落在**断言的语义**上（"两次取用不是同一对象"），而不是
        # 落在 setUp 的 AttributeError 上——否则这个测试守的是属性名，不是行为。
        self._saved = (
            getattr(chain, "_embedding", None),
            getattr(chain, "_vector_store", None),
        )
        chain._embedding = None
        chain._vector_store = None
        _CountingEmbeddings.instances = []
        _CountingChroma.instances = []

    def tearDown(self):
        chain._embedding, chain._vector_store = self._saved

    # ---------------- embedding ----------------

    def test_embedding_is_process_singleton(self):
        """连续两次取用必须返回同一对象，且只构造一次（= 只加载一次模型）。"""
        with patch.dict(sys.modules, _fake_modules()):
            first = chain.get_embedding()
            second = chain.get_embedding()

        self.assertIs(first, second)
        self.assertEqual(len(_CountingEmbeddings.instances), 1)

    def test_embedding_kwargs_unchanged(self):
        """构造参数与修改前逐字一致——本改动只改"何时构造"，不改"怎么构造"。"""
        with patch.dict(sys.modules, _fake_modules()):
            chain.get_embedding()

        self.assertEqual(
            _CountingEmbeddings.instances[0].kwargs,
            {
                "model_name": "BAAI/bge-small-zh",
                "model_kwargs": {"device": "cpu"},
                "encode_kwargs": {"normalize_embeddings": True},
            },
        )

    # ---------------- vectorstore ----------------

    def test_vector_store_is_process_singleton(self):
        with patch.dict(sys.modules, _fake_modules()):
            first = chain.get_vector_store()
            second = chain.get_vector_store()

        self.assertIs(first, second)
        self.assertEqual(len(_CountingChroma.instances), 1)

    def test_warmup_object_is_the_one_requests_use(self):
        """门禁 G-3 的直接断言：启动预热取到的对象，必须就是后续取到的同一个对象。

        这条断言在修改前是 **False**（原实现每次返回新实例），也就是启动预热
        对请求路径完全无收益。
        """
        with patch.dict(sys.modules, _fake_modules()):
            warm = chain.get_vector_store()          # main.py startup 预热
            in_request = chain.get_vector_store()    # N07 请求路径

        self.assertIs(warm, in_request)
        self.assertEqual(len(_CountingChroma.instances), 1)

    def test_vector_store_uses_the_cached_embedding(self):
        """vectorstore 必须绑定同一个 embedding 单例，不能各拿一份。"""
        with patch.dict(sys.modules, _fake_modules()):
            store = chain.get_vector_store()
            embedding = chain.get_embedding()

        self.assertIs(store.kwargs["embedding_function"], embedding)
        self.assertEqual(len(_CountingEmbeddings.instances), 1)

    # ---------------- 失败与惰性 ----------------

    def test_construction_failure_is_not_cached(self):
        """构造失败不得写入缓存，否则会把一次瞬时故障固化成一整个进程的不可用。"""
        broken = types.ModuleType("langchain_huggingface")

        def _raise(**kwargs):
            raise RuntimeError("model_load_failed")

        broken.HuggingFaceEmbeddings = _raise
        with patch.dict(sys.modules, {"langchain_huggingface": broken}):
            with self.assertRaises(RuntimeError):
                chain.get_embedding()
            self.assertIsNone(chain._embedding)

        # 故障恢复后仍能正常构造
        with patch.dict(sys.modules, _fake_modules()):
            self.assertIsNotNone(chain.get_embedding())

    def test_import_does_not_touch_langchain(self):
        """惰性构造性质：模块导入后缓存为空，不得在导入期构造任何客户端。

        这条性质是 main.py 在缺 langchain_chroma 时仍能启动的原因，缓存不得破坏它。
        """
        self.assertIsNone(chain._embedding)
        self.assertIsNone(chain._vector_store)


if __name__ == "__main__":
    unittest.main()
