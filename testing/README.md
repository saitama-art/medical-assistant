# Testing · 测试资产

本目录集中存放**用于验证系统行为的测试数据**。测试代码仍靠近被测后端，
评估指标、优化实验和报告仍归 `Evaluation/`，两者不混放。

## 目录边界

| 位置 | 职责 |
|---|---|
| `testing/datasets/` | 可版本化、可独立评审的正式测试数据集 |
| `testing/fixtures/` | 单元测试共享的小型输入、模拟响应和文档片段 |
| `backend/tests/` | 单元测试代码（FakeLLM、纯规则，默认零网络） |
| `backend/acceptance/` | 数据集加载器与真实验收执行器 |
| `Evaluation/` | 指标计算、优化实验、评分卡、基线与评估报告 |

原始验收运行结果仍按既有行为写入 `backend/acceptance_results/`；本目录不放运行产物。

## 当前数据集

| 文件 | 版本 | 数量 | 用途 | 执行器 |
|---|---|---:|---|---|
| `datasets/health72.json` | 2026-10-21 | 72 | 12 个健康领域的知识覆盖测试 | `acceptance.health72` |
| `datasets/scenarios.json` | 2026-10-25 | 22 | 可答、拒答、高风险、多轮、限定与降级场景 | `acceptance.scenarios` |
| `datasets/fr01.json` | 2026-10-20 | 8 | FR-01 引用支持率诊断样本 | `acceptance --fr01` |

三份数据由 2026-10-25 前的 Python 常量原样迁出，题目、标签、顺序和既有 Python 接口不变。
`health72`、`scenarios` 的版本会进入运行条件指纹；`fr01` 版本当前用于数据追溯，
不改变既有评分卡的条件指纹结构。

## 公共 JSON 结构

```json
{
  "name": "health72",
  "version": "2026-10-21",
  "description": "数据集用途",
  "cases": [
    {"id": "A01", "question": "……"}
  ]
}
```

- `name` 必须与文件名一致；`version` 为非空字符串；`cases` 为对象数组。
- 具体字段由套件定义：`health72` 使用 `id/domain/question`；`scenarios` 使用
  `turns/mode/flags/expected_*` 等；`fr01` 使用 `question/expect_keyword`。
- JSON 使用 UTF-8，不写注释。字段语义和断言仍由对应执行器维护。

## 修改纪律

1. 不改题只移动文件时，版本不变；新增、删除或修改题目、标签、期望结果时，必须更新版本。
2. 已在历史报告中使用的数据集不得静默改题；修改需在 `docs/开发进度.md` 记录原因。
3. 数据集只放合成或公开资料，不放真实患者信息、密钥和认证数据。
4. 修改后运行相关单元测试，并核对数量、ID、顺序及版本指纹。

## 运行入口

在 `backend/` 目录、已配置依赖的环境中执行：

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m acceptance.health72
python -m acceptance.scenarios
python -m acceptance --fr01
```

后三类验收会使用真实索引；`scenarios` 和 `acceptance --fr01` 还会调用真实 LLM。
