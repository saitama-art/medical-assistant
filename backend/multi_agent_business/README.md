# Multi-Agent Business Analysis (Supervisor Architecture)

基于真实 UCI Online Retail 数据集、DeepSeek 和 LangGraph 的多 Agent 业务分析项目。

## 架构

```text
用户问题
   |
Supervisor 主 Agent（制定任务计划）
   |
按计划分发给专业 Agent（并行执行）
   |-- revenue_agent：收入/订单分析
   |-- customer_agent：客户分析
   |-- product_agent：商品分析
   |
Validator Agent（检查证据充分性）
   |-- 充分 --> Report Agent --> END
   |-- 不足 --> Targeted Research --> Validator Agent
```

Supervisor 负责协调和分发；专业 Agent 负责各自领域分析；Validator 负责证据检查。LangGraph 管理共享 State、并行、路由和循环。

## 1. 环境准备

建议 Python 3.11+。在项目目录运行：

```bash
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，填写真实 DeepSeek API Key。

## 2. 数据准备

从 UCI 官方页面下载 Online Retail 数据集：
https://archive.ics.uci.edu/dataset/352/online+retail

把 `Online Retail.xlsx` 放到项目根目录的 `data/Online Retail.xlsx`。

## 3. 运行

```bash
python main.py
```

默认问题可在 `main.py` 的 `QUESTION` 中修改。报告写入 `output/business_report.md`。

## 4. 设计说明

- `Send` 用于把 Supervisor 生成的任务动态分发给多个专业 Agent。
- `findings` 使用 `operator.add` reducer，汇总并行节点的返回结果。
- 验证不通过时只触发定向补充研究，不会重新运行全部专业 Agent。
- `MAX_RESEARCH_ROUNDS` 限制补充研究轮数；达到上限后会生成带有“证据仍有限”说明的报告，而不是伪装成验证通过。
- UCI 数据是 2010–2011 年的历史零售数据，适合练习分析工作流，不代表当前市场趋势。

首次运行前请检查本机安装的 LangGraph/LangChain 版本。此项目是完整的模块化起点，但我没有在你的本地环境实际执行过。
