# 测试目录索引（tests/ 唯一入口）

生成：2026-10-28，静态盘点（AST 统计，未运行）。规模：**36 个文件 / 558 个用例**。

## 分层定义

| 层 | 含义 | 本目录实例 |
|:---|:---|:---|
| 单元 | 单模块/函数行为 | calculator、safety、chapter_split、case_* |
| 集成 | 多模块交互（FakeLLM/打桩） | medical_agent_step4、api_auth、context_followup |
| 回归 | 锁定历史 bug | emergency_negation_scope、judge_dedupe |
| 评测口径 | 指标/评分卡/报告公式 | scorecard、report、trend、refusal_metrics、context_metrics |
| 数据资产 | 数据集结构校验 | datasets、golden_seed、case_extraction_seed |
| 辅助 | fixture（不被收集） | _case_fixtures |

统一入口命令：`cd backend && python -m unittest discover -s tests`（当前基线见开发进度）。

## 索引表

| 文件 | 模块 | 目的 | 用例 | 外部依赖 | 重叠风险 | 建议 |
|:---|:---|:---|---:|:---|:---|:---|
| test_acceptance_harness.py | 验收框架 | 验收 runner 的用例选择/红acted/失败语义 | 13 | TestClient 打桩 | 低 | 保留 |
| test_api_auth.py | 鉴权·接口 | 认证/CSRF/越权矩阵/上传治理（HTTP 级） | 25 | 临时库+TestClient | 与 test_auth 分层互补 | 保留 |
| test_auth.py | 鉴权·领域 | 密码哈希/登录态/限流/会话归属（非 HTTP） | 29 | 临时 SQLite | 与 api_auth 分层互补 | 保留 |
| test_baseline.py | 评估·基线 | baseline 任命规则与条件指纹可比性 | 11 | 临时目录 | 与 datasets 不同层 | 保留 |
| test_calculator.py | Agent·工具 | BMI/标准体重/换算的解析与执行 | 33 | FakeLLM | 低 | 保留 |
| test_case_analysis.py | 病例分析 | 指标比较 E + 候选关联 F（15） | 15 | 临时库 | 与 case 系列同 fixture，无重复 | 保留 |
| test_case_extraction_seed.py | 病例分析·数据 | 抽取评测种子集结构校验 | 5 | 无 | 低 | 保留 |
| test_case_isolation.py | 病例分析 | 跨用户读写/run 越权矩阵 C | 8 | 临时库 | 低 | 保留 |
| test_case_models.py | 病例分析 | 模型与基础校验 A | 8 | 临时库 | 低 | 保留 |
| test_case_repository.py | 病例分析 | 持久化与查询 B + 迁移幂等 | 8 | 临时库 | 低 | 保留 |
| test_case_safety.py | 病例分析 | 异常与安全边界 G | 5 | 临时库 | 低 | 保留 |
| test_case_service.py | 病例分析 | 时间线语义 D | 8 | 临时库 | 低 | 保留 |
| test_chapter_split.py | RAG | 章节切分（说明书/平台词条/通用） | 12 | 无 | 低 | 保留 |
| test_complex_handler.py | Agent | 复杂处理层决策（澄清/未知/失败/修复） | 14 | 无 | 低 | 保留 |
| test_config.py | Agent | 预算配置单源与守卫 | 3 | 无 | 低 | 保留 |
| test_context_engineering.py | 上下文工程 | 任务层/事实/ContextPackage/SQLite 闭环 | 31 | 临时库 | 与 followup/metrics 分层 | 保留 |
| test_context_followup.py | Agent·上下文 | 隐性回指触发、约束保留、响应契约 | 19 | FakeLLM+HTTP | 与 context_engineering 分层 | 保留 |
| test_context_metrics.py | 评估·上下文 | 上下文五指标公式 | 17 | 无 | 低 | 保留 |
| test_coverage.py | 评估 | coverage 分类逻辑单测 | 9 | 无 | 与 audit 不同对象 | 保留 |
| test_coverage_audit.py | 测试资产 | coverage_audit 工具口径与引用校验 | 5 | 临时目录 | 与 coverage 不同对象 | 保留 |
| test_datasets.py | 测试资产 | testing/datasets 结构与版本指纹 | 6 | 临时目录 | 与 golden_seed 互补 | 保留 |
| test_emergency_negation_scope.py | Agent·安全 | 否定窗口/偏侧词表安全回归 | 28 | 无 | 低 | 保留 |
| test_golden_seed.py | 测试资产 | 黄金种子集结构/审核/哈希校验 | 4 | 临时目录 | 低 | 保留 |
| test_judge_dedupe.py | RAG | 判定上下文去重与全量回退 | 9 | FakeLLM | 低 | 保留 |
| test_medical_agent.py | Agent·安全 | 急症规则：strong/元问题/否定/规则故障 | 32 | 无 | 与 step4 分层互补 | 保留 |
| test_medical_agent_step4.py | Agent·集成 | router/session/orchestrator/API 全链路 | 58 | FakeLLM | 与 medical_agent 分层互补 | 保留 |
| test_promptmeta.py | 评估 | promptmeta 指纹采集与报告接线 | 13 | 无 | 低 | 保留 |
| test_rag_client_cache.py | RAG | embedding/向量库客户端单例化 | 7 | 打桩 | 低 | 保留 |
| test_refusal_metrics.py | 评估 | 拒答率与健康分公式 | 14 | 无 | 低 | 保留 |
| test_report.py | 评估 | report 聚合与安全判定 | 27 | 临时目录 | 低 | 保留 |
| test_scorecard.py | 评估 | 评分卡五维度公式 | 23 | 无 | 低 | 保留 |
| test_session_persist.py | Agent·会话 | 会话文件持久化与回答剥离（**保护文件**） | 8 | 临时目录 | 低 | 保留（禁改） |
| test_task_spec.py | Agent | TaskSpec 授权与 Worker 结果验收 | 13 | 无 | 低 | 保留 |
| test_traceagg.py | 评估 | trace 时延/成本聚合与百分位 | 13 | 临时目录 | 低 | 保留 |
| test_trend.py | 评估 | 趋势提取/环比/CLI 渲染 | 25 | 临时目录 | 低 | 保留 |
| _case_fixtures.py | 病例分析 | 共享 fixture（临时库+构造器） | 0 | — | 被 6 个 case 文件依赖 | 保留 |

## 重点重叠核查结论

| 组合 | 结论 | 依据 |
|:---|:---|:---|
| test_medical_agent vs step4 | **不重复** | 前者只测 safety.py 规则本体（无编排）；后者测编排/路由/API 集成，FakeLLM 打桩 |
| test_auth vs test_api_auth | **不重复** | 领域层（哈希/会话/限流/归属）vs HTTP 层（Cookie/CSRF/越权矩阵） |
| test_coverage vs test_coverage_audit | **不重复** | acceptance.coverage 分类逻辑 vs testing/tools 审计工具口径 |
| baseline/golden_seed/datasets | **不重复** | 基线任命规则、黄金集结构、数据集指纹，三个不同资产 |
| context_engineering/followup/metrics | **不重复** | 持久化闭环、触发与契约、指标公式，三层 |
| test_case_analysis vs _case_fixtures | **不重复** | fixtures 是依赖件，非测试 |

## 维护建议汇总

- **保留**：全部 36 个文件均有独立覆盖价值，本轮无合并项；
- **待确认**：无（全部静态可判定）；
- **纪律**：`test_session_persist.py` 为保护文件，任何治理不得修改；
  新增病例分析测试必须复用 `_case_fixtures.py`，不得另建临时库方案；
  评测口径文件（scorecard/report/trend 等）改公式必须先改测试。
