# 纵向病例分析模块：测试计划与覆盖矩阵

日期：2026-10-28。状态：**测试代码已编写，按任务约束未运行，不得视为已通过。**

## 1. 测试目标与边界

为 `app/case_analysis/`（models / repository / analysis / service）建立可重复、
可自动执行的单元测试集，并给终端 2 的抽取器准备虚构评测数据。

**边界（不测什么）**：
- 病历文本 → 结构化事实的抽取逻辑（终端 2 范围，本计划只有数据集与结构校验）；
- LLM 报告生成、自动诊断/治疗建议（产品红线，永远不测"能否诊断"）；
- 真实患者数据（全部虚构，文件名/正文均含"虚构"标记）。

## 2. 环境隔离方案

- `tests/_case_fixtures.py::CaseTestBase`：每条用例 `TemporaryDirectory` +
  `APP_DB_PATH` 环境变量指向独立 SQLite，setUp 建库、tearDown 清理；
- 不使用、不读取开发库 `backend/data/app.db`；
- 两用户（owner/other）预置，归属用例直接可用；
- 全部确定性：无 LLM、无网络、无时间依赖（时间戳用常量 T0~T4）。

## 3. 用例清单（57 个）

| 类 | 文件 | 用例数 | 覆盖模块/接口 |
|:---|:---|---:|:---|
| A 数据模型与校验 | `test_case_models.py` | 8 | models 枚举/必填/可空时间/JSON 序列化/非法值 → `CaseValidationError` |
| B 持久化与查询 | `test_case_repository.py` | 8 | service 全写读、时间窗/类型/审核过滤、空结果、稳定排序、未知时间、整批回滚、迁移幂等 |
| C 归属与隔离 | `test_case_isolation.py` | 8 | 跨用户读/写/run 操作全部拒绝且**数据未被改动**；防枚举同异常；patient_id/encounter 替换拦截 |
| D 时间线语义 | `test_case_service.py` | 8 | `get_patient_timeline` 排序、event vs recorded、未知时间排尾、同刻稳定、边界闭区间、范围外排除 |
| E 指标比较 | `test_case_analysis.py::TestCaseComparison` | 8 | `compare_observations` delta、unit_mismatch、value_missing、参考范围保留、同刻重复、空范围、原因码、无趋势语言 |
| F 候选关联 | `test_case_analysis.py::TestCaseAssociations` | 6 | `find_temporal_associations` 接近候选、BEFORE/缺失/否定排除、处方缺省提示、永不自动确认 |
| 时间关系矩阵 | `test_case_analysis.py::TestSymptomMedRelationMatrix` | 1 | `symptom_medication_relation` 全分支 |
| G 异常边界 | `test_case_safety.py` | 5 | 非法参数、批量回滚不泄露、非法状态不覆盖、畸形草稿、越权不能标成功 |
| 抽取数据集 | `test_case_extraction_seed.py` | 5 | 种子集结构/虚构标记/枚举合法/语义考察点齐备 |

## 4. 覆盖矩阵

| 能力 | 已编写 | 未编写 | 实现限制无法验证 | 需终端 2 联调 |
|:---|:---:|:---:|:---:|:---:|
| Patient/Encounter/Fact/Med/Obs CRUD | ✅ | | | |
| 归属校验与防枚举 | ✅ | | | |
| 时间线排序与未知时间 | ✅ | | | |
| 指标确定性比较 | ✅ | | | |
| 候选关联生成 | ✅ | | | |
| 分析运行与关联存储 | ✅ | | | |
| 日期粒度时间（无时分秒） | D3 记录现状 | | ⚠️ 模型只有 epoch 秒，无精度标记 | |
| 单位换算比较 | | | ⚠️ 刻意不支持（unit_mismatch） | |
| 病历文本抽取正确率 | 数据集已备 | 抽取逻辑未实现 | | ✅ CE-001~012 |
| 抽取产物入库端到端 | | | | ✅ 抽取器就绪后 |
| 候选关联人工审核流 | | 审核接口未实现 | | 后续版本 |
| HTTP API 层 | | 路由未建 | | 需要时加薄壳再测 |

## 5. 预计高风险缺陷

1. **时间语义混淆**：把 recorded_at 当发生时间参与排序/关联（D 类重点盯）；
2. **越权读**：新增查询函数忘记过 `get_patient` 校验（C 类守门）；
3. **部分写入**：批量保存中途失败留下半截数据（B7/G2 锁定回滚）；
4. **否定语义丢失**：抽取器把"否认高血压"抽成阳性（CE-007 专门考）；
5. **处方当服药**：用药-症状关联把 prescribed 当实际服用（F5 要求显式缺省提示）。

## 6. 当前实现已知限制

- 时间只有 float epoch，无日期精度/时区概念（D3 如实断言现状）；
- 跨单位指标一律 `unit_mismatch`，不做换算（设计决策，非缺陷）；
- `normalized_name` 仅小写归一，未接术语词典；
- 无审核状态变更接口（结构化结果永远 pending，待后续版本）。

## 7. 执行命令

```powershell
cd backend
python -m unittest tests.test_case_models tests.test_case_repository `
    tests.test_case_isolation tests.test_case_service `
    tests.test_case_analysis tests.test_case_safety `
    tests.test_case_extraction_seed
```

## 8. 通过与失败处理

- 全绿前不得宣称基础设施完成；任何失败先归因（测试错/实现错），改最小面；
- 抽取器联调通过 CE-001~012 的结构对比前，不得把 Agent 结果标记为已审核；
- 失败分析永远不得修改原始病历/事实行，只能新建修正记录。
