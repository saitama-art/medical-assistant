# 窗口二交接简报（阶段三 · S3 检索与拒答指标）

> 这是自包含简报：你是第二个 Claude Code 实例，在主窗口（阶段二）并发开发。
> 先读完再动手；有任何与简报冲突的现场情况，停下来记录并询问用户，不擅自决策。

## 1. 任务

完成"评估体系建设计划"的**阶段三**（见 `Evaluation/评估计划跟踪表-20261025.md`）：
trace 时延/成本聚合 + 拒答指标纯函数 + `metric/eval.py` 终决。
**只做评估层，零改动 `backend/app/` 生产代码。**

## 2. 环境

- 工作目录（git worktree）：`f:/medical-assistant-s3`，分支 `eval/stage3`（基线 f3aa68a）。
  **只在此目录工作**，绝不碰 `f:/medical-assistant` 主目录（主窗口在用）。
- Python：`G:\annacoda\envs\ai_agent\python.exe`（唯一装齐依赖的解释器，写作 `/g/annacoda/envs/ai_agent/python.exe`）。
- 运行 cwd 必须是 `backend/`；`.env` 已复制到 worktree 根（DeepSeek key）。
- chroma 向量库经绝对路径读主库（`F:\medical-assistant\backend\chroma_db`），**只读**，不要重建/写入。
- 测试命令：`python -m unittest discover -s tests`（worktree 的 backend/ 下）。

## 3. 用户规则（硬性）

1. git commit：author 用 git config 默认值（saitama-art），**不加 Co-Authored-By 尾注**；不 push；每完成一块就 commit 到 `eval/stage3`。
2. 诚实纪律：没跑过的测试不得说通过；API/额度失败记 blocked，绝不记 0 或剔除。
3. **不做真实 LLM 端到端验收**（主窗口统一跑，共享 DeepSeek 额度）；你的验证 = 单测 + 对历史 run JSON 的重算。
4. 冻结集（health72/coverage）不得改题面。

## 4. 文件所有权（只能改这些；其余只读）

| 文件 | 动作 |
|---|---|
| `Evaluation/metrics.py` | **扩展**：新增拒答指标纯函数（见 §5 契约） |
| `backend/acceptance/traceagg.py` | **新建**：时延 P50/P95 聚合 + 分档计分 + 成本估算 |
| `backend/tests/test_refusal_metrics.py`、`backend/tests/test_traceagg.py` | **新建**单测 |
| `metric/eval.py` | **终决**（见 §6） |
| `docs/开发进度.md` | 追加你自己的日期条目（置于文件顶部日期清单内） |

**禁止触碰**：`backend/acceptance/scenarios.py`、`context_metrics.py`、`coverage.py`（主窗口阶段二在用/将建）；`backend/app/` 全部。

## 5. 契约：拒答指标（contract-first，与阶段二解耦）

阶段二（主窗口）会产出 scenario run JSON，schema 约定如下——你的纯函数**按此 schema 写，等真实数据落地后直接可用**：

```json
{
  "kind": "scenarios",
  "records": [
    {"id": "CS-05", "flags": ["refuse"], "expected_outcome": "insufficient_evidence",
     "outcome": "insufficient_evidence", "answer": "……", "sources": [],
     "blocked": false},
    {"id": "CS-09", "flags": ["answerable", "sentinel"], "outcome": "completed", "blocked": false}
  ]
}
```

在 `Evaluation/metrics.py` 新增（纯函数、零 IO、单测锁公式）：

```python
def refusal_rates(records):
    """返回 {correct_refusal_rate, false_refusal_rate, n_refuse, n_answerable,
    n_na}。分母纪律：blocked=True 记 N/A 不入任何分母；flags 语义见契约。
    correct = flags 含 'refuse' 且 outcome == expected_outcome 且 answer 无
    「【参考来源】」段；false_refusal = flags 含 'answerable' 且
    outcome == 'insufficient_evidence'；flags 含 'sentinel'（哨兵观察题）
    只单独计数（n_sentinel_refused），不进 false_refusal 分子但要在返回值里
    暴露（评分卡拒答健康维度用）。"""
def refusal_health_score(records):
    """max(0, correct_refusal_rate*100 - false_refusal_count*5)——硬性指标
    总表第三节公式，下限 0。N/A 时分母为空返回 None（不得记 0）。"""
```

## 6. traceagg.py 规格

输入：一个或多个 run JSON 路径（coverage/health72/门禁报告均可，取 `records[].ms`）。
输出 dict：

```python
def latency_summary(records_ms):
    """{p50_ms, p95_ms, n, tier}；tier 按硬性指标总表第三节分档：
    ≤5s→100 / ≤8s→80 / ≤15s→50 / ≤30s→20 / >30s→0。空列表返回 None。"""
def cost_estimate(records):
    """成本代理（估算口径必须标注）：每 completed 题按知识路径 3 次 LLM 调用
    计，token≈字符数/4（answer 文本）；返回 {avg_calls_per_completed,
    est_input_tokens, note:'估算'}。"""
def aggregate_runs(paths):
    """读多个 run JSON → {latency, cost, sources:[文件名]}；任何文件缺
    records 字段记 warning 不中断。"""
```

## 7. metric/eval.py 终决

决策规则：grep 全仓确认无 import/文档硬引用 → 它是死代码（依赖 store.py，而 store.py
被技术方案 5.7 判不可用；sys.path 指向无效路径）→ **删除并在开发进度记录依据**；
若发现意外引用，改为文件头加"历史原型，不可运行"标注并在进度中说明。

## 8. 参考资料（先读再写）

- 口径权威：`docs/评估体系实施方案V2-上下文优先-20261024.md`（S3 节 + 评分卡节）
- 打分线：`docs/硬性指标总表-20261025.md`
- 指标函数范式：`Evaluation/metrics.py`（照它的纯函数风格写）
- 历史 run JSON 样例：`backend/acceptance_results/`（主目录，只读，可对账）

## 8.5 数据格式勘误（W1 于 2026-10-25 补充，以实测为准）

run JSON 的记录的键**因 harness 不同而不同**（W2 侦察发现，原简报"取 records[].ms"不完全准）：
- coverage / health72 / scenarios 报告：`records[]`，每题有 `ms`；
- **门禁报告**（`full_*.json`、`auth_gate_*.json`）：`results[]`（不是 records），
  每题有 `ms` 与 `ok`；`summary` 块有 `passed/failed`。
`traceagg.aggregate_runs` 必须两种键都兼容（缺失时记 warning 不中断）。

## 9. 验收标准（完成定义）

1. worktree 内 `python -m unittest discover -s tests` 全绿（含你的新单测）；
2. 用历史 run JSON 验证 `aggregate_runs` 能跑通且 p50/p95 与手算一致（挑 1-2 个文件写进进度）；
3. `metric/eval.py` 终决完成且有依据记录；
4. 开发进度条目（本次完成/修改文件/测试结果/遗留）；
5. commit 到 `eval/stage3`（无 Claude 尾注），向用户报告等验收。
