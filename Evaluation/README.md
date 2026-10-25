# Evaluation · 评估资产总目录

本目录是项目评估体系的**内容与工具统一出口**（2026-10-25 确立约定）：

| 位置 | 内容 | 说明 |
|---|---|---|
| `metrics.py` | 检索指标纯函数（Recall@K/MRR/top1/失败构成） | 零依赖 CLI：`python Evaluation/metrics.py <run.json>...`；S3 接线 coverage 时经 sys.path 导入（路径 `Evaluation/`，注意跨目录） |
| `reports/` | 评估产出归档（**以后每次评估的文件都放这里**） | 命名 `reports/<YYYYMMDD>/<运行名>/`，含 scorecard.json、MD 报告、原始 run JSON 副本 |
| `baseline.json` | 评分卡基线（分数+条件指纹，S4 落地） | 仅全量 run 可任命为基线，换基线须记原因 |
| `experiments/` | 一次性实验脚本与原始数据（如 PA-03 rerank 评估） | 从 `backend/scripts/` 迁入的历史实验也归这里 |
| `Model.docx` | 历史遗留文件 | 与本体系无关，暂留 |

**边界约定**：
- 评估**代码**中需要被 `backend/` 导入的模块才放 `backend/acceptance/`；
  纯工具/纯计算的脚本放本目录（metrics.py 属此类：零依赖、不被 import）。
- 原始运行数据默认仍落 `backend/acceptance_results/`（既有 harness 行为不变），
  进入报告体系的副本/汇总放本目录 `reports/`。
- 口径唯一权威：`docs/评估体系实施方案V2-上下文优先-20261024.md`；
  硬性指标对照表：`docs/硬性指标总表-20261025.md`。
