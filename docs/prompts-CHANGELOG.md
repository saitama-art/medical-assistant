# Prompt / 安全词表 变更日志（评估体系 S4a）

> 规则：**改 Prompt 或安全词表 = 改口径**。任何变更必须在此登记（日期、指纹前后值、
> 原因、影响维度），否则跨 run 的分数差异无法归因。
> 指纹由 `backend/acceptance/promptmeta.py` 采集（AST 静态提取，不 import 生产模块；
> 换行归一 LF 后取 SHA-256 前 12 位），随 run JSON 的 `promptmeta` 块落盘。
> 复跑命令（backend/ 下）：`python -m acceptance.promptmeta`（`--json` 输出 meta 块）。

## 指纹口径

| 指纹名 | 内容 | 来源 |
|---|---|---|
| classify | N01 意图分类 system+user | `app/medical_agent/intent.py` CLASSIFY_SYSTEM_PROMPT / CLASSIFY_USER_PROMPT（2026-10-29 起，原 router.py） |
| resolve | N02 上下文补全 system+user | `app/medical_agent/resolver.py` RESOLVE_SYSTEM_PROMPT / RESOLVE_USER_PROMPT（2026-10-29 起，原 orchestrator.py） |
| judge | 证据充分性判定 system+user | `app/rag/service.py` JUDGE_SYSTEM_PROMPT / JUDGE_USER_PROMPT |
| generate | 知识问答生成 system+user | `app/rag/chain.py` MEDICAL_SYSTEM_PROMPT / MEDICAL_HUMAN_PROMPT |
| rewrite | 检索改写模板（f-string → `{expr}` 占位） | `app/rag/store.py` rewrite_query 内局部 prompt |
| safety | 8 个安全词表常量规范化 JSON 整体哈希 | `app/medical_agent/safety.py`：ACUTE_TERMS / FIRST_PERSON / THIRD_PERSON_SUBJECTS / CURRENT_CONTEXT / HELP_SEEKING / NEGATION / NEGATION_EXCEPTIONS / META_QUESTION |

- system 与 user 两段用 `\n\n` 连接后一起哈希，任一段变化指纹都变。
- 数值/配置参数（如 `NEGATION_WINDOW`、`CLAUSE_BREAK`、分类置信门槛）**不在本次
  指纹范围**——S4d 条件指纹统一决策，勿把本表当作"全部行为条件"。

## 基线（2026-10-25 首采）

采集环境：worktree `eval/stage3`（基线 f3aa68a + S3/S4a 改动）。已验证 AST 提取哈希
与**运行时真实常量**哈希逐一相等（classify/resolve/judge/generate/safety 五项；
store.py 因模块级坏 import 无法运行时导入，仅静态提取）。

| 指纹名 | 短哈希 |
|---|---|
| classify | `ce19615d2bc2` |
| resolve | `1d4580276648` |
| judge | `cccfe9cb540e` |
| generate | `4dfc0ef32045` |
| rewrite | `b09bb3c51dd1` |
| safety | `4f2306d51441` |

同日对照：主目录（main，含 S2 的 CS-13 急症语境修复）safety = `84503784e9b7`，
5 份 Prompt 与本基线一致。**两分支 safety 指纹不同 → 急症拦截相关维度不可直接
跨分支比 diff**；合并后须重采基线（S4d 条件指纹规则的实例）。

## 变更记录

（此后新增：日期 ｜ 指纹名 ｜ 旧哈希 → 新哈希 ｜ 原因/关联提交 ｜ 影响维度）

- 2026-10-25 ｜ 全部 ｜ — → 上表基线 ｜ S4a 首采，非行为变更 ｜ 仅新增观测，不改变任何运行行为
- 2026-10-29 ｜ classify ｜ `b73b9519bdcf` → `1246ad459979` ｜ badcase 修复：分类体系补
  "身份/对话元问题与用户自我陈述 → daily"锚点（实测「我是谁」同句 4 次落在
  daily/context_required/unclassifiable 三种处置）；daily.py 的 DAILY_SYSTEM_PROMPT
  规则 3 同步扩展到身份问题（该 Prompt 不在本表 5 份指纹范围内，附记于此）｜
  影响维度：意图分类（daily 覆盖率），知识/急症判定路径不变
