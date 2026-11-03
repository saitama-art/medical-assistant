---
name: badcase-fix
description: 修复用户实测反馈的 badcase（错误回答/错误路由/漏拦）。当用户粘贴或描述线上实测的异常问答时使用。强制归因先行、电池锁定回归、指纹登记收尾。
---

# badcase 修复流程

用户报告的每一个 badcase 都必须走完整流程。禁止跳过归因直接改提示词。

## 步骤

### 1. 归因先行（禁止猜测）

先拿真实证据，再下结论：
- 会话内容：`backend/data/app.db` 的 messages 表按 conversation_id 对账
  （容器部署时查挂载出来的同名库）；
- 响应字段：`outcome` / `category` / `resolved_question`（API 调试字段，
  不落库，看实时响应）；
- 复现：同一输入重跑 ≥3 次。生产模型 Kimi 为 temperature=1（既定风险），
  单次异常先确认是稳定复现还是温度抖动，再归因。

### 2. 定位层级（按链路由近及远排查）

1. 安全规则层：`app/medical_agent/safety.py`（词表覆盖、否定窗口、主体校验）；
2. 意图识别层：`app/medical_agent/intent.py`（分类提示词锚点缺失）；
3. 追问触发/改写层：`app/medical_agent/router.py` 触发器、
   `app/medical_agent/resolver.py` 校验；
4. 证据判定层：`app/rag/service.py` 三段判定；
5. 检索召回层：用 `backend/acceptance/coverage.py` 诊断期望文档是否在 top-k；
6. 内容缺口层：知识库确实无此内容——诚实拒答是**正确行为**，不修代码，
   记「待补充来源」。

### 3. 修复纪律

- 意图理解交给模型：用**示例锚定**边界，不堆关键词；
- 语义保真用**程序校验**（参考 resolver.dropped_negation），不指望模型自觉；
- 用户可见文案只进 `respond.py`；安全相关文案（急症/危机）改动必须提请
  用户逐字审核；
- 不得擅自扩大修复范围（CLAUDE.md §三.4）。

### 4. 电池锁定

- 分类边界类：T-14b 加题（`backend/acceptance/cases.py` 的 CLASSIFY_BATTERY）；
- 其他层：加对应验收组或单测。修复必须有测试锁定，不接受"口头已修"。

### 5. 验证

- 先跑新增/相关单测，再跑全量：`cd backend && python -m unittest discover -s tests`；
- 分类改动跑真实电池：`python -m acceptance --only T-14b`
  （消耗真实 API，先确认余额/限速）；
- **疑似换模漂移**（该题历史上通过、本次不过）：用 git HEAD 的旧 Prompt 做
  A/B 对账。确证是模型漂移则如实记录并提请用户决策——**不得静默改期望**；
- 涉及安全路径（急症/危机/拒答）必须回归 T-03 / T-02b。

### 6. 收尾

- 改了任何 Prompt 常量或 safety 词表 → 执行 `prompt-fingerprint` 流程
  （复采指纹 + 登记 `docs/prompts-CHANGELOG.md`）；
- `docs/开发进度.md` 当日条目：badcase 原文、归因、修复点、电池编号、
  实测结果；
- 提醒用户：8081 容器跑的是镜像代码，需重建后端容器后复测。

## 验收标准

- 新增电池题通过，旧电池零回归；
- 全量单测绿；
- 原 badcase 在真实链路实测修复（或明确记录为内容层缺口、不修）。
