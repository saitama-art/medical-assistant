# 主 Agent 与 F4 路由编排 · 技术实施方案

**版本：** V1.0
**日期：** 2026-10-15
**文档状态：** 技术方案，未经评审，尚未实现
**上游依据：**

- `docs/普通用户版_V1.0_产品需求.md` —— 需求基线。本文不修改该文件。
- `docs/汇总.md` —— 仓库实测盘点结论。**本文不重复盘点**，只在其结论上做技术设计。
- `docs/开发进度.md` —— 进度日志。

**本文包含：** 主 Agent 执行流程与状态、紧急识别/意图识别/RAG/证据判定的关系、RAG 复用方式与失败区分、工具层控制机制、文件清单与调用关系、验收测试、缺失信息。
**本文不包含：** 代码实现。**本轮未修改任何代码。**

---

## 0. 方案前提：复用边界

### 0.1 直接复用（不改行为）

| 复用对象 | 位置 | 用途 |
|:---|:---|:---|
| `get_llm()` | `backend/app/rag/chain.py:20` | 全项目唯一 LLM 入口，已修复 `.env` 读取 |
| `get_embedding()` / `get_vector_store()` | `backend/app/rag/chain.py:41` / `:52` | 指向真实库 `backend/chroma_db` |
| 医疗 Prompt 文本（含免责声明） | `backend/app/rag/chain.py:73-85` | 生成环节复用同一份 Prompt |
| `extract_sources()` 元数据提取逻辑 | `backend/app/rag/chain.py:91` | 来源取自 `doc.metadata`，非模型生成 |

### 0.2 明确不复用（有缺陷，且不在本轮修复范围）

| 对象 | 不复用原因 |
|:---|:---|
| `backend/app/rag/store.py` | 模块级 `from chain import get_llm`（`store.py:10`）非包内相对导入，`import app.rag.store` 必失败；且模块级创建 embedding / vectorstore / reranker，导入即加载模型 |
| `store.py:78` `search()` | `CHROMA_DIR = r"/chroma_db"` 指向无效路径；`similarity_search` 不返回距离，无法做证据判定 |
| `store.py:83-98` rerank / rewrite | 盘点中列为「未实测」，属 P1，本轮不启用 |
| `chain.py:63` `get_rag_chain()` 整体 | 检索、判定、生成写死在同一个闭包里，中间无处插入证据判定。**保留不动**，供现有 `/chat/stream` 继续使用 |

### 0.3 Step 0（前置，必做但极小）

新建 `backend/app/config.py` 集中阈值 / 超时 / 轮数 / 路径；把 `chain.py:57` 硬编码的 `F:\medical-assistant\backend\chroma_db` 和 `chain.py:29` 的 `timeout=35` 改为读配置（**默认值不变，行为不变**）。

此事项汇总第 6 节已列入 P0，本方案直接依赖它。

---

## 1. 主 Agent 的执行流程、状态与条件路由

### 1.1 架构决策：确定性编排，非模型驱动的 tool-calling 循环

**采用：** 显式 Python 流水线，节点间用条件分支连接，工具由编排器按路由结果**直接调用**。

**不采用：** LLM 自主 tool-calling 循环（`while True` + `tool_calls`）、LangGraph。

**理由：**

1. 需求 AG-02 明确「不允许模型觉得需要就调」。若用 tool-calling，模型每轮都可能自主发起调用，需额外机制才能约束；而本方案里「什么条件调什么工具」本就是确定性的，直接 `if` 表达更准、可审计。
2. 需求 AG-05 / AG-06 要求终止条件明确、轮数上限真实生效。显式流水线的最大轮数在编译期就是常量，不存在"模型不肯结束"的失控形态。
3. 需求 F4 要求「简单问题不做无意义的多轮调用」。闲聊 / 就医指引路径在本方案中是 **0 次工具调用、0~1 次 LLM 调用**。
4. 引入 LangGraph 需要状态定义、检查点、图编译，在职责边界未稳定时违背项目原则（与 `docs/汇总.md` 第 8 节结论一致）。

**取舍：** 代价是新增问题类别时需要改代码而非改配置。V1.0 类别集合固定为 5 类（FR-05），可接受。

### 1.2 状态对象（单一 `AgentState`，节点间传递）

```
AgentState
  session_id, turn_id
  raw_question          原始输入
  history               该会话最近 N 轮（已截断）
  context_missing       布尔：上下文已被丢弃，需用户重述
  resolved_question     补全后的问题（无历史时 = raw_question）
  emergency             EmergencyHit{level: strong|weak|none, rule_id, terms, text}
  category              knowledge | triage | emergency | chitchat | unclassifiable
  route                 最终执行路径
  tool_rounds           已消耗轮数
  outcome               RetrievalOutcome（见第 3 节）
  docs / evidence       证据片段与判定结果 {sufficient, strategy, top_score, quote}
  answer_text, sources, risk_notice, disclaimer
  trace                 list[TraceRecord]
  timeout_budget_left   距整体 deadline 的剩余预算
```

**关键约束：** `emergency` 与 `outcome` **互不依赖**。证据不足不能改变紧急结论（FR-04 边界「不得因知识库没有资料而跳过紧急提示」由第 2 节的时序在结构上保证）。

### 1.3 执行流程（N0–N9）

| 节点 | 职责 | 输入 → 输出 | 失败降级 | 消耗轮数 | LLM |
|:---|:---|:---|:---|:---|:---|
| **N0** 输入安全预检 | 对**本轮原始输入**跑紧急规则 | raw → `emergency` | 规则本身异常 → 保守返回就医提示 | 0 | 0 |
| **N1** 上下文补全 | 指代消解：把「那它的副作用呢」补成独立问题 | history + raw → `resolved_question` | LLM 失败 → 用 raw，置 `context_missing=True` | 0 | 0–1（条件触发） |
| **N2** 紧急风险识别 | 对 `resolved_question` **再跑一次**规则 | → `emergency`（合并 N0 结果，取更强） | 同 N0 | 0 | 0 |
| **N3** 意图识别 | 规则短路 + LLM 单次结构化分类 | → `category` | 输出非法 / 超时 → `unclassifiable` | 0 | 0–1 |
| **N4** 路由 | 确定性 if/elif，写 `route` + trace | → `route` | — | 0 | 0 |
| **N5** 工具执行 | Schema 校验 → 超时包装 → 调用 | → 三态结果 | 见第 4 节 | +1 | 0 |
| **N6** 证据充分性判定 | 双闸门（见 3.3） | docs → `sufficient` | **任何异常 → insufficient** | 0 | 0–1 |
| **N7** 生成 | 仅当 N6 判充分 | docs + question → `answer_text` | 超时 / 异常 → GENERATION_ERROR 文案 | 0 | 1 |
| **N8** 回答组装 | 正文 + 来源 + 风险提示 + 免责 | → 最终响应 | — | 0 | 0 |
| **N9** 追踪收尾 | 落 trace、写日志 | → `trace` | 落盘失败不影响响应 | 0 | 0 |

**N0 与 N2 为何拆两次：** N0 对本轮原文（零成本、零延迟）先拦一道，保证疑似急症不被上下文补全的 LLM 调用拖延；N2 再对补全后的问题扫一次，捕获「那个症状呢」这类跨轮指代。两次都是纯规则，成本可忽略。

**为什么 N2 要在 N1 之后：** 用户前一轮说「胸口疼」，本轮说「要挂哪个科」。只看本轮原文，紧急信号会丢失。这是 FR-04「输入含多轮上下文」的落地方式。

### 1.4 条件路由表（N4）

| category | 触发条件 | 执行路径 | 工具调用 | 重试 |
|:---|:---|:---|:---|:---|
| `emergency` | N0 / N2 命中 **strong**（第一人称当前症状） | 紧急应答 | `check_emergency_risk`（已在 N0 执行） | 不允许 |
| `chitchat` | 规则寒暄表命中，或分类器判定 | 固定话术 | 无 | 不允许 |
| `knowledge` | 分类器判定 | RAG 问答 | `search_medical_knowledge` | 最多 1 次 |
| `triage` | 分类器判定且非紧急 | 一般就医指引 | `check_emergency_risk`（取引导文案） | 不允许 |
| `unclassifiable` | 分类器失败 / 超时 / 输出非法 / 类别置信不足 | 无法处理 + 引导 | 无 | 不允许 |

**不存在的路径：** 没有"默认进知识库问答"这一 fallback。分类器异常时走 `unclassifiable`（FR-05 失败场景要求「不得默认进入知识库问答」）。

### 1.5 紧急规则的强弱分级（N0 / N2）

需求第 10 节待确认 9 指出「紧急词出现在非紧急语境」的行为未定义。本方案给出**可配置的默认策略**，待确认后只改配置、不改结构：

- **强信号（拦截，直接返回紧急应答）：** 第一人称 / 当前时态的急症描述。命中词表 ∧ 出现主语标记（`我 | 本人 | 现在 | 刚才 | 突然 | 一直`）∧ 不含元问题标记。
- **弱信号（不拦截，走常规流程 + 顶部加风险提示）：** 命中急症词，但同句出现元问题标记（`说明书 | 是否提到 | 副作用里 | 禁忌 | 这个药会不会`）。
- **判定异常：** 按需求 9.3.5「保守」原则，返回就医提示，**不继续常规问答**。

急症词表初值（可配置，≥8 条，对应 FR-04 验收）：胸痛 / 胸闷剧痛、呼吸困难 / 喘不上气、意识障碍 / 昏迷 / 抽搐、大出血 / 呕血 / 咯血、剧烈头痛、言语不清 / 一侧无力、误服过量、高热惊厥。

---

## 2. 紧急识别 / 意图识别 / RAG / 证据充分性判定 的关系

```
raw_question
   │
   ├─► N0 紧急规则（纯规则，0 LLM，0 检索）──strong──► 立即返回紧急应答 ──► 结束
   │                                        └─异常──► 保守就医提示 ──► 结束
   ▼
 N1 上下文补全（LLM，条件触发）
   ▼
 N2 紧急规则（纯规则）──strong──► 立即返回紧急应答 ──► 结束
   │ (weak: 记 emergency_hint=True，继续)
   ▼
 N3 意图识别（LLM 单次分类）
   ▼
 N4 路由 ──knowledge──► N5 检索 ──► N6 证据判定 ──充分──► N7 生成 ──► N8 组装
                              │            │
                              │            └─不足 / 异常──► 固定「证据不足」应答 ──► N8
                              └─检索失败──► 固定「服务暂不可用」应答 ──► N8
```

四条关系的明文约定：

1. **紧急识别优先于意图识别，且不依赖它。** 紧急路径不产生 LLM 分类调用、不产生检索调用（流程 3「不等待检索结果」）。
2. **紧急识别不依赖 RAG。** 时序上在 N5 之前完成，因此「知识库无资料」在结构上不可能影响紧急结论（FR-04 第 5 条边界）。
3. **证据充分性判定严格在生成之前。** N6 判定不通过时，N7 根本不被调用 —— 无关片段**永远不进 Prompt**（FR-02 第 1 条边界）。这不是让模型自己说"我不确定"。
4. **意图识别只决定路径，不参与证据判定。** 分类器不判断"库里有没有"，那是 N6 的职责，两者不重叠。

---

## 3. RAG 复用方式与失败区分

### 3.1 复用方式：新增 `service.py`，不改 `chain.py` 行为

`backend/app/rag/service.py` 提供三个可组合函数，全部落在**同一个 vectorstore** 上（来自 `chain.py:52` `get_vector_store()`）：

| 函数 | 实现 | 复用来源 |
|:---|:---|:---|
| `retrieve_with_scores(question, k)` | `vectorstore.similarity_search_with_score(question, k=k)` | 同一 vectorstore；返回 `[(Document, distance)]` |
| `judge_evidence(question, scored_docs)` | 见 3.3 | Prompt 风格参照 `chain.py:73` |
| `generate_answer(question, docs)` | `medical_prompt` + `llm.invoke()` | **完全复用** `chain.py:73-85` 的 Prompt 文本与 `chain.py:20` `get_llm()` |

**关于 Prompt 复用的一处最小改动（属 Step 0）：** 把 `chain.py:73-85` 的 `medical_prompt` 与 `chain.py:91` 的 `extract_sources` 提为模块级常量 / 函数，`get_rag_chain()` 内部改为引用它们。**对外行为零变化**，`/chat/stream` 回归不变。这样避免"同一份医疗 Prompt 抄两遍、日后改一处漏一处"。

> **说明：这里没有重写 RAG。** `store.py` 的三个检索函数各是一行 `similarity_search`，本方案绕开它们的原因是它们**不返回距离、路径指向无效目录、模块级导入不可用**，而非要替换检索实现。底层 vectorstore、embedding、chroma 库、329 个 chunk 全部原样复用。

### 3.2 三态 + 两态的严格区分（对应 FR-08 边界）

`RetrievalOutcome` 枚举，**五种状态互斥，用户看到五种不同提示**：

| outcome | 含义 | 判定条件 | 用户可见文案（固定，不含医学结论） |
|:---|:---|:---|:---|
| `SUFFICIENT` | 检索成功且证据充分 | Gate1 ∧ Gate2 通过 | 正常作答 |
| `INSUFFICIENT` | **无检索依据**（最常发生） | 检索成功返回片段，但 Gate1 或 Gate2 不过 | 「知识库暂无该问题的足够依据，无法据此给出结论。建议线下咨询医生。」 |
| `EMPTY_RESULT` | **检索成功但空结果**（库为空 / 集合不存在） | 调用未抛异常，返回 0 条 | 「知识库暂无相关资料。」（与上一档措辞区分） |
| `RETRIEVAL_ERROR` | **检索失败** | 调用抛异常 / 超时 | 「知识库检索服务暂时不可用，请稍后重试。」 |
| `GENERATION_ERROR` | **生成失败** | 检索与判定均通过，LLM 调用异常 / 超时 | 「回答生成服务暂时不可用，请稍后重试。」 |

**`EMPTY_RESULT` 与 `RETRIEVAL_ERROR` 必须不同** —— 这是 FR-08 验收里明写的边界项，上表即是它的落地。

四类文案统一放 `respond.py` 常量表，禁止在各节点内联字符串。

**任何文案都不得拼接 `str(e)`。** 现有 `main.py:86` 的 `f"...{str(e)}"` 会把内部信息吐给用户，新接口不得沿用。

### 3.3 证据充分性判定：双闸门

`docs/汇总.md` 第 4 节的关键结论是：**仅靠距离阈值无法解决「指标类问题」**（实测距离 0.596 命中说明书，但文档里并不包含所问的答案）。因此判定必须是两道闸门：

**Gate 1 —— 距离阈值（确定性，必跑，0 成本）**

- `top1_distance ≤ EVIDENCE_THRESHOLD`（初值 0.85，来自实测分离带）
- 不通过 → 直接 `INSUFFICIENT`，不进入 Gate 2，不调用 LLM
- **可审计、可复现**：阈值来源与逐条距离写入 trace

**Gate 2 —— 模型判别 + 逐字校验（默认开启，可配置关闭）**

- 输入：`resolved_question` + top-k 片段
- 输出严格结构：`{"sufficient": bool, "supporting_quote": str}`
- **关键防幻觉机制：** `supporting_quote` 必须**逐字**出现在上下文中；程序做字符串包含校验，校验失败 → 判为 `INSUFFICIENT`。模型不能凭"感觉像是"判充分。
- 以下任一情况 → `INSUFFICIENT`：输出非法 JSON、超时、抛异常、quote 不在原文中。
- 这满足 FR-02「判定环节失败时必须保守降级为证据不足，不得降级为正常作答」。

**配置开关：** `EVIDENCE_JUDGE = threshold | llm | both`，默认 `both`。关闭 Gate 2 可省一次 LLM 调用，但会放过指标类误判 —— 默认不开这个口子。

### 3.4 来源展示（FR-03）顺带解决

N8 只展示**通过 Gate 2 的那 1–2 个切片**的来源，而不是把 top-k 全堆在末尾 —— 这正好满足 FR-03「来源要与回答中的**具体结论**对应，而不是把全部检索结果堆在末尾」。

- 文档名 / 页码取自 `Document.metadata`（与 `chain.py:91` 同一逻辑），**不经过模型**，因此 FR-03「来源不得由模型生成」在结构上成立。
- 元数据缺 `page` 时只输出文档名，不推测页码（FR-03 边界）。
- 检索为空时 `sources = []`，测试用例「把检索结果置空后回答中不得出现任何文档名」自动通过。

---

## 4. 工具层：校验、轮数、超时、异常、追踪

### 4.1 工具注册表与调用条件（AG-02、AG-03）

| 工具 | 允许调用的条件 | 参数 Schema | 禁止情形 |
|:---|:---|:---|:---|
| `search_medical_knowledge` | `route == knowledge` 且 `tool_rounds < MAX` | `{question: str(1..500), k: int(1..10)=3}` | 紧急路径、triage、闲聊、unclassifiable 一律不调 |
| `check_emergency_risk` | N0 / N2（`text` = 原始或补全后问题）；triage 路径取引导文案 | `{text: str(1..2000)}` | 纯计算场景 |
| `get_document_detail` | **P1**，本轮不实现 | — | — |
| `explain_indicator_reference` | **不实现**（FR-09 属 P1，知识库无资料） | — | — |

**参数校验执行位置：** 编排器在调用前校验，**校验不通过不得发出调用**，直接记 `ToolStatus.INVALID_ARGS` 并按路由的降级文案响应。校验用 `pydantic.BaseModel`（FastAPI 已带，无新增依赖）。

**为什么工具表这么短：** 需求 5.1 已把「判断类别、决定路径、组织回答、控制轮数」划归 Agent 本体。工具只保留"取外部信息"这一类职责。BMI 之类的确定性计算不进工具表（`docs/汇总.md` 7.4 同结论）。

### 4.2 轮数限制（AG-05、AG-06）

- `MAX_TOOL_ROUNDS = 3`（配置项，对应需求待确认 5 的建议值）
- **本方案每条路径的实际轮数：** knowledge 1 次，证据不足时允许 1 次 query 改写重试 = 最多 **2** 次；其余路径 0 次。
- 唯一的重试触发条件：Gate 1 通过但 Gate 2 判定不足 **且** `tool_rounds < MAX` **且** 剩余预算 ≥ 单次工具超时。三个条件同时满足才重试，否则直接返回「证据不足」。
- **不存在 `while True`。** 循环体是 `for round in range(MAX_TOOL_ROUNDS)`，天然的终止上界。
- 达到上限且未得到可用结果 → 按 AG-05 边界要求「明确说明」，返回「未能完成本次查询，请补充信息或稍后重试」，**不静默截断**。此路径在 V1.0 属理论路径（实际最多 2 轮），但仍实现并测试。

### 4.3 超时

三层预算，全部来自 config：

| 层级 | 值 | 实现方式 |
|:---|:---|:---|
| 单次工具（检索） | 10s | `ThreadPoolExecutor` + `future.result(timeout=...)`，超时抛 `ToolTimeout` |
| 单次 LLM 调用 | 15s | `get_llm()` 的 `http_client` timeout 改读配置（现为硬编码 35s） |
| 整体请求 | 30s | `deadline = time.monotonic() + 30`，每个节点入口先检查剩余预算；剩余不足时**不再启动新的重试轮**，直接走当前最佳结果的降级文案 |

超时必须是**可判定的**：用线程池包装后，`concurrent.futures.TimeoutError` 与业务异常是不同类型，因此「检索超时」与「检索异常」都能被明确区分并写入 trace。

> **注意：** 线程池包裹同步调用只能让调用方"及时放弃"，不能真正中断底层线程。V1.0 接受这一点（检索是本地 CPU 推理，会在后台自然结束），但**必须在 trace 中记录为 timeout，不得当作成功**（AG-04）。

### 4.4 异常体系（AG-04、FR-08）

```
AgentError
 ├── ToolError ──┬── ToolInvalidArgs   → 不发出调用
 │                ├── ToolTimeout       → 该工具的降级文案
 │                └── ToolRuntimeError  → 该工具的降级文案
 ├── RetrievalError                     → RETRIEVAL_ERROR 文案
 ├── EvidenceJudgeError                 → **保守降级为 INSUFFICIENT**（不是正常作答）
 └── GenerationError                    → GENERATION_ERROR 文案
```

**三条硬规则：**

1. 异常 → 固定文案映射表，**永远不产生医学结论**（FR-08「任何异常都不得降级为编造一个回答」）。
2. 工具失败**不得被当作成功**，也不得把空结果当作"成功但无内容"混为一谈（AG-04）。
3. 未知异常兜底 → 「服务暂时不可用，请稍后重试」，同样不含 `str(e)`。

### 4.5 执行追踪（AG-07、AG-08）

`TraceRecord`：`{seq, node, decision, tool, args_digest, status, latency_ms, round, timeout_budget_left}`

- 每次路由决策与工具调用各记一条，节点名与 1.3 表一致。
- `args_digest` 只记**参数摘要**（长度 + 短 hash），不记用户问题原文；原始问题全文不进日志。
- **理由：** 需求待确认 7 指出用户输入可能含个人健康信息，原需求完全未提及隐私处理。在确认前，追踪记录保留"定位问题所需"的元数据（类别、outcome、距离、耗时、轮数），不落盘原文 —— 既不牺牲可定位性，也不擅自扩大数据留存范围。
- 排查需要原文时用环境变量显式开启调试模式，且在配置注释中标注该风险。
- trace 中**绝不含** API Key、堆栈、内部路径；响应体默认只返回 `category / outcome / sources`，调试模式才返回完整 trace。

---

## 5. 文件清单与调用关系

### 5.1 新增

| 文件 | 职责 |
|:---|:---|
| `backend/app/config.py` | **Step 0**。阈值、超时、轮数、路径、判定开关、急症词表路径。被下述所有模块读取 |
| `backend/app/rag/service.py` | RAG 组合层：`retrieve_with_scores` / `judge_evidence` / `generate_answer`，不含任何编排逻辑 |
| `backend/app/medical_agent/state.py` | `AgentState`、`QuestionCategory`、`RetrievalOutcome`、`ToolStatus`、`EmergencyHit`、`TraceRecord` |
| `backend/app/medical_agent/safety.py` | 紧急规则：词表 + 强弱分级 + 元问题语境判别。**纯函数，无 LLM、无 IO** |
| `backend/app/medical_agent/router.py` | 寒暄短路 + LLM 结构化分类 + 输出校验 + 确定性路由表 |
| `backend/app/medical_agent/tools.py` | 工具注册表、pydantic 参数校验、超时包装、轮数计数、异常归一化 |
| `backend/app/medical_agent/respond.py` | 全部固定文案常量表（证据不足 / 空结果 / 检索失败 / 生成失败 / 紧急 / 就医指引 / 无法处理 / 免责） |
| `backend/app/medical_agent/orchestrator.py` | `run_agent(state) -> AgentResult`。N0–N9 的编排与条件分支，**唯一持有控制流的地方** |
| `backend/app/medical_agent/session.py` | 会话上下文：`{session_id: deque(maxlen=2*MAX_TURNS)}`，进程内内存 |
| `backend/tests/test_medical_agent_*.py` | 验收测试（见第 6 节） |

**目录选择说明：** 新建 `backend/app/medical_agent/`，**不放进** `backend/app/agent/`。原因是该目录下 18 个文件全为与医疗无关的演示，需求待确认 12 也提出"开发时易被误认为主 Agent 已存在"。归档演示代码不在本轮范围，用独立目录隔离是零成本的规避方式。

### 5.2 修改（三处，均为增量）

| 文件 | 改动 | 为什么必须改 |
|:---|:---|:---|
| `backend/app/rag/chain.py` | `medical_prompt` / `extract_sources` 提为模块级；`CHROMA_DIR`、`timeout` 改读 config | 供 `service.py` 复用，避免 Prompt 双份；默认值不变 |
| `backend/app/rag/main.py` | **新增** `POST /agent/chat`（JSON 进、JSON 出），`/chat/stream` 与 `/upload` **不动** | 新接口需要 session_id、结构化返回（`answer / sources / category / outcome`）。旧接口保持兼容 |
| `backend/requirements.txt` | 补齐实际 import 的包 | `docs/汇总.md` 第 6 节已列 |

**不做的事：** 不修 `store.py`、不动 `ingest.py`、不动前端、不动 `backend/multi_agent_business/`、不引入 LangGraph。伪流式、引用展开交互、真实流式均属 P1。

### 5.3 调用关系（单向，无循环依赖）

```
main.py /agent/chat
   └── orchestrator.run_agent
          ├── session.py          取 / 存上下文
          ├── safety.py           N0 / N2 紧急规则
          ├── router.py           N3 / N4（内部用 chain.get_llm）
          ├── tools.py            N5（校验 / 超时 / 计数）
          │      └── rag/service.py
          │             ├── chain.get_vector_store / get_embedding
          │             ├── chain.get_llm
          │             └── chain.medical_prompt
          └── respond.py          固定文案
   state.py 被以上全部引用（只含数据类，无逻辑）
```

---

## 6. 验收测试

测试文件与 1.3 节点一一对应；除标注「集成」外全部**不依赖网络**（monkeypatch LLM 与 vectorstore）：

| # | 用例 | 对应需求 | 方式 | 通过条件 |
|:---|:---|:---|:---|:---|
| T-01 | 情绪类闲聊输入 | FR-05 / AG-06 | 直接调 orchestrator | 0 工具调用、0 LLM 调用，返回固定话术 |
| T-02 | 5 类问题各 ≥5 条（mock 分类器） | FR-05 正常 | 参数化 | 路由结果 100% 与预期一致 |
| T-03 | ≥5 条模糊输入 | FR-05 边界 | 同上 | 全部落 `unclassifiable`，无一条被强行归类 |
| T-04 | 分类器抛异常 | FR-05 失败 | monkeypatch raise | 走「无法处理」，**未进入检索** |
| T-05 | 急症清单 ≥8 条 | FR-04 正常 | 纯规则 | 100% 命中 strong，返回含"急诊 / 120"，且**不含病因与用药** |
| T-06 | 元问题语境（如"二甲双胍说明书是否提到胸痛"） | FR-04 边界 | 纯规则 | 按待确认 9 的默认策略：不拦截，加风险提示 |
| T-07 | 紧急规则抛异常 | FR-04 失败 | monkeypatch raise | 返回保守就医提示，**不进入常规问答** |
| T-08 | 库内无资料的紧急问题 | FR-02 安全 | 端到端 | 命中紧急提示，**未因"无资料"跳过** |
| T-09 | 空结果 vs 检索异常 | **FR-08 边界** | monkeypatch | 两种情形**文案不同**且均不含结论 |
| T-10 | Gate1 不通过（构造 distance=0.95） | FR-02 正常 | 固定分数 | `INSUFFICIENT`，**LLM 生成未被调用** |
| T-11 | Gate2 quote 不在原文中 | FR-02 正常 | 固定 LLM 输出 | 判为 `INSUFFICIENT` |
| T-12 | Gate2 抛异常 | **FR-02 失败** | monkeypatch raise | 降级 `INSUFFICIENT`，**未降级为正常作答** |
| T-13 | 检索 / 生成各环节逐一注入异常 | FR-08 失败 | monkeypatch | 逐条检查输出，**无一条含医学结论** |
| T-14 | 回答 / 提示文本 | FR-08 安全 | 字符串断言 | 不含 key 前缀、`Traceback`、盘符路径 |
| T-15 | 检索结果置空 | FR-03 安全 | 端到端 | 回答中**不出现任何文档名** |
| T-16 | 元数据缺 `page` | FR-03 边界 | 构造 Document | 只显示文档名，无推测页码 |
| T-17 | 工具参数非法（k=999 / question 超长） | AG-03 | 直接调 tools | 记 `INVALID_ARGS`，**底层检索未被调用** |
| T-18 | 工具超时 | AG-04 / AG-06 | 注入 sleep 工具 | 记 timeout，**按失败处理，不当成功** |
| T-19 | 强制重试场景 | AG-05 边界 | 构造连续不足 | 轮数 ≤ 3，终止时输出明确说明，**无静默截断** |
| T-20 | 无限循环检测 | AG-05 失败 | 注入重复调用 | 循环 0 次；工具失败不被当成功 |
| T-21 | 多轮指代（"那它的副作用呢"） | FR-06 正常 | 端到端 | 补全后结果 == 直接输入完整表述的结果 |
| T-22 | 超长会话 + 上下文丢失 | FR-06 边界 / 失败 | 清空 history | 明确要求重述，**不猜测指代对象** |
| T-23 | 两会话并发 | FR-06 安全 | 并发调用 | 输出归属正确，无串用 |
| T-24 | trace 完整性 | AG-07 / AG-08 | 检查 trace | 每次决策与调用均有记录，含 latency / round / outcome，不含 key |
| T-25 | **集成**：药品题 ≥20 条 | FR-01 正常 | 真实 LLM + RAG | 引用支持率 ≥0.85（人工核对表） |
| T-26 | **集成**：无依据题 4 类各 ≥2 条 | FR-02 正常 | 真实链路 | 100% 返回证据不足，文本不含结论 |

**判定通则（照抄需求 8.3）：** 出现"无证据却输出结论 / 伪造来源 / 未触发紧急提示 / 异常时编造回答 / 无终止循环"任一情形，本轮验收不通过，不看平均分。

---

## 7. 缺失信息（只列具体项，未据此扩大检查）

| # | 缺失项 | 影响 | 处理 |
|:---|:---|:---|:---|
| M-1 | `get_vector_store()` 在目标 conda 环境 `ai_agent` 下能否真实运行 | 决定 `RETRIEVAL_ERROR` 分支是否会在开发期频繁触发 | 不阻塞方案。Step 0 第一件事就是跑一次 `count == 329` 复验 |
| M-2 | `similarity_search_with_score` 返回距离的度量类型（L2 / squared-L2 / cosine） | 决定阈值 0.85 的语义 | 不阻塞结构。阈值一律由测试集校准，初值仅占位 |
| M-3 | 前端是否在本轮切到 `/agent/chat` | 决定 FR-03 来源展示是否有人看得到 | 本方案只做后端；前端切换建议留到下轮（`docs/汇总.md` 已列为需修复项） |

**关于 M-2：** 因为度量类型未定，方案里所有判定都用**同一个函数返回的同一个距离值**，Gate 1 只在"同一刻度内"比较，因此即使度量类型未确认，方案结构也不会错，只有具体阈值需要校准后填。

---

## 8. 实施顺序与验收标准

| 步骤 | 内容 | 完成标志 |
|:---|:---|:---|
| Step 0 | `config.py` + `chain.py` 三处小改 + 依赖补齐 | 复验 `count == 329`；`/chat/stream` 行为不变 |
| Step 1 | state / respond / safety（纯规则，无依赖） | T-05、T-06、T-07 通过 |
| Step 2 | `service.py` + 双闸门判定 | T-09 ~ T-12、T-15、T-16 通过 |
| Step 3 | `tools.py`（校验 / 超时 / 轮数 / 异常） | T-17 ~ T-20 通过 |
| Step 4 | router / session / orchestrator + `/agent/chat` | T-01 ~ T-04、T-13、T-14、T-21 ~ T-24 通过 |
| Step 5 | 集成测试与阈值校准 | T-25、T-26 通过；产出距离分布与阈值报告 |

**本轮验收标准：** T-01 ~ T-24（免网络）全绿；T-25、T-26 完成人工核对表；`docs/开发进度.md` 按增量方式追加本次记录。

**本轮显式不做：** 前端改造、真实流式、Reranker / Query Rewrite 启用、指标解释（FR-09）、演示代码归档、Multi-Agent。

---

## 9. 待确认事项（沿用需求第 10 节编号）

本方案对以下待确认项给出了不等确认即可先实现的默认策略，确认后只需改配置：

| 编号 | 待确认事项 | 本方案的默认处理 |
|:---|:---|:---|
| 5 | AG-06 的具体数值 | 轮数 3、单次工具超时 10s、整体 30s（需求建议值），全部为 config 项 |
| 6 | FR-06 上下文保留轮数 | 最近 5 轮，config 项 |
| 7 | 隐私处理承诺 | trace 不落盘用户原文，仅记元数据；如需原文须显式开启调试模式 |
| 9 | 非紧急就医指引是否给具体科室 / 元问题语境处理 | triage 只给通用指引不给科室；紧急词强弱分级策略（见 1.5） |
| 10 | 阈值 0.85 与各项比例目标 | 0.85 仅为 Gate 1 初值，Step 5 用测试集校准后替换 |
| 12 | `backend/app/agent/` 演示代码归档 | 本轮不归档，改用新建 `backend/app/medical_agent/` 隔离 |
