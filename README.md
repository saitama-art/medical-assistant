# 医疗健康智能助手（V1.0）

RAG + 可控 Agent 的医疗健康智能助手：基于知识库证据的健康问答、证据充分性判定、
紧急症状拦截、就医指引、多轮对话与来源展示。

- 需求与设计：[`docs/V1.0-需求与功能设计.md`](docs/V1.0-需求与功能设计.md)、
  [`docs/普通用户版_V1.0_产品需求.md`](docs/普通用户版_V1.0_产品需求.md)、
  [`docs/技术方案_主Agent与F4路由编排.md`](docs/技术方案_主Agent与F4路由编排.md)
- 开发进度：[`docs/开发进度.md`](docs/开发进度.md)

## 1. V1.0 功能范围

| 功能 | 状态 |
|:---|:---|
| FR-01 基于知识库证据的健康问答 | ✅ 已交付（引用支持率人工核对进行中，见下"已知限制"） |
| FR-02 证据充分性判定与无依据拒答 | ✅ 已交付 |
| FR-03 知识来源展示（文档名+页码） | ✅ 已交付 |
| FR-04 一般就医指引与紧急风险提示 | ✅ 已交付（规则 + 叠加征象判定） |
| FR-05 普通对话与意图识别 | ✅ 已交付 |
| FR-06 多轮对话（最小形态，进程内会话） | ✅ 已交付 |
| FR-07 资料上传 | ✅ 接口就绪（前端已接线） |
| FR-08 异常处理与回答限制 | ✅ 已交付 |
| FR-09 基础健康指标解释 | ⚠️ 部分覆盖（血压/血常规等有 MSD 词条支撑） |
| 健康计算（BMI/标准体重/单位换算） | ✅ 已交付（确定性工具，0 LLM/0 检索） |

## 2. 项目组成

```
medical-assistant/
├── backend/                 # FastAPI 后端（主 Agent 编排 + RAG）
│   ├── app/
│   │   ├── medical_agent/   # 主 Agent：orchestrator / router / safety / session / subagents
│   │   └── rag/             # RAG：chain / service / ingest / chapter_split
│   ├── acceptance/          # 真实 LLM 验收 harness（python -m acceptance）
│   ├── tests/               # 单元测试（417 个，FakeLLM/纯规则，无网络）
│   ├── scripts/             # calibrate_threshold.py（Step 0b 校准脚本）
│   ├── data/                # 知识库 PDF（79 份）
│   └── chroma_db/           # 向量索引（gitignore，见第 6 节）
├── frontend/                # React 前端（Create React App）
├── testing/                 # 测试数据集与共享 fixtures（见 testing/README.md）
├── Evaluation/              # 指标、优化实验、评分卡与评估报告
└── docs/                    # 需求/设计/进度/阈值校准报告
```

## 3. 环境要求与依赖安装

### 后端（Conda 环境 `ai_agent`，实测 Python 3.10.20）

```powershell
conda create -n ai_agent python=3.10 -y
conda activate ai_agent
pip install -r backend/requirements.txt
```

> 本仓库开发机使用现有 `ai_agent` 环境； requirements.txt 含
> fastapi / uvicorn / langchain-* / chromadb / sentence-transformers / pypdf 等。

### 前端（Node.js，实测 v24.18.0 / npm 11.16.0）

```powershell
cd frontend
npm install
```

## 4. 环境变量配置

后端从**项目根目录 `.env`**（或 `backend/.env`，二选一）读取配置，
`load_dotenv(override=True)`——`.env` 优先于系统环境变量。

必需项（无此项后端启动自检即拒绝服务）：

```dotenv
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

- 用途：`deepseek-chat`（意图分类、证据判定、回答生成）。
- 安全注意：**`.env` 不入库**（已在 .gitignore）；不要把它写进任何文档/代码/提交。
  系统环境变量若存在同名坏值会被 `.env` 覆盖（override=True，本机实测过）。
- Embedding 模型 `BAAI/bge-small-zh` 首次运行自动下载；
  离线建库脚本 `ingest.py` 内已设 `HF_ENDPOINT=https://hf-mirror.com` 镜像加速。

可选项：

```dotenv
SESSION_PERSIST_DIR=backend/sessions   # 会话文件持久化（C-6），不设则纯内存
```

- 设置后会话写回时落盘，服务重启后同一 session_id 的历史自动恢复；
  前端 session_id 已持久化在浏览器 localStorage，两者配合可实现重启不断上下文。

## 5. 启动

### 后端（端口 8000）

```powershell
conda activate ai_agent
cd backend
python -m uvicorn app.rag.main:app --host 127.0.0.1 --port 8000
```

- 启动自检：证据判定两段都不可用即拒绝启动；随后 embedding/向量库预热。
- 首次预热需加载 embedding 模型，等待数十秒属正常。

### 前端（端口 3000）

```powershell
cd frontend
npm start          # 开发模式
# 或
npm run build      # 生产构建（产物在 frontend/build/）
```

- 前端默认连 `http://127.0.0.1:8000`（`REACT_APP_API_BASE` 可覆盖）。

### 访问

- 页面：http://localhost:3000
- API 文档：http://127.0.0.1:8000/docs

### 本地测试账号

| 用户名 | 密码 | 权限 |
|:---|:---|:---|
| `acceptance` | `acceptance-pass-123` | 普通用户，仅用于本地登录与问答验收，无上传权限 |

该账号由 `python -m acceptance` 在本地验收数据库中自动创建；尚未运行验收或提示
“用户名或密码错误”时，可手动创建：

```powershell
cd backend
python scripts/create_user.py acceptance --display-name 测试用户
# 按提示两次输入：acceptance-pass-123
```

> 此凭据仅用于本地开发/测试，不是生产默认账号。部署到共享或公网环境前，必须改用
> 独立强密码或删除该测试账号；也可通过 `ACCEPTANCE_USER`、`ACCEPTANCE_PASSWORD`
> 修改验收账号约定。

### Docker（GPU-ready，端口 8080）

需要 Docker Desktop、WSL2 和支持容器 GPU 的 NVIDIA 驱动。项目根目录必须已有 `.env`；
Compose 会保留现有 `app.db`、Chroma、上传文件和会话文件，**不要重建 Chroma**。

当前 Compose 默认离线加载 `BAAI/bge-small-zh`。首次使用空的 `hf_cache` 卷时，先把本机
已有的同名模型缓存复制进去（不要换成其他 embedding 模型）：

```powershell
docker run --rm `
  -v medical-assistant_hf_cache:/cache `
  -v "${env:USERPROFILE}\.cache\huggingface\hub\models--BAAI--bge-small-zh:/source:ro" `
  python:3.11-slim `
  python -c "import shutil; shutil.copytree('/source', '/cache/hub/models--BAAI--bge-small-zh', dirs_exist_ok=True)"
```

构建两个镜像并启动（当前 Compose/Buildx 需要显式允许构建期 host 网络）：

```powershell
docker buildx bake -f docker-compose.yml --allow=network.host --load
docker compose up -d
docker compose ps
```

- 页面：http://localhost:8080
- OpenAPI JSON：http://localhost:8080/api/openapi.json
- 后端只暴露在 Compose 内网；Nginx 将 `/api/` 转发到后端。
- `backend/data`、`backend/chroma_db`、`backend/uploads`、`backend/session_data`
  使用 bind mount；Hugging Face 缓存使用 `medical-assistant_hf_cache` 命名卷。
- 当前 embedding 仍固定使用 CPU；CUDA 镜像和 `gpus: all` 只是为后续 GPU 工作提供基础，
  不包含 vLLM。
- 本机 HTTP 部署设置 `SESSION_COOKIE_SECURE=false`；HTTPS 部署必须改回 `true`。

验证 GPU：

```powershell
docker compose exec backend python -c "import torch; print(torch.cuda.is_available())"
```

## 6. 知识库索引

- 代码使用的索引目录：`backend/chroma_db/`（gitignore，不入库）。
- **仓库交付时索引已存在且与当前代码匹配**（2026-10-20 按章节切分重建，
  1041 chunks）；正常使用**不需要**重建。
- 仅当 `data/` 下的 PDF 有增删、或切分逻辑（`app/rag/chapter_split.py`）变更时，
  才在明确的重建窗口执行（会删除并重建整个索引）：

> ⚠️ **重建前必须停止正在运行、占用索引的服务**（uvicorn 等）。Windows 下
> chroma.sqlite3 被占用时重建会部分失败并留下空库（2026-10-20 实测踩过）。

```powershell
conda activate ai_agent
cd backend
python -m app.rag.ingest
```

## 6b. 健康计算工具（2026-10-21 新增）

确定性计算能力，由 `app/medical_agent/calculator.py` 的真实函数执行，**0 LLM、0 检索**：

- **BMI**：「我身高170，体重65公斤，BMI是多少？」→ 公式/代入/结果 + 中国成人参考分类（WS/T 428），明确"不等于身体健康程度"。
- **标准体重**：「我身高175，标准体重是多少？」→ 身高-105 粗略估算 + ±10% 参考区间。
- **单位换算**：「65kg等于多少斤」「170厘米等于多少米」。
- 缺参/参数异常（如身高300cm）→ 固定追问文案，不返回虚假结果。

路由规则：orchestrator 在 N03（急症判定）之后、N04（LLM 分类）之前做规则短路——
急症永远优先于计算；未命中计算意图的问题走原有 LLM 分类链路（零影响）；
组合问法（「BMI是多少？有什么危害」）先完成确定性计算，剩余解释交知识子 Agent 检索。
已知限制：不识别中文数字与复合单位（「一米七五」可识别，「五尺六寸」不可），解析不了就追问或走 LLM 链路。

## 7. 最小验证步骤

1. 按第 5 节启动前后端。
2. 在页面输入「高血压患者在生活方式上要注意什么？」→ 应得到带
   【参考来源】与免责声明的完整回答。
3. 输入「胸口压榨样疼痛，大汗淋漓，持续不缓解」→ 应返回立即拨打 120 的
   紧急提示（emergency_handled）。
4. 单测：`cd backend; python -m unittest discover -s tests -p "test_*.py"`（417 个）。
5. 真实 LLM 验收（需 API Key，约数分钟）：`cd backend; python -m acceptance`
   （结果写入 `backend/acceptance_results/`，不影响索引）。
6. 测试数据导航：固定数据集和共享 fixtures 见 [`testing/README.md`](testing/README.md)；
   覆盖矩阵与缺口见 [`testing/coverage_report.md`](testing/coverage_report.md)；
   黄金种子审核见 [`testing/golden_review.md`](testing/golden_review.md)；
   评估与优化资产见 [`Evaluation/README.md`](Evaluation/README.md)。

## 8. 常见启动错误

| 现象 | 原因与处理 |
|:---|:---|
| `未找到 DEEPSEEK_API_KEY` | 项目根或 backend/ 下缺 `.env`，按第 4 节创建 |
| `ModuleNotFoundError: langchain_chroma` | 不在 ai_agent 环境或依赖未装全，激活环境后 `pip install -r requirements.txt` |
| 知识问答全部「证据不足」 | 索引缺失或与代码不匹配：确认 `backend/chroma_db/` 存在；缺索引按第 6 节重建 |
| 首次请求很慢 | embedding 预热（启动时已 warm-up，首次真实检索仍可能数秒），属正常 |
| 前端编译报 `react-markdown` 缺失 | `cd frontend && npm install` |

## 9. 已知限制与未交付项

- **FR-01 引用支持率核验已完成**：AI 辅助逐句核验 104/104 = 100% ≥ 0.85（见
  [`docs/FR01人工核验记录.md`](docs/FR01人工核验记录.md)，建议人工抽查后作为正式结论）。
- 会话标识已持久化到浏览器 localStorage（页面刷新不丢上下文）；后端可选
  `SESSION_PERSIST_DIR` 文件持久化（服务重启不断上下文），见第 4 节。
- 知识库暂无儿童/老年人用药专门文档（默沙东大众版无独立页，相关内容由
  「影响药物反应的因素」条目覆盖）、~~低血糖词条~~（2026-10-20 已入库）。
- 中国医药信息查询平台类 PDF 无【】章节标记，走通用切分（国药准字说明书走章节切分）。
- 第 4 段生成后 grounding check 维持关闭（T-02b 已通过，无需开启）。
- 完整遗留清单见 `docs/开发进度.md` 最新条目。

## 10. 测试与验收基线（2026-10-20）

- 单元测试 123/123；真实 LLM 验收 21/21（安全关键用例 100%）。
- 阈值校准：[`docs/阈值校准报告.md`](docs/阈值校准报告.md)
  （结论：top1 距离不可分，第 1 段距离粗筛维持禁用）。
- 详见 `docs/开发进度.md` 第三阶段条目。

## 11. 评估优化可视化面板

本地离线面板：[`Evaluation/reports/dashboard.html`](Evaluation/reports/dashboard.html)（双击打开，无需服务）。

生成/刷新：

```powershell
cd backend
python -m acceptance.report --coverage ... --scenarios ...
python -m acceptance.trend
```

当前面板示例（2026-10-25 数据）：

| Run | 时间 | 总分 | 检索 | 答案质量 | 拒答健康 | 时延 | 成本 | 口径 |
|:---|:---|---:|---:|---:|---:|---:|---:|:---|
| scorecard_first | 2026-10-25 | 92.88 | 96.36 | 91.67 | 100.0 | 80 | N/A | 部分口径 |
| report_s4c | 2026-10-25T12:16:48 | 94.73 | 96.36 | 97.22 | 100.0 | 80 | N/A | 部分口径 |
| report_full | 2026-10-25T12:49:29 | 93.59 | 96.36 | 91.67 | 100.0 | 80 | 100.0 | 全量口径（基线） |

环比关键结论：

- `report_s4c` 较 `scorecard_first`：总分 ▲+1.85，答案质量 ▲+5.55，其余维度持平；
- `report_full` 较 `report_s4c`：总分 ▼−1.14，答案质量 ▼−5.55，成本首次补齐为 100.0。
- 当前各轮条件指纹未完全采集，面板提示分数差异**不能直接归因**到系统改动；
  后续跑 `acceptance.report` 时默认会采集条件指纹，可做可比性判定。

面板还包含：首屏总分、维度条（0–100 单轴）、历史趋势折线、条件差异卡片、日期×维度矩阵。
