# 项目说明

本项目为大模型应用开发项目，包含 **后端（Python / FastAPI）** 与 **前端（Node.js）** 两部分。

---

## 一、环境要求

| 环境 | 版本要求 |
|:---|:---|
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |
| Docker（可选） | 24+ |

---

## 二、项目结构

```text
project_root/
├── backend/                 # 后端目录
│   ├── app/
│   │   ├── rag/             # RAG 相关模块
│   │   │   ├── chain.py
│   │   │   ├── ingest.py
│   │   │   ├── loader.py
│   │   │   ├── store.py
│   │   │   └── ...
│   │   ├── agent/           # Agent 相关模块
│   │   │   ├── tools.py
│   │   │   ├── agent_loop.py
│   │   │   └── ...
│   │   ├── chroma_db/       # 向量数据库
│   │   ├── data/            # 数据目录
│   │   ├── uploads/         # 上传文件目录
│   │   ├── main.py          # 后端入口
│   │   └── requirements.txt # Python 依赖
│   └── Dockerfile
├── frontend/                # 前端目录
│   ├── node_modules/
│   ├── package.json
│   └── package-lock.json

进入后端目录
bash
cd backend
2. 创建并激活虚拟环境（推荐）
bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. 安装依赖
bash
pip install -r requirements.txt
4. 配置环境变量
在 backend/ 目录下创建 .env 文件，参考内容：

env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
5. 启动后端服务
bash
python -m app.main
或使用 uvicorn（如果是 FastAPI）：

bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
默认后端地址：http://127.0.0.1:8000

四、前端启动方式
1. 进入前端目录
bash
cd frontend
2. 安装依赖
bash
npm install
3. 启动开发服务器
bash
npm run dev
默认前端地址：http://localhost:5173（Vite）或 http://localhost:3000（Next.js / CRA，具体以控制台输出为准）

4. 打包生产版本
bash
npm run build
五、Docker 启动方式（可选）
在项目根目录执行：

bash
docker-compose up -d
停止服务：

bash
docker-compose down
六、常用命令速查
操作	命令
启动后端	uvicorn app.main:app --reload
启动前端	npm run dev
安装后端依赖	pip install -r requirements.txt
安装前端依赖	npm install
Docker 启动	docker-compose up -d
Docker 停止	docker-compose down
七、注意事项
后端启动前请确保 .env 文件已配置好 API Key。

前端启动前请确认后端服务已运行，否则接口会报错。

向量数据库 chroma_db 首次使用需先执行 ingest.py 导入数据。

如果端口被占用，可在启动命令中修改 --port 参数。

八、RAG 模块说明
backend/app/rag/ 下各文件职责：

文件	作用
loader.py	加载文档（PDF / TXT / Word 等）
ingest.py	文档切分 + 向量化 + 写入向量库
store.py	向量库连接与检索封装
chain.py	RAG 主流程（检索 → 拼接 → LLM 生成）
debug_chroma.py	调试向量库
show_chunks.py	查看切分后的 chunk
testVec.py	向量检索测试
ragTest.py	RAG 整体测试
yanzheng.py	验证脚本
RAG 首次使用流程
bash
cd backend
python -m app.rag.ingest      # 导入文档到向量库
python -m app.rag.ragTest     # 测试 RAG 是否跑通
九、Agent 模块说明
backend/app/agent/ 下各文件职责：

文件	作用
tools.py	工具定义 + Tool Schema + TOOL_MAP
agent_loop.py	Agent Loop（while True + tool_calls 判断）
get_weather.py	天气工具示例
workflow.py	Agent Workflow（LangGraph）
Agent 测试方式
bash
cd backend
python -m app.agent.tools     # 测试 Tool Calling
python -m app.agent.agent_loop # 测试 Agent Loop
Agent 核心流程
text
用户提问
  ↓
LLM 决策（是否调工具 / 调哪个）
  ↓
执行工具 → 结果回传
  ↓
LLM 生成最终回答
  ↓
tool_calls = None → 结束
十、评估模块说明
metric/ 与 数据评估/ 用于评估 RAG / Agent 效果：

文件 / 目录	作用
metric/eval.py	评估脚本（准确率 / 召回率 / 响应质量等）
数据评估/	评估数据集与结果

├── metric/                  # 评估指标
│   └── eval.py
├── 数据评估/                 # 数据评估相关
├── docker-compose.yml
└── README.md

---

## 十一、提交规范

提交信息采用 `类型: 简述` 的格式，类型取值如下：

| 类型 | 含义 |
|:---|:---|
| `feat` | 新增功能 |
| `fix` | 修复缺陷 |
| `docs` | 仅文档改动 |
| `refactor` | 重构（不改变外部行为） |
| `chore` | 构建 / 依赖 / 配置等杂项 |

示例：

```bash
git commit -m "feat: 新增 Agent 工具调用重试逻辑"
git commit -m "fix: 修复 RAG 检索返回空结果的问题"
```

注意事项：

- `backend/data/` 与 `backend/chroma_db/` 下包含较大的二进制文件，改动前请确认是否确实需要入库。
- `.env` 已在 `.gitignore` 中忽略，请勿将真实 API Key 写入代码或提交到仓库。