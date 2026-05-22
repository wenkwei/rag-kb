# RAG 知识库问答系统 - 配置指南

## 1. 环境要求

- Python 3.11+
- OpenAI API Key

## 2. 配置 OpenAI API

### 方式一：命令行设置（推荐）

```bash
# Windows CMD
set OPENAI_API_KEY=sk-your-key-here
set OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，使用代理时需要修改

# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"

# macOS / Linux
export OPENAI_API_KEY=sk-your-key-here
export OPENAI_BASE_URL=https://api.openai.com/v1
```

### 方式二：在 Python 代码中直接修改

编辑 `backend/config.py`，找到以下行并填入你的 Key：

```python
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-key-here")  # 改这里
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")                 # 改这里
```

## 3. 启动后端服务

```bash
# 进入项目目录
cd rag-knowledge-base

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 启动服务（端口 8002）
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8002
```

启动后访问 http://localhost:8002/docs 可查看所有 API 接口文档。

## 4. 启动前端页面

```bash
# 进入前端目录
cd rag-knowledge-base/frontend

# 启动静态文件服务（端口 3000）
python -m http.server 3000
```

启动后访问 http://localhost:3000/admin-login.html

默认登录账号：`admin` / `admin123`

## 5. 修改 RAG 参数

通过管理后台「系统配置」页面可以调节以下参数，保存后写入 `rag_config.json`，重启后端生效：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| CHUNK_SIZE | 文本分块大小（token 数） | 500 |
| CHUNK_OVERLAP | 分块重叠大小 | 50 |
| TOP_K | 检索返回的切片数量 | 4 |
| TEMPERATURE | LLM 生成温度（0-2） | 0 |

如需永久修改默认值，编辑 `backend/config.py` 中的环境变量读取部分。

## 6. API 接口总览

| 接口 | 说明 | 需要认证 |
|------|------|---------|
| `POST /api/auth/login` | 登录获取 Token | 否 |
| `GET/POST /api/config` | 读取/保存系统配置 | 是 |
| `GET/POST/PUT/DELETE /api/users` | 用户管理 | 是 |
| `POST /api/documents/upload` | 上传文档 | 是 |
| `GET/DELETE /api/documents` | 文档列表/删除 | 是 |
| `POST /api/chat` | 问答检索 | 是 |
| `GET/POST/PUT/DELETE /api/badcases` | Bad Case 管理 | 是 |
| `GET/POST/DELETE /api/conversations` | 对话历史 | 是 |
| `GET /health` | 健康检查 | 否 |
