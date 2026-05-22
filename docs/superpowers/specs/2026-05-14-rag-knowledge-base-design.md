# RAG 知识库问答系统 — 设计规格

## 概述
基于 LangChain + Chroma + FastAPI 构建 RAG 知识库问答系统，支持 PDF/TXT 文档上传、语义检索、LLM 问答与来源引用。

## 技术栈
- **后端**: Python 3.10+, FastAPI, LangChain, Chroma, OpenAI API
- **前端**: 原生 HTML + Tailwind CSS（CDN）
- **向量库**: Chroma（本地持久化）

## 架构

```
Frontend (HTML + Tailwind CSS)
       ↕ HTTP REST API
FastAPI Backend
       ↕
core/loader.py → core/chunker.py → db/chroma_client.py
       ↕
rag/retriever.py → rag/qa_chain.py → OpenAI API
```

## 项目结构

```
rag-knowledge-base/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   ├── api/
│   │   ├── __init__.py
│   │   ├── documents.py
│   │   └── chat.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── processor.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   └── qa_chain.py
│   └── db/
│       ├── __init__.py
│       └── chroma_client.py
├── frontend/
│   ├── index.html
│   └── documents.html
└── uploads/
```

## 配置项

| 参数 | 默认值 | 说明 |
|---|---|---|
| CHUNK_SIZE | 500 | 切片字符数 |
| CHUNK_OVERLAP | 50 | 切片重叠字符数 |
| K_RETRIEVAL | 4 | 检索返回 Top-K 切片 |
| EMBEDDING_MODEL | text-embedding-3-small | OpenAI 嵌入模型 |
| LLM_MODEL | gpt-4o-mini | 大语言模型 |
| CHROMA_PERSIST_DIR | ./chroma_db | 向量库持久化路径 |
| TEMPERATURE | 0 | LLM 温度参数 |

## 接口设计

### 文档管理
- `POST /api/documents/upload` — 上传 PDF/TXT 文件(自动触发索引)
- `GET /api/documents` — 获取文档列表
- `DELETE /api/documents/{id}` — 删除文档(同时清除向量)
- `GET /api/documents/{id}/chunks` — 查看文档切片详情

### 问答
- `POST /api/chat` — 提交问题，返回回答+引用来源
- `POST /api/retrieval/test` — 纯检索测试(返回 Top-K chunks)

### Request/Response 示例

**POST /api/chat**
```json
// Request
{"question": "上海地区2024年增长率是多少？", "k": 4}
// Response
{
  "answer": "根据报告，上海地区2024年第一季度增长率为12.5%...",
  "sources": [
    {
      "filename": "2024市场报告.pdf",
      "content": "...上海Q1增长12.5%...",
      "chunk_index": 3,
      "score": 0.92
    }
  ]
}
```

## 核心数据流

### 文档索引
1. 上传文件保存至 `uploads/`
2. `loader.py` 提取文本(PDF 用 PyMuPDF, TXT 用内置读取)
3. `chunker.py` 用 `RecursiveCharacterTextSplitter` 切片
4. `processor.py` 编排：生成 Embedding → 存入 Chroma(metadata 记录 filename, chunk_index)
5. 文件元信息(名称、大小、切片数、状态)返回前端

### 问答
1. 接收用户问题
2. `retriever.py` 问题 Embedding → Chroma 相似度检索 Top-K
3. `qa_chain.py` 组装 Prompt(系统指令 + 检索上下文 + 问题)
4. 调用 OpenAI Chat Completion API
5. 返回回答文本 + 引用来源列表(每个 source 含原文片段、文件名、相关度分数)

## 前端页面

1. **问答页面** (index.html) — 三栏布局参考原型 #3: 左侧对话历史 | 中间聊天区(含引用折叠展示) | 右侧推荐/收藏
2. **文档管理页面** (documents.html) — 参考原型 #6: 上传拖拽区、文档列表表格、搜索筛选、状态标签

## 错误处理
- PDF 解析失败 → 返回 400 + 明确错误信息
- OpenAI API 调用失败 → 返回 503 + 提示检查 API Key
- 空知识库检索 → 告知用户"知识库暂无相关内容，请先上传文档"
- 文件类型校验 → 仅允许 .pdf/.txt，其他格式返回 400

## 非功能性要求
- Chroma 数据持久化至 `chroma_db/` 目录
- 上传文件存储至 `uploads/` 目录
- 支持 CORS (方便前后端分离开发)
- Uvicorn 运行，默认端口 8000
