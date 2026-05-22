# RAG 知识库问答系统 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个完整的本地 RAG 知识库问答系统，支持 PDF/TXT 上传、切片索引、语义检索、LLM 问答带引用来源

**Architecture:** FastAPI 后端提供 REST API，LangChain 编排 RAG 管线，Chroma 向量库持久化，前端原生 HTML+Tailwind 呈现。上传文档 → 文本提取 → 切片 → Embedding → Chroma 入库；用户提问 → 检索 Top-K → Prompt 组装 → LLM 回答 → 返回带来源引用

**Tech Stack:** Python 3.10+, FastAPI, LangChain, Chroma, OpenAI API (GPT-4o-mini + text-embedding-3-small), HTML + Tailwind CSS

**Base path:** `c:\Users\wenkwei\Desktop\面试项目\1、RAG\RAG高保真原型\rag-knowledge-base`

---

### Task 1: 项目脚手架与配置

**Files:**
- Create: `rag-knowledge-base/requirements.txt`
- Create: `rag-knowledge-base/backend/__init__.py`
- Create: `rag-knowledge-base/backend/config.py`
- Create: `rag-knowledge-base/backend/api/__init__.py`
- Create: `rag-knowledge-base/backend/core/__init__.py`
- Create: `rag-knowledge-base/backend/rag/__init__.py`
- Create: `rag-knowledge-base/backend/db/__init__.py`
- Create: `rag-knowledge-base/uploads/.gitkeep`
- Create: `rag-knowledge-base/chroma_db/.gitkeep`

- [ ] **Step 1: Create requirements.txt**

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
langchain==0.3.0
langchain-community==0.3.0
langchain-openai==0.2.0
langchain-chroma==0.1.0
chromadb==0.5.0
pypdf==5.0.0
python-multipart==0.0.12
pydantic==2.9.0
pydantic-settings==2.5.0
```

- [ ] **Step 2: Create config.py**

```python
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
K_RETRIEVAL = int(os.getenv("K_RETRIEVAL", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
```

- [ ] **Step 3: Create all \_\_init\_\_.py files** (all empty files)

```python
# Empty __init__.py
```

Files to create:
- `backend/__init__.py`
- `backend/api/__init__.py`
- `backend/core/__init__.py`
- `backend/rag/__init__.py`
- `backend/db/__init__.py`

- [ ] **Step 4: Create .gitkeep files**

```bash
mkdir -p uploads chroma_db
touch uploads/.gitkeep chroma_db/.gitkeep
```

- [ ] **Step 5: Verify structure**

```bash
cd rag-knowledge-base
find . -type f | sort
```

Expected output:
```
./backend/__init__.py
./backend/api/__init__.py
./backend/config.py
./backend/core/__init__.py
./backend/db/__init__.py
./backend/rag/__init__.py
./chroma_db/.gitkeep
./requirements.txt
./uploads/.gitkeep
```

---

### Task 2: Chroma 向量库客户端

**Files:**
- Create: `backend/db/chroma_client.py`

- [ ] **Step 1: Create chroma_client.py**

```python
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
)

vector_store = Chroma(
    collection_name="rag_knowledge_base",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PERSIST_DIR),
)

# Native Chroma client for filtered operations
_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
_collection = _client.get_or_create_collection("rag_knowledge_base")


def get_vector_store() -> Chroma:
    return vector_store


def get_collection():
    return _collection


def get_embedding_function():
    return embeddings
```

---

### Task 3: 文档加载器 (PDF/TXT 文本提取)

**Files:**
- Create: `backend/core/loader.py`

- [ ] **Step 1: Create loader.py**

```python
from pathlib import Path
from typing import Union

from pypdf import PdfReader


def load_pdf(file_path: Union[str, Path]) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(str(file_path))
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def load_txt(file_path: Union[str, Path]) -> str:
    """Read text from a TXT file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_document(file_path: Union[str, Path]) -> str:
    """Load text from a supported document type (PDF or TXT)."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix == ".txt":
        return load_txt(path)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}，仅支持 PDF 和 TXT")
```

---

### Task 4: 文本切片器

**Files:**
- Create: `backend/core/chunker.py`

- [ ] **Step 1: Create chunker.py**

```python
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    length_function=len,
)


def split_text(text: str) -> List[str]:
    """Split text into chunks using recursive character splitting."""
    if not text.strip():
        return []
    return _text_splitter.split_text(text)
```

---

### Task 5: 文档处理管线

**Files:**
- Create: `backend/core/processor.py`

- [ ] **Step 1: Create processor.py**

```python
from pathlib import Path
from typing import Union

from langchain_core.documents import Document

from core.loader import load_document
from core.chunker import split_text
from db.chroma_client import get_vector_store, get_collection


def process_document(file_path: Union[str, Path], filename: str) -> int:
    """Process a document: extract text, chunk, embed, and store in Chroma.

    Args:
        file_path: Path to the uploaded file.
        filename: Original filename for metadata tracking.

    Returns:
        Number of chunks created.
    """
    raw_text = load_document(file_path)
    chunks = split_text(raw_text)

    if not chunks:
        return 0

    # Remove existing chunks for this filename (supports re-indexing)
    delete_chunks_by_filename(filename)

    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "filename": filename,
                "chunk_index": i,
                "source": str(file_path),
            },
        )
        documents.append(doc)

    vector_store = get_vector_store()
    vector_store.add_documents(documents)

    return len(chunks)


def delete_chunks_by_filename(filename: str) -> int:
    """Delete all vector chunks associated with a filename.

    Returns:
        Number of chunks removed.
    """
    collection = get_collection()
    existing = collection.get(where={"filename": filename})
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def get_all_documents() -> list[dict]:
    """Get list of all documents (unique filenames) with chunk counts."""
    collection = get_collection()
    all_data = collection.get()
    filenames = set()

    for meta in all_data.get("metadatas", []):
        if meta and "filename" in meta:
            filenames.add(meta["filename"])

    result = []
    for fn in sorted(filenames):
        docs = collection.get(where={"filename": fn})
        result.append({
            "filename": fn,
            "chunk_count": len(docs.get("ids", [])),
        })
    return result
```

---

### Task 6: 文档管理 API

**Files:**
- Create: `backend/api/documents.py`

- [ ] **Step 1: Create documents.py**

```python
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from config import UPLOAD_DIR
from core.processor import process_document, delete_chunks_by_filename, get_all_documents

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF/TXT document, automatically process and index it."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 PDF 和 TXT 文件")

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        chunk_count = process_document(str(file_path), file.filename)
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

    return {
        "filename": file.filename,
        "size": len(content),
        "chunk_count": chunk_count,
        "status": "success",
    }


@router.get("")
async def list_documents():
    """List all indexed documents with chunk counts."""
    documents = get_all_documents()
    return {"documents": documents, "total": len(documents)}


@router.delete("/{filename:path}")
async def delete_document(filename: str):
    """Delete a document and all its vector chunks."""
    file_path = UPLOAD_DIR / filename

    chunks_removed = delete_chunks_by_filename(filename)

    if file_path.exists():
        file_path.unlink()

    if chunks_removed == 0 and not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文档 '{filename}' 不存在")

    return {
        "deleted": True,
        "filename": filename,
        "chunks_removed": chunks_removed,
    }
```

---

### Task 7: 语义检索器

**Files:**
- Create: `backend/rag/retriever.py`

- [ ] **Step 1: Create retriever.py**

```python
from typing import List, Tuple

from langchain_core.documents import Document

from db.chroma_client import get_vector_store


def retrieve(query: str, k: int = 4) -> List[Tuple[Document, float]]:
    """Semantic search: retrieve top-k relevant document chunks for a query.

    Returns:
        List of (Document, relevance_score) tuples. Higher score = more relevant.
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    return results
```

---

### Task 8: 问答链

**Files:**
- Create: `backend/rag/qa_chain.py`

- [ ] **Step 1: Create qa_chain.py**

```python
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import LLM_MODEL, TEMPERATURE, K_RETRIEVAL, OPENAI_API_KEY, OPENAI_BASE_URL
from rag.retriever import retrieve

_SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请基于以下提供的参考内容回答用户的问题。

【要求】
1. 如果参考内容中有相关信息，请基于参考内容给出准确、详细的回答
2. 如果参考内容中没有足够信息，请如实告知用户，不要编造
3. 回答时尽量引用参考内容中的具体表述

【参考内容】
{context}"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", "{question}"),
])


def _get_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
        openai_base_url=OPENAI_BASE_URL,
    )


def qa_with_sources(question: str, k: int = None) -> dict:
    """Answer a question using RAG: retrieve relevant chunks, then ask LLM.

    Returns:
        dict with keys:
            - answer: str, the LLM's answer
            - sources: list of {filename, content, chunk_index, score}
    """
    llm = _get_llm()
    top_k = k or K_RETRIEVAL

    docs_with_scores = retrieve(question, k=top_k)

    context_parts = []
    sources = []

    for doc, score in docs_with_scores:
        context_parts.append(doc.page_content)
        sources.append({
            "filename": doc.metadata.get("filename", "未知文档"),
            "content": doc.page_content,
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "score": round(float(score), 4),
        })

    if not context_parts:
        return {
            "answer": "知识库中暂无相关内容，请先上传相关文档后再提问。",
            "sources": [],
        }

    context = "\n\n---\n\n".join(context_parts)
    chain = _prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "sources": sources,
    }
```

---

### Task 9: 聊天 API

**Files:**
- Create: `backend/api/chat.py`

- [ ] **Step 1: Create chat.py**

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import K_RETRIEVAL
from rag.qa_chain import qa_with_sources

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    k: int = Field(default=K_RETRIEVAL, ge=1, le=20, description="检索返回的切片数量")


class ChatResponse(BaseModel):
    answer: str
    sources: list


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question and get an answer with cited sources."""
    try:
        result = qa_with_sources(request.question, k=request.k)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")
```

---

### Task 10: FastAPI 主入口

**Files:**
- Create: `backend/main.py`

- [ ] **Step 1: Create main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.documents import router as documents_router
from api.chat import router as chat_router

app = FastAPI(title="RAG 知识库问答系统", version="1.0.0")

# CORS — allow frontend dev server / file access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "rag-knowledge-base"}
```

---

### Task 11: 前端 — 知识库问答页面

**Files:**
- Create: `frontend/index.html`

- [ ] **Step 1: Create index.html**

Complete HTML file based on prototype #3 with real API integration:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>企业知识库 - 智能助手</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body {
            font-family: 'Inter', 'Microsoft YaHei', sans-serif;
            background-color: #f8fafc;
        }
        .custom-scrollbar::-webkit-scrollbar { width: 5px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 10px; }
        .chat-bubble-system { border: 1px solid #dbeafe; background-color: #ffffff; }
        .active-chat { background-color: #eff6ff; border-left: 4px solid #2563eb; }
        .glass-nav { backdrop-filter: blur(8px); background-color: rgba(255, 255, 255, 0.9); }
    </style>
</head>
<body class="h-screen flex flex-col overflow-hidden">
    <!-- 顶部导航栏 -->
    <header class="h-16 glass-nav border-b border-slate-200 px-6 flex items-center justify-between shrink-0 z-20 shadow-sm">
        <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-200">
                <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path></svg>
            </div>
            <h1 class="text-xl font-bold text-slate-800 tracking-tight">企业知识库</h1>
        </div>
        <div class="flex items-center gap-4">
            <a href="/documents.html" class="text-slate-500 hover:text-blue-600 transition-colors flex items-center gap-1.5 font-medium text-sm">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                知识库管理
            </a>
        </div>
    </header>

    <div class="flex-1 flex overflow-hidden">
        <!-- 左侧边栏：历史对话 -->
        <aside class="w-[240px] bg-white border-r border-slate-200 flex flex-col shrink-0">
            <div class="flex-1 overflow-y-auto custom-scrollbar pt-4 px-3">
                <p class="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-3 px-3">对话历史</p>
                <div id="conversation-list" class="space-y-1">
                    <div class="active-chat p-3 rounded-xl cursor-pointer shadow-sm transition-all" data-id="default">
                        <p class="text-sm font-bold text-blue-700 truncate">当前对话</p>
                        <p class="text-[10px] text-blue-400 mt-1.5 flex items-center gap-1">
                            <span class="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></span>
                            进行中
                        </p>
                    </div>
                </div>
            </div>
            <div class="p-4 border-t border-slate-100">
                <button onclick="newConversation()" class="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-bold flex items-center justify-center gap-2 shadow-lg shadow-blue-100 transition-all active:scale-[0.98]">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
                    新建对话
                </button>
            </div>
        </aside>

        <!-- 主聊天区 -->
        <main class="flex-1 flex flex-col bg-slate-50/30">
            <div id="chat-messages" class="flex-1 overflow-y-auto px-8 py-6 space-y-6 custom-scrollbar">
                <!-- 系统欢迎消息 -->
                <div class="flex items-start gap-4">
                    <div class="w-9 h-9 bg-blue-600 rounded-lg flex items-center justify-center shrink-0 shadow-md">
                        <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>
                    </div>
                    <div class="max-w-[85%]">
                        <div class="chat-bubble-system p-4 rounded-2xl rounded-tl-none shadow-sm">
                            <p class="text-sm text-slate-700 leading-relaxed">
                                您好！我是企业智能助手。我已经学习了知识库中的所有文档，您可以向我提问任何相关问题。
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 输入区 -->
            <div class="p-6 bg-white border-t border-slate-200 shrink-0">
                <div class="max-w-4xl mx-auto">
                    <div class="flex items-end bg-slate-100/80 rounded-2xl border border-slate-200 focus-within:ring-4 focus-within:ring-blue-500/10 focus-within:bg-white focus-within:border-blue-300 transition-all overflow-hidden p-1">
                        <textarea id="question-input" rows="1" placeholder="输入您的问题，如：文档中提到了哪些重要数据？" class="flex-1 bg-transparent px-2 py-3.5 text-sm outline-none resize-none placeholder:text-slate-400 min-h-[52px] max-h-40"></textarea>
                        <button id="send-btn" onclick="sendQuestion()" class="m-1.5 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold rounded-xl shadow-lg shadow-blue-100 transition-all flex items-center gap-2 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed">
                            发送
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>
                        </button>
                    </div>
                    <p class="text-[10px] text-slate-400 text-center mt-3 tracking-wide">智能回答仅供参考，核心数据请以正式文档为准。</p>
                </div>
            </div>
        </main>

        <!-- 右侧边栏 -->
        <aside class="w-[260px] bg-white border-l border-slate-200 flex flex-col shrink-0">
            <div class="p-6 space-y-8 overflow-y-auto custom-scrollbar">
                <!-- 知识库统计 -->
                <section>
                    <div class="flex items-center gap-2 mb-4">
                        <div class="w-1 h-4 bg-blue-600 rounded-full"></div>
                        <h3 class="text-xs font-bold text-slate-800 uppercase tracking-widest">知识库概览</h3>
                    </div>
                    <div id="kb-stats" class="space-y-3">
                        <div class="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                            <span class="text-xs text-slate-500">文档总数</span>
                            <span class="text-sm font-bold text-slate-700" id="doc-count">-</span>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                            <span class="text-xs text-slate-500">切片总数</span>
                            <span class="text-sm font-bold text-slate-700" id="chunk-count">-</span>
                        </div>
                    </div>
                </section>
            </div>
        </aside>
    </div>

    <script>
        const API_BASE = 'http://localhost:8000';

        // Load KB stats on page load
        document.addEventListener('DOMContentLoaded', loadStats);

        async function loadStats() {
            try {
                const res = await fetch(`${API_BASE}/api/documents`);
                const data = await res.json();
                const docs = data.documents || [];
                document.getElementById('doc-count').textContent = docs.length;
                const totalChunks = docs.reduce((s, d) => s + (d.chunk_count || 0), 0);
                document.getElementById('chunk-count').textContent = totalChunks;
            } catch (e) {
                // Backend might not be running
            }
        }

        function addMessage(type, content, sources) {
            const container = document.getElementById('chat-messages');
            const isUser = type === 'user';
            const div = document.createElement('div');
            div.className = `flex items-start gap-4 ${isUser ? 'justify-end' : ''}`;

            if (!isUser) {
                div.innerHTML = `
                    <div class="w-9 h-9 bg-blue-600 rounded-lg flex items-center justify-center shrink-0 shadow-md">
                        <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>
                    </div>
                    <div class="max-w-[85%]">
                        <div class="chat-bubble-system p-4 rounded-2xl rounded-tl-none shadow-sm">
                            <p class="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">${content}</p>
                        </div>
                        ${sources && sources.length > 0 ? `
                        <div class="mt-2">
                            <button onclick="toggleSources(this)" class="text-xs font-bold text-blue-600 hover:text-blue-700 flex items-center gap-1">
                                查看引用来源 (${sources.length})
                                <svg class="w-3 h-3 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                            </button>
                            <div class="hidden mt-2 space-y-2">
                                ${sources.map(s => `
                                <div class="p-3 bg-blue-50/80 rounded-xl border-l-4 border-blue-400">
                                    <div class="flex items-center gap-2 mb-1">
                                        <svg class="w-3 h-3 text-blue-600" fill="currentColor" viewBox="0 0 20 20"><path d="M9 2a2 2 0 00-2 2v8a2 2 0 002 2h6a2 2 0 002-2V6.414A2 2 0 0016.414 5L14 2.586A2 2 0 0012.586 2H9z"></path><path d="M3 8a2 2 0 012-2v10h8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"></path></svg>
                                        <span class="text-[10px] font-bold text-blue-800">${s.filename}</span>
                                        <span class="text-[10px] text-blue-400 ml-auto">相关度: ${(s.score * 100).toFixed(0)}%</span>
                                    </div>
                                    <p class="text-xs text-slate-600 italic">"...${s.content.substring(0, 150)}..."</p>
                                </div>
                                `).join('')}
                            </div>
                        </div>
                        ` : ''}
                    </div>
                `;
            } else {
                div.innerHTML = `
                    <div class="max-w-[80%]">
                        <div class="bg-blue-100/80 p-4 rounded-2xl rounded-tr-none text-left">
                            <p class="text-sm text-slate-800 whitespace-pre-wrap">${content}</p>
                        </div>
                    </div>
                    <div class="w-9 h-9 bg-slate-300 rounded-lg overflow-hidden shrink-0 shadow-sm flex items-center justify-center">
                        <svg class="w-5 h-5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
                    </div>
                `;
            }

            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        function addTypingIndicator() {
            const container = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.id = 'typing-indicator';
            div.className = 'flex items-start gap-4';
            div.innerHTML = `
                <div class="w-9 h-9 bg-blue-600 rounded-lg flex items-center justify-center shrink-0 shadow-md">
                    <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>
                </div>
                <div class="chat-bubble-system p-4 rounded-2xl rounded-tl-none shadow-sm">
                    <div class="flex gap-1">
                        <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay:0s"></span>
                        <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay:0.15s"></span>
                        <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay:0.3s"></span>
                    </div>
                </div>
            `;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        function removeTypingIndicator() {
            const el = document.getElementById('typing-indicator');
            if (el) el.remove();
        }

        function toggleSources(btn) {
            const sourcesDiv = btn.nextElementSibling;
            const isHidden = sourcesDiv.classList.contains('hidden');
            sourcesDiv.classList.toggle('hidden');
            btn.querySelector('svg').style.transform = isHidden ? 'rotate(180deg)' : '';
        }

        async function sendQuestion() {
            const input = document.getElementById('question-input');
            const question = input.value.trim();
            if (!question) return;

            const sendBtn = document.getElementById('send-btn');
            sendBtn.disabled = true;

            addMessage('user', question);
            input.value = '';
            input.style.height = 'auto';

            addTypingIndicator();

            try {
                const res = await fetch(`${API_BASE}/api/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, k: 4 }),
                });

                removeTypingIndicator();

                if (!res.ok) {
                    const err = await res.json();
                    addMessage('system', `请求失败: ${err.detail || '未知错误'}`);
                    return;
                }

                const data = await res.json();
                addMessage('system', data.answer, data.sources || []);
            } catch (e) {
                removeTypingIndicator();
                addMessage('system', '无法连接到后端服务，请确认服务器已启动 (localhost:8000)');
            } finally {
                sendBtn.disabled = false;
            }
        }

        function newConversation() {
            const container = document.getElementById('chat-messages');
            // Keep only the welcome message
            while (container.children.length > 1) {
                container.removeChild(container.lastChild);
            }
        }

        // Enter to send, Shift+Enter for newline
        document.getElementById('question-input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuestion();
            }
        });

        // Auto-resize textarea
        document.getElementById('question-input').addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 160) + 'px';
        });
    </script>
</body>
</html>
```

---

### Task 12: 前端 — 文档管理页面

**Files:**
- Create: `frontend/documents.html`

- [ ] **Step 1: Create documents.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>知识库管理系统 - 文档管理</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body { font-family: 'Inter', 'Microsoft YaHei', sans-serif; background-color: #f8fafc; }
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: #f1f5f9; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
        .upload-dragover { border-color: #3b82f6 !important; background-color: #eff6ff !important; }
    </style>
</head>
<body class="min-h-screen flex flex-col">
    <!-- 顶部导航 -->
    <header class="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between sticky top-0 z-30">
        <div class="flex items-center gap-4">
            <nav class="flex text-sm text-slate-500" aria-label="Breadcrumb">
                <ol class="inline-flex items-center space-x-1 md:space-x-3">
                    <li class="inline-flex items-center">
                        <a href="/index.html" class="hover:text-slate-800 flex items-center gap-1">
                            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z"></path></svg>
                            问答首页
                        </a>
                    </li>
                    <li aria-current="page">
                        <div class="flex items-center">
                            <svg class="w-6 h-6 text-slate-300" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd"></path></svg>
                            <span class="ml-1 font-semibold text-slate-800">知识库管理</span>
                        </div>
                    </li>
                </ol>
            </nav>
        </div>
        <div class="flex items-center gap-2 text-sm text-slate-400">
            <span class="flex items-center gap-1">
                <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
                系统运行正常
            </span>
        </div>
    </header>

    <main class="flex-1 flex overflow-hidden">
        <div class="flex-1 flex flex-col min-w-0 bg-white">
            <!-- 上传区域 -->
            <div class="px-6 pt-4 pb-2">
                <div id="upload-zone" class="border-2 border-dashed border-slate-300 rounded-xl p-8 flex flex-col items-center justify-center bg-white hover:border-blue-400 hover:bg-blue-50/30 transition-all cursor-pointer group">
                    <svg class="w-10 h-10 text-slate-400 group-hover:text-blue-500 mb-2 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                    <p class="text-sm font-medium text-slate-600">点击或将文件拖拽至此处上传</p>
                    <p class="text-xs text-slate-400 mt-1">支持 PDF, TXT (单个文件最大 50MB)</p>
                    <input id="file-input" type="file" accept=".pdf,.txt" class="hidden" multiple>
                </div>
                <!-- 上传进度提示 -->
                <div id="upload-status" class="hidden mt-3"></div>
            </div>

            <!-- 文档列表标题 -->
            <div class="px-6 py-3 border-b border-slate-100 flex items-center justify-between">
                <h2 class="text-base font-bold text-slate-800">文档列表</h2>
                <span id="doc-count-badge" class="text-xs text-slate-400 bg-slate-100 px-2 py-1 rounded-full">共 0 个文档</span>
            </div>

            <!-- 文档列表 -->
            <div class="flex-1 overflow-auto custom-scrollbar">
                <table class="w-full text-left border-collapse">
                    <thead class="bg-slate-50 sticky top-0 shadow-[0_1px_0_rgba(0,0,0,0.05)]">
                        <tr>
                            <th class="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">文档名称</th>
                            <th class="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">类型</th>
                            <th class="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">切片数</th>
                            <th class="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">状态</th>
                            <th class="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">操作</th>
                        </tr>
                    </thead>
                    <tbody id="doc-list-body" class="divide-y divide-slate-100">
                        <tr id="empty-row">
                            <td colspan="5" class="px-6 py-12 text-center text-sm text-slate-400">
                                暂无文档，请上传 PDF 或 TXT 文件
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </main>

    <!-- 删除确认弹窗 -->
    <div id="delete-modal" class="hidden fixed inset-0 bg-black/30 flex items-center justify-center z-50">
        <div class="bg-white rounded-2xl p-6 w-96 shadow-xl">
            <h3 class="text-lg font-bold text-slate-800 mb-2">确认删除</h3>
            <p class="text-sm text-slate-600 mb-6">确定要删除 <span id="delete-filename" class="font-bold text-slate-800"></span> 吗？此操作不可撤销。</p>
            <div class="flex justify-end gap-3">
                <button onclick="closeDeleteModal()" class="px-4 py-2 text-sm font-bold text-slate-500 hover:bg-slate-100 rounded-lg transition-all">取消</button>
                <button id="confirm-delete-btn" class="px-4 py-2 text-sm font-bold text-white bg-red-500 hover:bg-red-600 rounded-lg transition-all">删除</button>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = 'http://localhost:8000';
        let deleteTarget = null;

        // Upload zone handlers
        const uploadZone = document.getElementById('upload-zone');
        const fileInput = document.getElementById('file-input');

        uploadZone.addEventListener('click', () => fileInput.click());

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('upload-dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('upload-dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('upload-dragover');
            const files = e.dataTransfer.files;
            if (files.length) uploadFiles(files);
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) uploadFiles(fileInput.files);
        });

        async function uploadFiles(files) {
            const statusDiv = document.getElementById('upload-status');
            statusDiv.classList.remove('hidden');

            for (const file of files) {
                const ext = file.name.split('.').pop().toLowerCase();
                if (!['pdf', 'txt'].includes(ext)) {
                    statusDiv.innerHTML += `<div class="text-sm text-red-500">❌ ${file.name}: 不支持的文件格式</div>`;
                    continue;
                }

                statusDiv.innerHTML += `<div class="text-sm text-slate-600" id="status-${file.name}">⏳ ${file.name}: 上传处理中...</div>`;

                try {
                    const formData = new FormData();
                    formData.append('file', file);

                    const res = await fetch(`${API_BASE}/api/documents/upload`, {
                        method: 'POST',
                        body: formData,
                    });

                    const data = await res.json();
                    const statusEl = document.getElementById(`status-${file.name}`);

                    if (res.ok) {
                        statusEl.className = 'text-sm text-emerald-600';
                        statusEl.textContent = `✅ ${file.name}: 处理完成 (${data.chunk_count} 个切片)`;
                    } else {
                        statusEl.className = 'text-sm text-red-500';
                        statusEl.textContent = `❌ ${file.name}: ${data.detail || '上传失败'}`;
                    }
                } catch (e) {
                    const statusEl = document.getElementById(`status-${file.name}`);
                    if (statusEl) {
                        statusEl.className = 'text-sm text-red-500';
                        statusEl.textContent = `❌ ${file.name}: 网络错误`;
                    }
                }
            }

            // Refresh document list
            setTimeout(loadDocuments, 1000);
        }

        async function loadDocuments() {
            try {
                const res = await fetch(`${API_BASE}/api/documents`);
                const data = await res.json();
                const docs = data.documents || [];
                renderDocumentList(docs);
            } catch (e) {
                // Backend may not be running
            }
        }

        function renderDocumentList(docs) {
            const tbody = document.getElementById('doc-list-body');
            const badge = document.getElementById('doc-count-badge');
            badge.textContent = `共 ${docs.length} 个文档`;

            if (docs.length === 0) {
                tbody.innerHTML = `<tr id="empty-row"><td colspan="5" class="px-6 py-12 text-center text-sm text-slate-400">暂无文档，请上传 PDF 或 TXT 文件</td></tr>`;
                return;
            }

            tbody.innerHTML = docs.map(doc => {
                const ext = doc.filename.split('.').pop().toLowerCase();
                const isPdf = ext === 'pdf';
                const iconColor = isPdf ? 'text-red-500 bg-red-50' : 'text-blue-500 bg-blue-50';
                const typeLabel = isPdf ? 'PDF' : 'TXT';

                return `
                    <tr class="hover:bg-slate-50 transition-colors group">
                        <td class="px-6 py-4">
                            <div class="flex items-center gap-3">
                                <div class="p-2 ${iconColor} rounded-lg">
                                    <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z"></path></svg>
                                </div>
                                <span class="text-sm font-semibold text-slate-700 group-hover:text-blue-600">${doc.filename}</span>
                            </div>
                        </td>
                        <td class="px-6 py-4">
                            <span class="px-2 py-0.5 text-xs font-bold ${isPdf ? 'text-red-600 bg-red-50' : 'text-blue-600 bg-blue-50'} rounded">${typeLabel}</span>
                        </td>
                        <td class="px-6 py-4 text-sm text-slate-500">${doc.chunk_count} 个切片</td>
                        <td class="px-6 py-4">
                            <span class="px-2 py-1 rounded-full bg-emerald-50 text-emerald-600 text-[11px] font-bold border border-emerald-100">已索引</span>
                        </td>
                        <td class="px-6 py-4">
                            <button onclick="confirmDelete('${doc.filename}')" class="p-1.5 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-md transition-all" title="删除">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                            </button>
                        </td>
                    </tr>
                `;
            }).join('');
        }

        function confirmDelete(filename) {
            deleteTarget = filename;
            document.getElementById('delete-filename').textContent = filename;
            document.getElementById('delete-modal').classList.remove('hidden');
        }

        function closeDeleteModal() {
            deleteTarget = null;
            document.getElementById('delete-modal').classList.add('hidden');
        }

        document.getElementById('confirm-delete-btn').addEventListener('click', async () => {
            if (!deleteTarget) return;
            try {
                const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(deleteTarget)}`, {
                    method: 'DELETE',
                });
                if (res.ok) {
                    closeDeleteModal();
                    loadDocuments();
                } else {
                    const err = await res.json();
                    alert(`删除失败: ${err.detail}`);
                    closeDeleteModal();
                }
            } catch (e) {
                alert('删除失败: 网络错误');
                closeDeleteModal();
            }
        });

        // Load documents on page load
        document.addEventListener('DOMContentLoaded', loadDocuments);
    </script>
</body>
</html>
```

---

### Task 13: 安装依赖并验证启动

- [ ] **Step 1: Create and activate virtual environment, install deps**

```bash
cd rag-knowledge-base
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

- [ ] **Step 2: Start the server**

```bash
cd backend
export OPENAI_API_KEY="sk-your-key-here"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Expected: Uvicorn running on http://0.0.0.0:8000

- [ ] **Step 3: Test health endpoint**

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok","service":"rag-knowledge-base"}`

- [ ] **Step 4: Open frontend in browser**

Open `frontend/index.html` and `frontend/documents.html` in a browser (served via `python -m http.server` or directly with file://).
