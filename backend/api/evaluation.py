import json
import re
import sys
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import BASE_DIR, K_RETRIEVAL, LLM_MODEL, TEMPERATURE, OPENAI_API_KEY, OPENAI_BASE_URL
from rag.retriever import retrieve


def _parse_score(raw: str) -> float:
    """Extract first float number from LLM judge response."""
    match = re.search(r'(\d+\.?\d*)', raw.strip())
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return 0.0

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

# ── File paths ──
TEST_SET_FILE = BASE_DIR / "evaluation_test_set.json"
RESULTS_FILE = BASE_DIR / "evaluation_results.json"


# ── Pydantic models ──
class TestQuestion(BaseModel):
    id: str = ""
    question: str
    expected: str


class TestSet(BaseModel):
    questions: List[TestQuestion]


class EvalConfig(BaseModel):
    k: int = Field(default=K_RETRIEVAL, ge=1, le=20)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    questions: List[TestQuestion]


class EvaluateResponse(BaseModel):
    id: str
    config: dict
    timestamp: str
    metrics: dict
    results: list


# ── Helpers ──
def _load_json(path, default=None):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default if default is not None else []


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _compute_text_overlap(retrieved_text: str, expected_text: str) -> float:
    """Compute word-level overlap ratio (Jaccard-like) between two texts."""
    if not expected_text or not retrieved_text:
        return 0.0
    # Simple Chinese-aware tokenization by character bigrams
    def tokenize(s):
        s = s.lower().strip()
        return {s[i:i+2] for i in range(len(s)-1)} | set(s.split())
    r_tokens = tokenize(retrieved_text)
    e_tokens = tokenize(expected_text)
    if not e_tokens:
        return 0.0
    intersection = r_tokens & e_tokens
    return round(len(intersection) / len(e_tokens), 4)


# ── LLM 生成 & 评判 ──

def _get_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL,
    )


def _generate_answer(question: str, context: str) -> str:
    """Generate an answer using the configured LLM."""
    if not context.strip():
        return "（无相关文档，无法回答）"
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的知识库问答助手。请基于以下提供的参考内容回答用户的问题。

【要求】
1. 如果参考内容中有相关信息，请基于参考内容给出准确、详细的回答
2. 如果参考内容中没有足够信息，请如实告知用户，不要编造

【参考内容】
{context}"""),
        ("human", "{question}"),
    ])
    chain = prompt | _get_llm()
    try:
        return chain.invoke({"context": context, "question": question}).content
    except Exception as e:
        return f"（生成回答失败: {str(e)}）"


def _judge_faithfulness(question: str, context: str, answer: str) -> float:
    """Judge how well the answer is grounded in the retrieved context (0.0 - 1.0)."""
    if not context.strip() or "无相关文档" in answer or "生成回答失败" in answer:
        return 0.0
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个严格的评测助手，评估回答是否忠实于给定的参考内容。

评分规则（只返回一个数字 0.0-1.0，不要任何其他文字）：
1.0 = 回答中的所有信息都能在参考内容中找到依据，完全没有编造
0.7 = 大部分信息有依据，但有少量推断或表述偏差
0.4 = 部分信息有依据，但存在明显编造或与参考内容矛盾
0.0 = 回答基本没有依据参考内容，大量编造"""),
        ("human", """【参考内容】
{context}

【问题】
{question}

【回答】
{answer}

请给出 faithfulness 评分（0.0-1.0）："""),
    ])
    chain = prompt | _get_llm()
    try:
        raw = chain.invoke({"context": context, "question": question, "answer": answer}).content
        return _parse_score(raw)
    except Exception:
        return 0.0


def _judge_correctness(expected: str, answer: str) -> float:
    """Judge how well the answer matches the expected answer (0.0 - 1.0)."""
    if not expected.strip() or "生成回答失败" in answer:
        return 0.0
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个严格的评测助手，评估回答与预期答案的匹配程度。

评分规则（只返回一个数字 0.0-1.0，不要任何其他文字）：
1.0 = 回答与预期答案完全一致，包含所有关键信息
0.7 = 回答包含大部分关键信息，但缺少一些细节或表述有差异
0.4 = 回答只包含了部分关键信息，或者有较多偏差
0.0 = 回答与预期答案完全不相关或完全错误"""),
        ("human", """【预期答案】
{expected}

【实际回答】
{answer}

请给出回答正确性评分（0.0-1.0）："""),
    ])
    chain = prompt | _get_llm()
    try:
        raw = chain.invoke({"expected": expected, "answer": answer}).content
        return _parse_score(raw)
    except Exception:
        return 0.0


def add_question_to_test_set(question: str, expected: str) -> dict:
    """Add a single question to the test set. Returns the added question dict."""
    questions = _load_json(TEST_SET_FILE, [])
    # Avoid duplicates by question text
    existing = [q for q in questions if q["question"] == question]
    if existing:
        return existing[0]
    new_id = f"q{len(questions) + 1}-{uuid.uuid4().hex[:4]}"
    new_q = {"id": new_id, "question": question, "expected": expected}
    questions.append(new_q)
    _save_json(TEST_SET_FILE, questions)
    return new_q


# ── Routes ──

# 1. Test set management
@router.get("/test-set")
async def get_test_set():
    """Get the saved test set."""
    questions = _load_json(TEST_SET_FILE, [])
    return {"questions": questions}


@router.post("/test-set")
async def save_test_set(data: TestSet):
    """Save/replace the entire test set."""
    questions = [q.model_dump() for q in data.questions]
    _save_json(TEST_SET_FILE, questions)
    return {"status": "success", "count": len(questions)}


# 2. Run evaluation
@router.post("/evaluate")
async def run_evaluation(config: EvalConfig):
    """Run evaluation: retrieve for each question, compute metrics."""
    questions = config.questions
    if not questions:
        raise HTTPException(status_code=400, detail="测试问题列表为空")
    k = config.k
    threshold = config.threshold

    eval_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()

    per_question_results = []

    for q in questions:
        try:
            docs_with_scores = retrieve(q.question, k=k)

            # Filter by threshold
            if threshold > 0:
                docs_with_scores = [(d, s) for d, s in docs_with_scores if s >= threshold]

            hit = len(docs_with_scores) > 0

            # Best score
            best_score = max((s for _, s in docs_with_scores), default=0.0)

            # Recall: compute text overlap between retrieved content and expected answer
            all_content = " ".join([d.page_content for d, _ in docs_with_scores])
            recall = _compute_text_overlap(all_content, q.expected)

            # Build context string for LLM
            context_str = "\n\n---\n\n".join([d.page_content for d, _ in docs_with_scores]) if docs_with_scores else ""

            # ── LLM generation & evaluation ──
            llm_answer = _generate_answer(q.question, context_str)
            faithfulness = _judge_faithfulness(q.question, context_str, llm_answer)
            correctness = _judge_correctness(q.expected, llm_answer) if q.expected else 0.0

            # Build source detail list
            sources = [
                {
                    "filename": d.metadata.get("filename", "未知文档"),
                    "content": d.page_content[:300],
                    "chunk_index": d.metadata.get("chunk_index", 0),
                    "score": round(float(s), 4),
                }
                for d, s in docs_with_scores
            ]

            per_question_results.append({
                "id": q.id,
                "question": q.question,
                "expected": q.expected,
                "llm_answer": llm_answer,
                "hit": hit,
                "recall": recall,
                "faithfulness": faithfulness,
                "correctness": correctness,
                "best_score": round(best_score, 4),
                "sources": sources,
            })
        except Exception as e:
            per_question_results.append({
                "id": q.id,
                "question": q.question,
                "expected": q.expected,
                "llm_answer": "（评测异常）",
                "hit": False,
                "recall": 0.0,
                "faithfulness": 0.0,
                "correctness": 0.0,
                "best_score": 0.0,
                "sources": [],
                "error": str(e),
            })

    # ── Aggregate metrics ──
    total = len(per_question_results)
    hits = sum(1 for r in per_question_results if r["hit"])
    hit_rate = round(hits / total, 4) if total else 0
    avg_recall = round(sum(r["recall"] for r in per_question_results) / total, 4) if total else 0
    avg_faithfulness = round(sum(r["faithfulness"] for r in per_question_results) / total, 4) if total else 0
    avg_correctness = round(sum(r["correctness"] for r in per_question_results) / total, 4) if total else 0

    # MRR: Mean Reciprocal Rank — 1/rank of first relevant result
    mrr_values = []
    for r in per_question_results:
        if r["best_score"] > 0:
            # Use best_score as a proxy: treat score > threshold as "relevant"
            mrr_values.append(r["best_score"])
        else:
            mrr_values.append(0.0)
    mrr = round(sum(mrr_values) / total, 4) if total else 0

    # NDCG: based on best_score distribution across top k
    ndcg_values = []
    for r in per_question_results:
        scores = [s["score"] for s in r["sources"]]
        if scores:
            # DCG: sum(score / log2(rank+1))
            dcg = sum(score / (i + 1) for i, score in enumerate(scores))
            # IDCG: ideal ordering (sorted descending)
            ideal = sorted(scores, reverse=True)
            idcg = sum(score / (i + 1) for i, score in enumerate(ideal))
            ndcg_values.append(dcg / idcg if idcg > 0 else 0)
        else:
            ndcg_values.append(0)
    ndcg = round(sum(ndcg_values) / total, 4) if total else 0

    metrics = {
        "total": total,
        "hit_rate": hit_rate,
        "avg_recall": avg_recall,
        "avg_faithfulness": avg_faithfulness,
        "avg_correctness": avg_correctness,
        "mrr": mrr,
        "ndcg": ndcg,
        "config": {"k": k, "threshold": threshold},
    }

    # Save to history
    eval_record = {
        "id": eval_id,
        "timestamp": timestamp,
        "metrics": metrics,
        "config": {"k": k, "threshold": threshold},
        "results": per_question_results,
    }
    history = _load_json(RESULTS_FILE, [])
    history.insert(0, eval_record)
    # Keep last 50 runs
    _save_json(RESULTS_FILE, history[:50])

    return {
        "id": eval_id,
        "config": {"k": k, "threshold": threshold},
        "timestamp": timestamp,
        "metrics": metrics,
        "results": per_question_results,
    }


# 3. Evaluation history
@router.get("/history")
async def list_history():
    """List all evaluation history summaries."""
    history = _load_json(RESULTS_FILE, [])
    return [
        {
            "id": h["id"],
            "timestamp": h["timestamp"],
            "metrics": h["metrics"],
            "config": h["config"],
            "question_count": len(h["results"]),
        }
        for h in history
    ]


@router.get("/history/{eval_id}")
async def get_history(eval_id: str):
    """Get a specific evaluation result by ID."""
    history = _load_json(RESULTS_FILE, [])
    for h in history:
        if h["id"] == eval_id:
            return h
    raise HTTPException(status_code=404, detail="评测记录未找到")
