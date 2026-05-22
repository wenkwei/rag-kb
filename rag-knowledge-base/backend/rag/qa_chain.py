from langchain_openai import ChatOpenAI
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
        openai_api_base=OPENAI_BASE_URL,
    )


def qa_with_sources(question: str, k: int = None, threshold: float = 0.0) -> dict:
    """Answer a question using RAG: retrieve relevant chunks, then ask LLM.

    Args:
        question: The user's question.
        k: Number of chunks to retrieve. Defaults to config.K_RETRIEVAL.

    Returns:
        dict with keys:
            - answer: str, the LLM-generated answer
            - sources: list of {filename, content, chunk_index, score}
    """
    llm = _get_llm()
    top_k = k or K_RETRIEVAL

    docs_with_scores = retrieve(question, k=top_k)

    # Filter by relevance threshold
    if threshold > 0:
        docs_with_scores = [(doc, score) for doc, score in docs_with_scores if score >= threshold]

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
