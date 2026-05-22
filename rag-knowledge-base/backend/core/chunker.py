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
    """Split text into chunks using recursive character splitting.

    Returns an empty list if the input text is empty or only whitespace.
    """
    if not text.strip():
        return []
    return _text_splitter.split_text(text)
