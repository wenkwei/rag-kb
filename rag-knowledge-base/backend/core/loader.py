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
    """Load text from a supported document type (PDF or TXT).

    Raises ValueError for unsupported file types.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix == ".txt":
        return load_txt(path)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}，仅支持 PDF 和 TXT")
