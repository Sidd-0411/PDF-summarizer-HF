"""
Core summarization logic (Hugging Face version):
1. Extract text from a PDF
2. Chunk it to fit the model's input limit
3. Summarize each chunk locally using a Hugging Face transformers pipeline
4. Combine chunk summaries into a final summary + key points
"""

from pypdf import PdfReader
from transformers import pipeline

# Loaded once and reused across calls (loading the model is the slow part)
_summarizer = None


def get_summarizer(model_name: str = "facebook/bart-large-cnn"):
    """
    Lazily load and cache the Hugging Face summarization pipeline.
    First call downloads the model (~1.6GB for bart-large-cnn) and caches it locally.
    """
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model=model_name)
    return _summarizer


def extract_text_from_pdf(file) -> tuple[str, int]:
    """Extract raw text from a PDF file. Returns (text, num_pages)."""
    reader = PdfReader(file)
    num_pages = len(reader.pages)

    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(page_text)

    text = "\n\n".join(text_parts)

    if not text.strip():
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned/image-only PDF, which needs OCR (not covered here)."
        )

    return text, num_pages


def chunk_text(text: str, max_words: int = 800) -> list[str]:
    """
    Split text into word-based chunks.
    BART's max input is ~1024 tokens (~700-800 words), so we chunk conservatively.
    Splits on paragraph boundaries where possible to keep chunks coherent.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_words = []

    for para in paragraphs:
        para_words = para.split()
        if len(current_words) + len(para_words) <= max_words:
            current_words.extend(para_words)
        else:
            if current_words:
                chunks.append(" ".join(current_words))
            current_words = para_words

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks if chunks else [text]


def summarize_chunk(summarizer, chunk: str) -> str:
    """Summarize a single chunk of text using the HF pipeline."""
    word_count = len(chunk.split())
    # Keep summaries proportional to input length, within the model's comfortable range
    max_len = min(180, max(40, word_count // 2))
    min_len = min(30, max_len - 10)

    result = summarizer(
        chunk,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
    )
    return result[0]["summary_text"]


def extract_key_points(chunk_summaries: list[str], max_points: int = 8) -> list[str]:
    """
    Derive key points from chunk summaries by splitting into sentences.
    (Simple heuristic — no second model call needed, keeps things fast and local.)
    """
    import re

    all_text = " ".join(chunk_summaries)
    sentences = re.split(r"(?<=[.!?])\s+", all_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    return sentences[:max_points]


def summarize_pdf(file, model_name: str = "facebook/bart-large-cnn") -> dict:
    """
    End-to-end pipeline: PDF -> extracted text -> summary + key points.
    Runs entirely locally using a Hugging Face model.
    """
    text, num_pages = extract_text_from_pdf(file)
    chunks = chunk_text(text)

    summarizer = get_summarizer(model_name)
    chunk_summaries = [summarize_chunk(summarizer, chunk) for chunk in chunks]

    # If we had multiple chunks, do one more summarization pass over the combined
    # chunk summaries to produce a tighter overall summary.
    if len(chunk_summaries) > 1:
        combined = " ".join(chunk_summaries)
        final_summary = summarize_chunk(summarizer, combined)
    else:
        final_summary = chunk_summaries[0]

    key_points = extract_key_points(chunk_summaries)

    return {
        "summary": final_summary,
        "key_points": key_points,
        "num_pages": num_pages,
        "num_chunks": len(chunks),
    }