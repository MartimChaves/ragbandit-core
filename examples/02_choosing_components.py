"""
02 — Choosing Components

Process the same document with two different component
combinations (Mistral-only vs mixed providers) and compare
chunk counts, embedding dimensions, and cost.
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

from ragbandit.documents import (
    DocumentPipeline,
    MistralOCR,
    DatalabOCR,
    FixedSizeChunker,
    SentenceChunker,
    MistralEmbedder,
    OpenAIEmbedder,
    ReferencesRefiner,
)

# ── Configuration ──────────────────────────────────────
PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "sample.pdf"
)
ENV_PATH = Path(__file__).resolve().parent.parent / "tests" / ".env"
# Change these paths to use your own document and API keys.
# ───────────────────────────────────────────────────────

load_dotenv(ENV_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DATALAB_API_KEY = os.getenv("DATALAB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pdf = str(PDF_PATH)

# ── Combo 1: All-Mistral ──────────────────────────────
pipeline_a = DocumentPipeline(
    ocr_processor=MistralOCR(api_key=MISTRAL_API_KEY),
    refiners=[ReferencesRefiner(api_key=MISTRAL_API_KEY)],
    chunker=FixedSizeChunker(chunk_size=1000, overlap=200),
    embedder=MistralEmbedder(
        api_key=MISTRAL_API_KEY, model="mistral-embed",
    ),
)
result_a = pipeline_a.process(pdf)

# ── Combo 2: Datalab OCR + SentenceChunker + OpenAI ───
pipeline_b = DocumentPipeline(
    ocr_processor=DatalabOCR(
        api_key=DATALAB_API_KEY, mode="fast",
    ),
    chunker=SentenceChunker(
        sentences_per_chunk=5, sentence_overlap=1,
    ),
    embedder=OpenAIEmbedder(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",
    ),
)
result_b = pipeline_b.process(pdf)


# ── Comparison ─────────────────────────────────────────
def summarise(label: str, r):
    chunks = (
        r.chunking_result.chunks
        if r.chunking_result else []
    )
    sizes = [len(c.text) for c in chunks]
    embs = (
        r.embedding_result.chunks_with_embeddings
        if r.embedding_result else []
    )
    dim = len(embs[0].embedding) if embs else 0
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    ocr_pages = (
        len(r.ocr_result.pages)
        if r.ocr_result else "n/a"
    )
    print(f"  OCR pages        : {ocr_pages}")
    print(f"  Chunks           : {len(chunks)}")
    if sizes:
        avg = sum(sizes) // len(sizes)
        print(
            f"  Chunk sizes      : "
            f"min={min(sizes)}, max={max(sizes)}, "
            f"avg={avg}"
        )
    print(f"  Embedding dim    : {dim}")
    print(f"  Total cost       : ${r.total_cost_usd:.6f}")
    print(f"  Total time       : {r.timings.total_duration:.2f}s")


print("\n" + "=" * 60)
print("COMPONENT COMPARISON")
print("=" * 60)

summarise(
    "Combo A — MistralOCR + FixedSizeChunker"
    " + MistralEmbedder",
    result_a,
)
summarise(
    "Combo B — DatalabOCR + SentenceChunker"
    " + OpenAIEmbedder",
    result_b,
)
