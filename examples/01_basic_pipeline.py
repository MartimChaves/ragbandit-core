"""
01 — Basic Pipeline

Run a full DocumentPipeline end-to-end with MistralOCR,
FixedSizeChunker, and MistralEmbedder.  Prints a summary
of every pipeline stage.
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

from ragbandit.documents import (
    DocumentPipeline,
    MistralOCR,
    FixedSizeChunker,
    MistralEmbedder,
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

# Build the pipeline
pipeline = DocumentPipeline(
    ocr_processor=MistralOCR(api_key=MISTRAL_API_KEY),
    chunker=FixedSizeChunker(chunk_size=1000, overlap=200),
    embedder=MistralEmbedder(
        api_key=MISTRAL_API_KEY, model="mistral-embed",
    ),
)

# Process the document
result = pipeline.process(str(PDF_PATH))

# ── Print summary ──────────────────────────────────────
print("\n" + "=" * 60)
print("PIPELINE RESULT SUMMARY")
print("=" * 60)

print(f"\nSource file : {result.source_file_path}")
print(f"Processed at: {result.processed_at}")

# OCR
if result.ocr_result:
    print("\n── OCR ──")
    print(f"  Pages       : {len(result.ocr_result.pages)}")
    print(f"  Model       : {result.ocr_result.model}")

# Chunking
if result.chunking_result:
    chunks = result.chunking_result.chunks
    sizes = [len(c.text) for c in chunks]
    print("\n── Chunking ──")
    print(f"  Chunks      : {len(chunks)}")
    print(f"  Size range  : {min(sizes)} – {max(sizes)} chars")

# Embedding
if result.embedding_result:
    embs = result.embedding_result.chunks_with_embeddings
    dim = len(embs[0].embedding) if embs else 0
    print("\n── Embedding ──")
    print(f"  Vectors     : {len(embs)}")
    print(f"  Dimensions  : {dim}")

# Timings & cost
print("\n── Timings ──")
t = result.timings
print(f"  OCR         : {t.ocr:.2f}s")
print(f"  Chunking    : {t.chunking:.2f}s")
print(f"  Embedding   : {t.embedding:.2f}s")
print(f"  Total       : {t.total_duration:.2f}s")

print("\n── Cost ──")
print(f"  Total       : ${result.total_cost_usd:.6f}")

print("\n── Step report ──")
print(f"  OCR       : {result.step_report.ocr}")
print(f"  Refining  : {result.step_report.refining}")
print(f"  Chunking  : {result.step_report.chunking}")
print(f"  Embedding : {result.step_report.embedding}")
