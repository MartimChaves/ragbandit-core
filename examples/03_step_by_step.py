"""
03 — Step-by-Step Pipeline

Run each pipeline stage manually
(OCR → Refiners → Chunker → Embedder) and inspect every
intermediate result.
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

from ragbandit.documents import (
    DocumentPipeline,
    MistralOCR,
    ReferencesRefiner,
    FootnoteRefiner,
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
pdf = str(PDF_PATH)

pipeline = DocumentPipeline(
    ocr_processor=MistralOCR(api_key=MISTRAL_API_KEY),
    refiners=[
        ReferencesRefiner(api_key=MISTRAL_API_KEY),
        FootnoteRefiner(api_key=MISTRAL_API_KEY),
    ],
    chunker=FixedSizeChunker(chunk_size=1000, overlap=200),
    embedder=MistralEmbedder(
        api_key=MISTRAL_API_KEY, model="mistral-embed",
    ),
)

# ── Step 1: OCR ───────────────────────────────────────
print("=" * 60)
print("STEP 1 — OCR")
print("=" * 60)

ocr_result = pipeline.run_ocr(pdf)

print(f"  Model  : {ocr_result.model}")
print(f"  Pages  : {len(ocr_result.pages)}")
for i, page in enumerate(ocr_result.pages):
    preview = page.markdown[:120].replace("\n", " ")
    print(f"  Page {i}: {preview}...")

# ── Step 2: Refiners ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — REFINERS")
print("=" * 60)

refining_results = pipeline.run_refiners(ocr_result)

for rr in refining_results:
    print(f"\n  Refiner : {rr.component_name}")
    print(f"  Pages   : {len(rr.pages)}")
    if rr.extracted_data:
        for key, val in rr.extracted_data.items():
            preview = str(val)[:100].replace("\n", " ")
            print(f"  {key}: {preview}")

final_doc = refining_results[-1]

# ── Step 3: Chunking ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — CHUNKING")
print("=" * 60)

chunk_result = pipeline.run_chunker(final_doc)

chunks = chunk_result.chunks
sizes = [len(c.text) for c in chunks]
print(f"  Chunks     : {len(chunks)}")
print(f"  Size range : {min(sizes)} – {max(sizes)} chars")

for i, c in enumerate(chunks[:3]):
    preview = c.text[:100].replace("\n", " ")
    print(f"  Chunk {i}: [{len(c.text)} chars] {preview}...")

if len(chunks) > 3:
    print(f"  ... and {len(chunks) - 3} more chunks")

# ── Step 4: Embedding ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — EMBEDDING")
print("=" * 60)

embedding_result = pipeline.run_embedder(chunk_result)

embs = embedding_result.chunks_with_embeddings
dim = len(embs[0].embedding) if embs else 0
print(f"  Vectors    : {len(embs)}")
print(f"  Dimensions : {dim}")
print(f"  Model      : {embedding_result.model_name}")

if embs:
    vec = embs[0].embedding
    print(f"  First vector (first 5 dims): {vec[:5]}")
