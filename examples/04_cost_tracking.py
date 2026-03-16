"""
04 — Cost Tracking

Use TokenUsageTracker standalone to monitor token usage
and costs across multiple operations.  Prints per-model
breakdown and totals.
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
from ragbandit.utils.token_usage_tracker import (
    TokenUsageTracker,
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

# Create a shared tracker
tracker = TokenUsageTracker()

# ── Run pipeline steps, passing the tracker ────────────
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

# Full pipeline run (tracks costs internally)
result = pipeline.process(pdf)

# ── Print cost breakdown from the pipeline result ─────
print("\n" + "=" * 60)
print("COST TRACKING REPORT")
print("=" * 60)

# Per-step metrics from the pipeline result
print("\n── Per-step metrics ──")
if result.total_metrics:
    for metric in result.total_metrics:
        print(f"  {metric}")

# Total cost
print("\n── Total cost ──")
print(f"  ${result.total_cost_usd:.6f}")

# ── Standalone tracker demo ────────────────────────────
# You can also use TokenUsageTracker independently to
# track costs across multiple separate operations.
print("\n" + "=" * 60)
print("STANDALONE TRACKER DEMO")
print("=" * 60)

standalone = TokenUsageTracker()

# Simulate tracking some LLM calls
standalone.add_usage(
    input_tokens=1500, output_tokens=300,
    model="mistral-small-latest",
)
standalone.add_usage(
    input_tokens=2000, output_tokens=500,
    model="mistral-small-latest",
)
standalone.add_usage(
    input_tokens=800, output_tokens=200,
    model="mistral-large-latest",
)

# Simulate tracking some embedding calls
standalone.add_embedding_tokens(
    tokens=50000, model="mistral-embed",
)
standalone.add_embedding_tokens(
    tokens=30000, model="text-embedding-3-small",
)

summary = standalone.get_summary()

print(f"\n  Total calls           : {summary.total_calls}")
print(f"  Total input tokens    : {summary.total_input_tokens}")
print(f"  Total output tokens   : {summary.total_output_tokens}")
print(f"  Total embedding tokens: {summary.total_embedding_tokens}")
print(f"  Total cost            : ${summary.total_cost_usd:.6f}")

print("\n── Per-model breakdown ──")
for model_name, usage in summary.models.items():
    print(f"  {model_name}:")
    print(f"    Calls           : {usage.calls}")
    print(f"    Input tokens    : {usage.input_tokens}")
    print(f"    Output tokens   : {usage.output_tokens}")
    print(f"    Embedding tokens: {usage.embedding_tokens}")
    print(f"    Cost            : ${usage.cost:.6f}")

# Also log to the standard logger
standalone.log_summary()
