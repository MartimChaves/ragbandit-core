# ragbandit-core

![Test Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)

Core utilities for:

* Document ingestion & processing (OCR, chunking, embedding)
* Building and running Retrieval-Augmented Generation (RAG) pipelines
* Evaluating answers with automated metrics

## Test Coverage

The codebase maintains **87% test coverage** with comprehensive integration tests covering all major components. See [tests/README.md](tests/README.md) for details on running tests and coverage reports.

## Quick start

```bash
pip install ragbandit-core
```

```python
from ragbandit.documents import (
    DocumentPipeline,
    ReferencesRefiner,
    FootnoteRefiner,
    MistralOCR,
    MistralEmbedder,
    SemanticChunker
)
import os
import logging
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

file_path = "./data/raw/[document_name].pdf"

doc_pipeline = DocumentPipeline(
    chunker=SemanticChunker(api_key=MISTRAL_API_KEY, min_chunk_size=500),
    embedder=MistralEmbedder(api_key=MISTRAL_API_KEY, model="mistral-embed"),
    ocr_processor=MistralOCR(api_key=MISTRAL_API_KEY),
    refiners=[
        ReferencesRefiner(api_key=MISTRAL_API_KEY),
        FootnoteRefiner(api_key=MISTRAL_API_KEY),
    ],
)

extended_response = doc_pipeline.process(file_path)
```

### Using Alternative OCR and Embedding Providers

The package supports multiple OCR and embedding providers:

```python
from ragbandit.documents import (
    DocumentPipeline,
    DatalabOCR,
    OpenAIEmbedder,
    FixedSizeChunker
)
import os

DATALAB_API_KEY = os.getenv("DATALAB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

file_path = "./data/raw/[document_name].pdf"

# Using Datalab OCR and OpenAI embeddings
doc_pipeline = DocumentPipeline(
    ocr_processor=DatalabOCR(
        api_key=DATALAB_API_KEY,
        model="marker",
        mode="balanced"  # Options: fast, balanced, accurate
    ),
    chunker=FixedSizeChunker(chunk_size=500, overlap=100),
    embedder=OpenAIEmbedder(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"  # or text-embedding-3-large
    ),
)

result = doc_pipeline.process(file_path)

```

### Running Steps Manually

For more control, you can run each pipeline step independently:

```python
from ragbandit.documents import (
    DocumentPipeline,
    ReferencesRefiner,
    MistralOCR,
    MistralEmbedder,
    SemanticChunker
)
import os
from dotenv import load_dotenv
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
file_path = "./data/raw/[document_name].pdf"

# Create pipeline with only the components you need
pipeline = DocumentPipeline(
    ocr_processor=MistralOCR(api_key=MISTRAL_API_KEY),
    refiners=[ReferencesRefiner(api_key=MISTRAL_API_KEY)],
    chunker=SemanticChunker(api_key=MISTRAL_API_KEY, min_chunk_size=500),
    embedder=MistralEmbedder(api_key=MISTRAL_API_KEY, model="mistral-embed"),
)

# Step 1: Run OCR
ocr_result = pipeline.run_ocr(file_path)

# Step 2: Run refiners (optional)
refining_results = pipeline.run_refiners(ocr_result)
final_doc = refining_results[-1]  # Get the last refiner's output

# Step 3: Chunk the document
chunk_result = pipeline.run_chunker(final_doc)

# Step 4: Embed chunks
embedding_result = pipeline.run_embedder(chunk_result)
```

You can also use components independently without a pipeline:

```python
# Run OCR directly - Mistral
ocr = MistralOCR(api_key=MISTRAL_API_KEY)
ocr_result = ocr.process(file_path)

# Or use Datalab OCR
from ragbandit.documents import DatalabOCR
datalab_ocr = DatalabOCR(
    api_key=DATALAB_API_KEY,
    mode="accurate",
    max_pages=10  # Optional: limit pages processed
)
ocr_result = datalab_ocr.process(file_path)

# Run refiners directly
refiner = FootnoteRefiner(api_key=MISTRAL_API_KEY)
refined_result = refiner.process(ocr_result)

# Run chunker directly
chunker = SemanticChunker(api_key=MISTRAL_API_KEY, min_chunk_size=500)
chunk_result = chunker.chunk(refined_result)

# Run embedder directly - Mistral
embedder = MistralEmbedder(api_key=MISTRAL_API_KEY)
embedding_result = embedder.embed_chunks(chunk_result)

# Or use OpenAI embeddings
from ragbandit.documents import OpenAIEmbedder
openai_embedder = OpenAIEmbedder(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-large"  # Higher quality, larger dimensions
)
embedding_result = openai_embedder.embed_chunks(chunk_result)
```

## Available Components

### OCR

| Class | Provider | Models | Key params |
|-------|----------|--------|------------|
| `MistralOCR` | Mistral | `mistral-ocr-2512` (default) | `api_key`, `model` |
| `DatalabOCR` | Datalab | `marker` | `api_key`, `mode` (`fast` / `balanced` / `accurate`), `max_pages`, `page_range` |

### Refiners

| Class | What it does |
|-------|-------------|
| `ReferencesRefiner` | Detects and extracts the references/bibliography section. Stores in `extracted_data["references_markdown"]`. |
| `FootnoteRefiner` | Detects footnotes, inlines explanations, and collects citations. |
| `TableOfContentsRefiner` | Detects and removes the table of contents. Stores in `extracted_data["toc_markdown"]`. |

### Chunkers

| Class | Params (defaults) | When to use |
|-------|-------------------|-------------|
| `FixedSizeChunker` | `chunk_size=1000`, `overlap=200` | Fast, deterministic splitting by character count |
| `SentenceChunker` | `sentences_per_chunk=5`, `sentence_overlap=1`, `min_chunk_size=100` | Sentence-aware sliding window, no external deps |
| `RecursiveMarkdownChunker` | `chunk_size=1000`, `overlap=100` | Heading-aware hierarchical splitting (H1→H2→H3→H4→paragraph→sentence) |
| `SemanticChunker` | `api_key`, `min_chunk_size=500` | LLM-based semantic boundary detection (uses Mistral) |

### Embedders

| Class | Provider | Models | Cost / 1M tokens |
|-------|----------|--------|-------------------|
| `MistralEmbedder` | Mistral | `mistral-embed` | $0.10 |
| `OpenAIEmbedder` | OpenAI | `text-embedding-3-small`, `text-embedding-3-large` | $0.02 / $0.13 |
| `VoyageAIEmbedder` | Voyage AI | `voyage-3.5` (default), `voyage-3.5-lite`, `voyage-3-large`, `voyage-3`, `voyage-3-lite` | $0.06 / $0.02 / $0.18 / $0.06 / $0.02 |
| `CohereEmbedder` | Cohere | `embed-v4.0` | $0.12 |

## Examples & Notebooks

### Example scripts (`examples/`)

| File | What it shows |
|------|---------------|
| [`01_basic_pipeline.py`](examples/01_basic_pipeline.py) | End-to-end `DocumentPipeline.process()` with MistralOCR + FixedSizeChunker + MistralEmbedder |
| [`02_choosing_components.py`](examples/02_choosing_components.py) | Same doc with two combos (Mistral-only vs mixed providers) — compares chunks, dims, cost |
| [`03_step_by_step.py`](examples/03_step_by_step.py) | Manual `run_ocr()` → `run_refiners()` → `run_chunker()` → `run_embedder()` with intermediate inspection |
| [`04_cost_tracking.py`](examples/04_cost_tracking.py) | `TokenUsageTracker` standalone — per-model breakdown and total cost |

```bash
.venv/bin/python examples/01_basic_pipeline.py
```

### Notebooks (`notebooks/`)

| File | What it shows |
|------|---------------|
| [`getting_started.ipynb`](notebooks/getting_started.ipynb) | Full pipeline walkthrough — one cell per stage |
| [`component_comparison.ipynb`](notebooks/component_comparison.ipynb) | Compares FixedSize vs Sentence vs RecursiveMarkdown chunking strategies |
| [`component_explorer.ipynb`](notebooks/component_explorer.ipynb) | Exercises every component with all valid configurations |

Each notebook has a setup cell where you set `PDF_PATH` and `ENV_PATH` to point to your own document and API keys.

## Package layout

```
ragbandit-core/
├── examples/          # Runnable example scripts
├── notebooks/         # Jupyter notebooks
├── src/ragbandit/
│   ├── config/        # Pricing and model configuration
│   ├── documents/     # Document ingestion, OCR, chunking, embedding
│   │   ├── chunkers/
│   │   ├── embedders/
│   │   ├── ocr/
│   │   └── refiners/
│   ├── prompt_tools/  # LLM-based tools
│   └── utils/         # Token tracking, logging, client managers
└── tests/
```

## License

MIT
