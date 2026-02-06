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
    MistralOCRDocument,
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
    ocr_processor=MistralOCRDocument(api_key=MISTRAL_API_KEY),
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
    MistralOCRDocument,
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
    ocr_processor=MistralOCRDocument(api_key=MISTRAL_API_KEY),
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
ocr = MistralOCRDocument(api_key=MISTRAL_API_KEY)
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

## Package layout

```
ragbandit-core/
├── src/ragbandit/
│   ├── documents/   # document ingestion, OCR, chunking, 
└── tests/
```

## License

MIT
