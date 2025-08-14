# ragbandit-core

Core utilities for:

* Document ingestion & processing (OCR, chunking, embedding)
* Building and running Retrieval-Augmented Generation (RAG) pipelines
* Evaluating answers with automated metrics

## Quick start

```bash
pip install -e .[web]
```

```python
from dotenv import load_dotenv
load_dotenv()
from ragbandit.documents import (  # noqa
    DocumentPipeline,
    ReferencesProcessor,
    FootnoteProcessor,
    MistralOCRDocument,
    MistralEmbedder,
    SemanticChunker
)
from ragbandit.utils import TokenUsageTracker  # noqa
import os  # noqa

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

file_path = "./data/raw/[document_name].pdf"

mistral_ocr = MistralOCRDocument()

doc_pipeline = DocumentPipeline(
    ocr_processor=mistral_ocr,
    processors=[ReferencesProcessor(), FootnoteProcessor()]
)

ocr_response = doc_pipeline.perform_ocr(file_path)

extended_ocr_response = doc_pipeline.process(
    ocr_response=ocr_response,
    document_id="1"
)

# Create a semantic chunker
usage_tracker = TokenUsageTracker()
semantic_chunker = SemanticChunker(min_chunk_size=500)
semantic_chunks = semantic_chunker.chunk(
    extended_ocr_response, usage_tracker
)
# Process chunks to merge small ones
semantic_chunks = semantic_chunker.process_chunks(semantic_chunks)

# Generate embeddings for the chunks using MistralEmbedder
mistral_embedder = MistralEmbedder(
    model="mistral-embed", api_key=MISTRAL_API_KEY
)
semantic_chunks_with_embeddings = mistral_embedder.embed_chunks(
    semantic_chunks, usage_tracker
)
# Extend the response with embedding metadata
extended_ocr_response = mistral_embedder.extend_response(
    extended_ocr_response, semantic_chunks_with_embeddings
)

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
