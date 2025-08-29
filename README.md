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
from ragbandit.documents import (
    DocumentPipeline,
    ReferencesProcessor,
    FootnoteProcessor,
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
    chunker=SemanticChunker(min_chunk_size=500, api_key=MISTRAL_API_KEY),
    embedder=MistralEmbedder(model="mistral-embed", api_key=MISTRAL_API_KEY),  # noqa
    ocr_processor=MistralOCRDocument(api_key=MISTRAL_API_KEY),
    processors=[
        ReferencesProcessor(api_key=MISTRAL_API_KEY),
        FootnoteProcessor(api_key=MISTRAL_API_KEY),
    ],
)

extended_response = doc_pipeline.process(file_path, "1")

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
