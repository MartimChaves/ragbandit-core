# ragbandit-core

Core utilities for:

* Document ingestion & processing (OCR, chunking, embedding)
* Building and running Retrieval-Augmented Generation (RAG) pipelines
* Evaluating answers with automated metrics

## Quick start

```bash
pip install ragbandit-core
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

extended_response = doc_pipeline.process(file_path)

```

### Running Steps Manually

For more control, you can run each pipeline step independently:

```python
from ragbandit.documents import (
    DocumentPipeline,
    ReferencesProcessor,
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
    processors=[ReferencesProcessor(api_key=MISTRAL_API_KEY)],
    chunker=SemanticChunker(min_chunk_size=500, api_key=MISTRAL_API_KEY),
    embedder=MistralEmbedder(model="mistral-embed", api_key=MISTRAL_API_KEY),
)

# Step 1: Run OCR
ocr_result = pipeline.run_ocr(file_path)

# Step 2: Run processors (optional)
processing_results = pipeline.run_processors(ocr_result)
final_doc = processing_results[-1]  # Get the last processor's output

# Step 3: Chunk the document
chunk_result = pipeline.run_chunker(final_doc)

# Step 4: Embed chunks
embedding_result = pipeline.run_embedder(chunk_result)
```

You can also create separate pipelines for different steps:

```python
# OCR-only pipeline
ocr_pipeline = DocumentPipeline(
    ocr_processor=MistralOCRDocument(api_key=MISTRAL_API_KEY)
)
ocr_result = ocr_pipeline.run_ocr(file_path)

# Later, chunk with a different pipeline
chunk_pipeline = DocumentPipeline(
    chunker=SemanticChunker(min_chunk_size=500, api_key=MISTRAL_API_KEY)
)
chunks = chunk_pipeline.run_chunker(ocr_result)
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
