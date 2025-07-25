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
from ragbandit.processing import DocumentProcessor
from ragbandit.rag import RAGConfig, RAGPipeline

processor = DocumentProcessor()
chunks = processor.process("docs/*.pdf")

config = RAGConfig.default()
pipeline = RAGPipeline(config)

answer = pipeline.query(chunks, "What is the total revenue in 2023?")
print(answer.text)
```

## Package layout

```
ragbandit-core/
├── src/ragbandit/
│   ├── processing/   # document ingestion, OCR, chunking, embedding
│   ├── rag/          # configs & pipelines
│   └── evaluation/   # evaluation helpers using RAGas
└── tests/
```

## License

MIT
