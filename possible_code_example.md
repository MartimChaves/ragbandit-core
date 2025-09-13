Improved Example Structure:

```python
# 1. Basic setup with configuration
from ragbandit import RagBandit, Config
from ragbandit.documents import DocumentPipeline, MistralOCRDocument
from ragbandit.processors import ReferencesProcessor, FootnoteProcessor
from ragbandit.chunkers import SemanticChunker
from ragbandit.embedders import MistralEmbedder
from ragbandit.document_groups import DocumentGroup
from ragbandit.evaluation import SyntheticDatasetGenerator, RagEvaluator
from ragbandit.rag_systems import BasicRAG, HybridRAG

# Initialize with configuration
config = Config.from_env()  # Loads from .env file or environment variables
rb = RagBandit(config)

# 2. Process documents
documents = rb.process_documents(
    file_paths=["./data/doc1.pdf", "./data/doc2.pdf"],
    ocr_processor=MistralOCRDocument(),
    processors=[ReferencesProcessor(), FootnoteProcessor()],
    chunker=SemanticChunker(min_chunk_size=500),
    embedder=MistralEmbedder(model="mistral-embed")
)

# 3. Create document group with semantic connections
doc_group = DocumentGroup.from_documents(
    documents=documents,
    connection_strategy="semantic_similarity",
    threshold=0.85
)

# 4. Generate synthetic test dataset
test_dataset = SyntheticDatasetGenerator(
    strategy="cluster_based",
    num_questions=50
).generate(doc_group)

# 5. Evaluate RAG systems
rag_systems = [
    BasicRAG(embedder=MistralEmbedder()),
    HybridRAG(embedder=MistralEmbedder(), reranker="cohere")
]

evaluation_results = RagEvaluator().evaluate(
    rag_systems=rag_systems,
    test_dataset=test_dataset,
    metrics=["relevance", "faithfulness", "answer_correctness"]
)

# 6. View results
rb.visualize_results(evaluation_results)
```