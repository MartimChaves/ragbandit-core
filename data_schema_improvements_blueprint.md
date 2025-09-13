# Data Schema Improvements Blueprint

This document outlines the final data schemas for the `ragbandit` package. The goal is to create a modular, extensible, and well-documented structure that supports running pipeline steps independently.

## 1. Core Principles

- **Modularity**: Each step of the document pipeline (OCR, Processing, Chunking, Embedding) has its own distinct input and output schema.
- **Explicitness**: Schemas are defined using Pydantic models to ensure type safety and clear contracts.
- **Statelessness**: The core package remains stateless. ID generation for database persistence is the responsibility of the calling application.

## 2. Final Schemas

Below are the Pydantic models that will be implemented in `src/ragbandit/schema.py`.

### 2.1. Metrics and Configuration Schemas

```python
class TokenUsageMetrics(BaseModel):
    """Metrics for token usage and cost."""
    model: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost: float | None = None
    prompt_cost: float | None = None
    completion_cost: float | None = None

class TimingMetrics(BaseModel):
    """Metrics for pipeline step durations in seconds."""
    total_duration: float
    ocr: float | None = None
    processing_steps: list[dict[str, float]] | None = None # e.g., [{"FootnoteProcessor": 0.5}, ...]
    chunking: float | None = None
    embedding: float | None = None
```

### 2.2. Core Data Schemas

```python
class PageDimensions(BaseModel):
    dpi: int
    height: int
    width: int

class Image(BaseModel):
    """Represents an image extracted from a page."""
    id: str  # e.g., 'img-01.jpg'
    top_left_x: int | None = None
    top_left_y: int | None = None
    bottom_right_x: int | None = None
    bottom_right_y: int | None = None
    image_base64: str
    image_annotation: str | None = None # JSON string

class BasePage(BaseModel):
    """Base schema for a single page of a document."""
    index: int  # Page number
    markdown: str
    images: list[Image] | None = None
    dimensions: PageDimensions

class OCRPage(BasePage):
    """Represents a single page from an OCR result."""
    pass

class ProcessedPage(BasePage):
    """Represents a single page after text processors have been applied."""
    pass

class ChunkMetadata(BaseModel):
    """Metadata associated with a chunk."""
    page_number: int
    source_references: list[str] | None = None
    footnotes: list[dict] | None = None
    images: list[Image] | None = None
    extra: dict[str, any] = {}

class Chunk(BaseModel):
    """Represents a chunk of text, ready for embedding."""
    text: str
    metadata: ChunkMetadata

class ChunkWithEmbedding(Chunk):
    """Represents a chunk that has been embedded."""
    embedding: list[float]
    embedding_model: str
```

### 2.3. Processor-Specific Schemas

```python
class ProcessingTraceItem(BaseModel):
    """Trace of a single processor's execution."""
    processor_name: str
    summary: str
    duration: float # Duration in seconds
```

### 2.4. Pipeline Step Result Schemas

```python
class OCRUsageInfo(BaseModel):
    pages_processed: int
    doc_size_bytes: int

class OCRResult(BaseModel):
    """Represents the output of the OCR process."""
    source_file_path: str
    processed_at: datetime
    model: str
    document_annotation: str | None = None
    pages: list[OCRPage]
    usage_info: OCRUsageInfo
    metrics: TokenUsageMetrics | None = None # If OCR uses an LLM

class ProcessingResult(BaseModel):
    """Represents the output of the text processors."""
    processed_at: datetime
    pages: list[ProcessedPage] # The text content, now structured per page
    processing_trace: list[ProcessingTraceItem]
    extracted_data: dict[str, any] # For footnotes, references, etc.
    metrics: list[TokenUsageMetrics] | None = None

class ChunkingResult(BaseModel):
    """Represents the output of the chunking process."""
    processed_at: datetime
    chunks: list[Chunk]
    metrics: TokenUsageMetrics | None = None # If chunker uses an LLM

class EmbeddingResult(BaseModel):
    """Represents the output of the embedding process."""
    processed_at: datetime
    chunks_with_embeddings: list[ChunkWithEmbedding]
    model_name: str
    metrics: TokenUsageMetrics
```

### 2.5. Final Composite Result Schema

```python
class DocumentPipelineResult(BaseModel):
    """The composite result for an end-to-end pipeline run."""
    source_file_path: str
    processed_at: datetime
    pipeline_config: dict
    timings: TimingMetrics
    total_metrics: list[TokenUsageMetrics]
    ocr_result: OCRResult
    processing_result: ProcessingResult
    chunking_result: ChunkingResult | None = None
    embedding_result: EmbeddingResult | None = None
