"""Base schema for data structures."""
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

##########################################
# ************* V2 Schema ************** #
##########################################

##########################################
#                Metrics                 #
##########################################


class TokenUsageMetrics(BaseModel):
    """Aggregated token/cost usage metrics returned by TokenUsageTracker."""
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_embedding_tokens: int
    total_tokens: int
    total_cost_usd: float

    class ModelUsage(BaseModel):
        calls: int
        input_tokens: int | None = 0
        output_tokens: int | None = 0
        embedding_tokens: int | None = 0
        cost: float

    models: dict[str, ModelUsage]


class PagesProcessedMetrics(BaseModel):
    """Metrics for pages processed and associated cost."""
    pages_processed: int
    cost_per_page: float
    total_cost_usd: float


class TimingMetrics(BaseModel):
    """Metrics for pipeline step durations in seconds."""
    total_duration: float | None = None
    ocr: float | None = None
    # processing_steps: list[dict[str, float]] | None = None
    processing: float | None = None
    chunking: float | None = None
    embedding: float | None = None

##########################################
#                  OCR                   #
##########################################


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
    image_annotation: str | None = None  # JSON string


class BasePage(BaseModel):
    """Base schema for a single page of a document."""
    index: int  # Page number
    markdown: str
    images: list[Image] | None = None
    dimensions: PageDimensions


class OCRPage(BasePage):
    """Represents a single page from an OCR result."""
    pass


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
    # Metrics for OCR; can include token-usage or page-processing metrics
    metrics: list[TokenUsageMetrics | PagesProcessedMetrics] | None = None

##########################################
#               Processing               #
##########################################


class ProcessedPage(BasePage):
    """Represents a single page after text processors have been applied."""
    pass


class ProcessingTraceItem(BaseModel):
    """Trace of a single processor's execution."""
    step_name: str  # Name of the step in the processing
    summary: str
    duration: float  # Duration in seconds


class ProcessingResult(BaseModel):
    """Represents the output of the text processors."""
    processor_name: str
    processed_at: datetime
    pages: list[ProcessedPage]  # The text content, now structured per page
    processing_trace: list[ProcessingTraceItem]
    extracted_data: dict[str, object]  # For footnotes, references, etc.
    processing_duration: float | None = None
    metrics: TokenUsageMetrics | None = None

##########################################
#                Chunking                #
##########################################


class ChunkMetadata(BaseModel):
    """Metadata associated with a chunk."""
    page_index: int
    source_references: list[str] | None = None
    footnotes: list[dict] | None = None
    images: list[Image] | None = None
    extra: dict[str, object] = {}


class Chunk(BaseModel):
    """Represents a chunk of text, ready for embedding."""
    text: str
    metadata: ChunkMetadata


class ChunkingResult(BaseModel):
    """Represents the output of the chunking process."""
    processed_at: datetime
    chunks: list[Chunk]
    metrics: TokenUsageMetrics | None = None  # If chunker uses an LLM


##########################################
#                Embedding               #
##########################################


class ChunkWithEmbedding(Chunk):
    """Represents a chunk that has been embedded."""
    embedding: list[float]
    embedding_model: str


class EmbeddingResult(BaseModel):
    """Represents the output of the embedding process."""
    processed_at: datetime | None = None
    chunks_with_embeddings: list[ChunkWithEmbedding]
    model_name: str
    metrics: TokenUsageMetrics | None = None

##########################################
#            Document Pipeline           #
##########################################


class StepStatus(str, Enum):
    success = "success"
    failed = "failed"
    skipped = "skipped"


class StepReport(BaseModel):
    ocr: StepStatus | None = None
    processing: StepStatus | None = None
    chunking: StepStatus | None = None
    embedding: StepStatus | None = None


class DocumentPipelineResult(BaseModel):
    """The composite result for an end-to-end pipeline run."""
    source_file_path: str
    processed_at: datetime
    pipeline_config: dict
    timings: TimingMetrics
    total_metrics: (
        list[TokenUsageMetrics | PagesProcessedMetrics] | None
    ) = None
    total_cost_usd: float | None = None
    ocr_result: OCRResult | None = None
    processing_results: list[ProcessingResult] | None = None
    chunking_result: ChunkingResult | None = None
    embedding_result: EmbeddingResult | None = None
    step_report: StepReport
    logs: str | None = None
