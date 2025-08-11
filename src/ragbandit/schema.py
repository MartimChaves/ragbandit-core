"""Base schema for data structures."""
from mistralai import OCRResponse
from pydantic import BaseModel, RootModel


class ProcessorConfig(BaseModel):
    k: float = 3.0
    min_chunk_size: int = 5


class PartitionMetadata(BaseModel):
    """Metadata for a partition: summary, topics, and chunk_count."""
    summary: str
    topics: list[str]
    chunk_count: int


class PartitionRecord(BaseModel):
    partition_id: str
    document_id: str
    summary_embedding: list[float]
    metadata: PartitionMetadata


class ChunkRecordMetadata(BaseModel):
    text: str
    page_index: int
    images: list[dict] | list
    sequence_in_partition: int


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    partition_id: str
    embedding: list[float]
    metadata: ChunkRecordMetadata


# Model for individual items in the list
class Footnote(BaseModel):
    symbol: str
    text: str


# Model for the dictionary mapping int keys to lists of DataEntry
class FootnoteDictModel(RootModel):
    root: dict[int, list[Footnote]]


class ExtendedOCRResponse(OCRResponse):
    processing_metadata: dict[str, str] | None = None


class ChunkImageModel(BaseModel):
    img_id: str
    image_base64: str


class ChunkModel(BaseModel):
    chunk_text: str
    page_index: int
    embedding: list[float]
    images: list[ChunkImageModel] | None = None


class ChunksModel(BaseModel):
    chunks: list[ChunkModel]
