"""
Document processing module for handling, analyzing, and transforming documents.

This package provides tools for OCR, chunking,
embedding, and processing documents.
"""

# Import key components from subdirectories
from ragbandit.documents.document_pipeline import DocumentPipeline

# Import from chunkers
from ragbandit.documents.chunkers import (
    BaseChunker,
    FixedSizeChunker,
    SemanticChunker,
    SemanticBreak
)

# Import from processors
from ragbandit.documents.processors import (
    BaseProcessor,
    FootnoteProcessor,
    ReferencesProcessor
)

# Import from embedders
from ragbandit.documents.embedders import (
    BaseEmbedder,
    MistralEmbedder
)

# Import from OCR
from ragbandit.documents.ocr import (
    BaseOCR,
    MistralOCRDocument
)

# Import from utils
from ragbandit.documents.utils import SecureFileHandler

__all__ = [
    # Main pipeline
    "DocumentPipeline",

    # Chunkers
    "BaseChunker",
    "FixedSizeChunker",
    "SemanticChunker",
    "SemanticBreak",

    # Processors
    "BaseProcessor",
    "FootnoteProcessor",
    "ReferencesProcessor",

    # Embedders
    "BaseEmbedder",
    "MistralEmbedder",

    # OCR
    "BaseOCR",
    "MistralOCRDocument",

    # Utils
    "SecureFileHandler"
]
