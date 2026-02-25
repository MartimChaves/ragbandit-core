"""
Document processing module for handling, analyzing, and transforming documents.

This package provides tools for OCR, refining,
chunking, embedding, and processing documents.
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

# Import from refiners
from ragbandit.documents.refiners import (
    BaseRefiner,
    FootnoteRefiner,
    ReferencesRefiner
)

# Import from embedders
from ragbandit.documents.embedders import (
    BaseEmbedder,
    MistralEmbedder,
    OpenAIEmbedder
)

# Import from OCR
from ragbandit.documents.ocr import (
    BaseOCR,
    MistralOCR,
    DatalabOCR
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

    # Refiners
    "BaseRefiner",
    "FootnoteRefiner",
    "ReferencesRefiner",

    # Embedders
    "BaseEmbedder",
    "MistralEmbedder",
    "OpenAIEmbedder",

    # OCR
    "BaseOCR",
    "MistralOCR",
    "DatalabOCR",

    # Utils
    "SecureFileHandler"
]
