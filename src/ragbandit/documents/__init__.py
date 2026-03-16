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
    SemanticBreak,
    SentenceChunker,
    RecursiveMarkdownChunker,
)

# Import from refiners
from ragbandit.documents.refiners import (
    BaseRefiner,
    FootnoteRefiner,
    ReferencesRefiner,
    TableOfContentsRefiner,
)

# Import from embedders
from ragbandit.documents.embedders import (
    BaseEmbedder,
    MistralEmbedder,
    OpenAIEmbedder,
    VoyageAIEmbedder,
    CohereEmbedder,
)

# Import from OCR
from ragbandit.documents.ocr import (
    BaseOCR,
    MistralOCR,
    DatalabOCR
)

__all__ = [
    # Main pipeline
    "DocumentPipeline",

    # Chunkers
    "BaseChunker",
    "FixedSizeChunker",
    "SemanticChunker",
    "SemanticBreak",
    "SentenceChunker",
    "RecursiveMarkdownChunker",

    # Refiners
    "BaseRefiner",
    "FootnoteRefiner",
    "ReferencesRefiner",
    "TableOfContentsRefiner",

    # Embedders
    "BaseEmbedder",
    "MistralEmbedder",
    "OpenAIEmbedder",
    "VoyageAIEmbedder",
    "CohereEmbedder",

    # OCR
    "BaseOCR",
    "MistralOCR",
    "DatalabOCR",
]
