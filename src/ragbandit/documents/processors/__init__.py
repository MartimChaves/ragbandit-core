"""
Document processors for enhancing and transforming document content.

This module provides various processors that can be applied to documents
to extract, transform, or enhance their content.
"""

from ragbandit.documents.processors.base_processor import BaseProcessor
from ragbandit.documents.processors.footnotes_processor import FootnoteProcessor  # noqa
from ragbandit.documents.processors.references_processor import ReferencesProcessor  # noqa

__all__ = [
    "BaseProcessor",
    "FootnoteProcessor",
    "ReferencesProcessor"
]
