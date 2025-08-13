"""
OCR (Optical Character Recognition) implementations for document processing.

This module provides OCR processors that convert document images to text.
"""

from ragbandit.documents.ocr.base_ocr import BaseOCR
from ragbandit.documents.ocr.mistral_ocr import MistralOCRDocument

__all__ = [
    "BaseOCR",
    "MistralOCRDocument"
]
