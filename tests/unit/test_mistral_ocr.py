"""Unit tests for Mistral OCR model availability and pricing."""

from ragbandit.config.pricing import OCR_MODEL_COSTS
from ragbandit.documents.ocr.mistral_ocr import MistralOCR


def test_supported_models():
    assert MistralOCR.VALID_MODELS == [
        "mistral-ocr-2512",
        "mistral-ocr-4-0",
        "mistral-ocr-4-1",
    ]


def test_supported_model_pricing():
    assert OCR_MODEL_COSTS == {
        "mistral-ocr-2512": 0.002,
        "mistral-ocr-4-0": 0.004,
        "mistral-ocr-4-1": 0.004,
    }
