"""
Integration tests for OCR processors.

These tests make real API calls and require valid API keys.
Set MISTRAL_API_KEY environment variable before running.
"""

import pytest
import os
from pathlib import Path
from ragbandit.documents.ocr.mistral_ocr import MistralOCRDocument


@pytest.fixture
def mistral_api_key():
    """Get Mistral API key from environment."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY not set")
    return api_key


@pytest.fixture
def sample_pdf_path():
    """Path to a sample PDF for testing."""
    # You'll need to provide a sample PDF in tests/fixtures/
    pdf_path = Path(__file__).parent.parent / "fixtures" / "sample.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF not found at {pdf_path}")
    return str(pdf_path)


@pytest.mark.integration
class TestMistralOCRDocument:
    """Integration tests for MistralOCRDocument with real API calls."""

    def test_ocr_with_default_model(self, mistral_api_key, sample_pdf_path):
        """Test OCR processing with default model."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        result = ocr.process(sample_pdf_path)

        assert result is not None
        assert result.component_name == "MistralOCRDocument"
        assert "model" in result.component_config
        assert result.source_file_path == sample_pdf_path
        assert len(result.pages) > 0
        assert result.model == "mistral-ocr-2512"

        # Check first page has content
        first_page = result.pages[0]
        assert first_page.index == 0
        assert len(first_page.markdown) > 0
        assert first_page.dimensions is not None

    def test_ocr_with_specific_model(self, mistral_api_key, sample_pdf_path):
        """Test OCR processing with specific model."""
        ocr = MistralOCRDocument(
            api_key=mistral_api_key,
            model="mistral-ocr-2505"
        )
        result = ocr.process(sample_pdf_path)

        assert result.component_name == "MistralOCRDocument"
        assert result.component_config["model"] == "mistral-ocr-2505"
        assert result.model == "mistral-ocr-2505"
        assert len(result.pages) > 0

    def test_ocr_invalid_model_raises_error(self, mistral_api_key):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            MistralOCRDocument(
                api_key=mistral_api_key,
                model="invalid-model"
            )

    def test_ocr_get_config(self, mistral_api_key):
        """Test get_config returns correct configuration."""
        ocr = MistralOCRDocument(
            api_key=mistral_api_key,
            model="mistral-ocr-2512"
        )
        config = ocr.get_config()

        assert "model" in config
        assert config["model"] == "mistral-ocr-2512"

    def test_ocr_get_name(self, mistral_api_key):
        """Test get_name returns correct component name."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        assert ocr.get_name() == "MistralOCRDocument"

    def test_ocr_result_has_metrics(self, mistral_api_key, sample_pdf_path):
        """Test that OCR result includes token usage metrics."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        result = ocr.process(sample_pdf_path)

        assert result.metrics is not None
        # metrics is a list of TokenUsageMetrics or PagesProcessedMetrics
        if result.metrics:
            # Check if we have token metrics
            token_metrics = [
                m for m in result.metrics
                if hasattr(m, 'total_tokens')
            ]
            if token_metrics:
                assert token_metrics[0].total_tokens > 0

    def test_ocr_multiple_pages(self, mistral_api_key, sample_pdf_path):
        """Test OCR processing handles multiple pages correctly."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        result = ocr.process(sample_pdf_path)

        # Verify pages are indexed correctly
        for i, page in enumerate(result.pages):
            assert page.index == i
            assert len(page.markdown) > 0
            assert page.dimensions.width > 0
            assert page.dimensions.height > 0
