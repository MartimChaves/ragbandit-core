"""
Integration tests for OCR processors.

These tests make real API calls and require valid API keys.
Set MISTRAL_API_KEY and DATALAB_API_KEY environment variables before running.
"""

import pytest
import os
from pathlib import Path
from ragbandit.documents.ocr.mistral_ocr import MistralOCR
from ragbandit.documents.ocr.datalab_ocr import DatalabOCR


@pytest.fixture
def mistral_api_key():
    """Get Mistral API key from environment."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY not set")
    return api_key


@pytest.fixture
def datalab_api_key():
    """Get Datalab API key from environment."""
    api_key = os.getenv("DATALAB_API_KEY")
    if not api_key:
        pytest.skip("DATALAB_API_KEY not set")
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
class TestMistralOCR:
    """Integration tests for MistralOCR with real API calls."""

    def test_ocr_with_default_model(self, mistral_api_key, sample_pdf_path):
        """Test OCR processing with default model."""
        ocr = MistralOCR(api_key=mistral_api_key)
        result = ocr.process(sample_pdf_path)

        assert result is not None
        assert result.component_name == "MistralOCR"
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
        ocr = MistralOCR(
            api_key=mistral_api_key,
            model="mistral-ocr-2505"
        )
        result = ocr.process(sample_pdf_path)

        assert result.component_name == "MistralOCR"
        assert result.component_config["model"] == "mistral-ocr-2505"
        assert result.model == "mistral-ocr-2505"
        assert len(result.pages) > 0

    def test_ocr_invalid_model_raises_error(self, mistral_api_key):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            MistralOCR(
                api_key=mistral_api_key,
                model="invalid-model"
            )

    def test_ocr_get_config(self, mistral_api_key):
        """Test get_config returns correct configuration."""
        ocr = MistralOCR(
            api_key=mistral_api_key,
            model="mistral-ocr-2512"
        )
        config = ocr.get_config()

        assert "model" in config
        assert config["model"] == "mistral-ocr-2512"

    def test_ocr_get_name(self, mistral_api_key):
        """Test get_name returns correct component name."""
        ocr = MistralOCR(api_key=mistral_api_key)
        assert ocr.get_name() == "MistralOCR"

    def test_ocr_result_has_metrics(self, mistral_api_key, sample_pdf_path):
        """Test that OCR result includes token usage metrics."""
        ocr = MistralOCR(api_key=mistral_api_key)
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
        ocr = MistralOCR(api_key=mistral_api_key)
        result = ocr.process(sample_pdf_path)

        # Verify pages are indexed correctly
        for i, page in enumerate(result.pages):
            assert page.index == i
            assert len(page.markdown) > 0
            assert page.dimensions.width > 0
            assert page.dimensions.height > 0


@pytest.mark.integration
class TestDatalabOCR:
    """Integration tests for DatalabOCR with real API calls."""

    def test_ocr_with_default_model(self, datalab_api_key, sample_pdf_path):
        """Test OCR processing with default model (marker)."""
        ocr = DatalabOCR(api_key=datalab_api_key)
        result = ocr.process(sample_pdf_path)

        assert result is not None
        assert result.component_name == "DatalabOCR"
        assert "model" in result.component_config
        assert result.source_file_path == sample_pdf_path
        assert len(result.pages) > 0
        assert result.model == "marker"

        # Check first page has content
        first_page = result.pages[0]
        assert first_page.index == 0
        assert len(first_page.markdown) > 0
        assert first_page.dimensions is not None

    def test_ocr_with_different_modes(
        self, datalab_api_key, sample_pdf_path
    ):
        """Test OCR with different modes (fast, balanced, accurate)."""
        for mode in ["fast", "balanced", "accurate"]:
            ocr = DatalabOCR(
                api_key=datalab_api_key,
                mode=mode
            )
            result = ocr.process(sample_pdf_path)

            assert result.component_name == "DatalabOCR"
            assert result.component_config["mode"] == mode
            assert len(result.pages) > 0

    def test_ocr_invalid_model_raises_error(self, datalab_api_key):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            DatalabOCR(
                api_key=datalab_api_key,
                model="invalid-model"
            )

    def test_ocr_invalid_mode_raises_error(self, datalab_api_key):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            DatalabOCR(
                api_key=datalab_api_key,
                mode="invalid-mode"
            )

    def test_ocr_get_config(self, datalab_api_key):
        """Test get_config returns correct configuration."""
        ocr = DatalabOCR(
            api_key=datalab_api_key,
            model="marker",
            mode="balanced",
            max_pages=10,
            disable_image_extraction=True
        )
        config = ocr.get_config()

        assert "model" in config
        assert config["model"] == "marker"
        assert config["mode"] == "balanced"
        assert config["max_pages"] == 10
        assert config["disable_image_extraction"] is True

    def test_ocr_get_name(self, datalab_api_key):
        """Test get_name returns correct component name."""
        ocr = DatalabOCR(api_key=datalab_api_key)
        assert ocr.get_name() == "DatalabOCR"

    def test_ocr_result_has_metrics(self, datalab_api_key, sample_pdf_path):
        """Test that OCR result includes cost metrics."""
        ocr = DatalabOCR(api_key=datalab_api_key)
        result = ocr.process(sample_pdf_path)

        assert result.metrics is not None
        # metrics is a list of PagesProcessedMetrics
        if result.metrics:
            pages_metrics = [
                m for m in result.metrics
                if hasattr(m, 'pages_processed')
            ]
            if pages_metrics:
                assert pages_metrics[0].pages_processed > 0
                assert hasattr(pages_metrics[0], 'total_cost_usd')

    def test_ocr_multiple_pages(self, datalab_api_key, sample_pdf_path):
        """Test OCR processing handles multiple pages correctly."""
        ocr = DatalabOCR(api_key=datalab_api_key)
        result = ocr.process(sample_pdf_path)

        # Verify pages are indexed correctly
        for i, page in enumerate(result.pages):
            assert page.index == i
            assert len(page.markdown) > 0

    def test_ocr_with_page_range(self, datalab_api_key, sample_pdf_path):
        """Test OCR processing with specific page range."""
        ocr = DatalabOCR(
            api_key=datalab_api_key,
            page_range="0-1"  # Only first two pages
        )
        result = ocr.process(sample_pdf_path)

        assert result.component_name == "DatalabOCR"
        assert len(result.pages) <= 2

    def test_ocr_with_max_pages(self, datalab_api_key, sample_pdf_path):
        """Test OCR processing with max_pages limit."""
        ocr = DatalabOCR(
            api_key=datalab_api_key,
            max_pages=1
        )
        result = ocr.process(sample_pdf_path)

        assert result.component_name == "DatalabOCR"
        assert len(result.pages) <= 1

    def test_ocr_behavior_markdown_content(
        self, datalab_api_key, sample_pdf_path
    ):
        """Behavior test: Verify markdown content is extracted."""
        ocr = DatalabOCR(api_key=datalab_api_key)
        result = ocr.process(sample_pdf_path)

        # Check that we got actual content
        assert len(result.pages) > 0

        # Check that markdown contains actual text (not just whitespace)
        for page in result.pages:
            assert page.markdown.strip(), (
                f"Page {page.index} has empty markdown"
            )
            # Markdown should have some reasonable length
            assert len(page.markdown.strip()) > 10, \
                f"Page {page.index} markdown too short: {len(page.markdown)}"

    def test_ocr_behavior_usage_info(self, datalab_api_key, sample_pdf_path):
        """Behavior test: Verify usage info is populated correctly."""
        ocr = DatalabOCR(api_key=datalab_api_key)
        result = ocr.process(sample_pdf_path)

        # Check usage info
        assert result.usage_info is not None
        assert result.usage_info.pages_processed > 0
        assert result.usage_info.doc_size_bytes > 0

        # Verify doc size matches actual file
        actual_size = Path(sample_pdf_path).stat().st_size
        assert result.usage_info.doc_size_bytes == actual_size

    def test_ocr_behavior_processed_timestamp(
        self, datalab_api_key, sample_pdf_path
    ):
        """Behavior test: Verify processed_at timestamp is set."""
        import datetime

        before = datetime.datetime.now()
        ocr = DatalabOCR(api_key=datalab_api_key)
        result = ocr.process(sample_pdf_path)
        after = datetime.datetime.now()

        # Check timestamp is within reasonable range
        assert result.processed_at is not None
        assert before <= result.processed_at <= after

    def test_ocr_behavior_image_extraction(
        self, datalab_api_key, sample_pdf_path
    ):
        """Behavior test: Verify images are extracted when enabled."""
        ocr_with_images = DatalabOCR(
            api_key=datalab_api_key,
            disable_image_extraction=False
        )
        result_with_images = ocr_with_images.process(sample_pdf_path)

        # If the PDF has images, they should be in the result
        # Check that image structure is correct if images exist
        for page in result_with_images.pages:
            if page.images:
                for img in page.images:
                    assert img.id is not None
                    assert img.image_base64 is not None
                    # Should have data URI prefix
                    assert img.image_base64.startswith('data:image/')

    def test_ocr_behavior_no_api_key_raises_error(self):
        """Behavior test: Verify error when no API key provided."""
        # Clear environment variable temporarily
        old_key = os.environ.get("DATALAB_API_KEY")
        if old_key:
            del os.environ["DATALAB_API_KEY"]

        try:
            with pytest.raises(ValueError, match="API key must be provided"):
                DatalabOCR(api_key=None)
        finally:
            # Restore environment variable
            if old_key:
                os.environ["DATALAB_API_KEY"] = old_key
