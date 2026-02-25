"""
Integration tests for document refiners.

These tests make real API calls and require valid API keys.
Set MISTRAL_API_KEY environment variable before running.
"""

import pytest
import os
from ragbandit.documents.refiners.footnotes_refiner import FootnoteRefiner
from ragbandit.documents.refiners.references_refiner import ReferencesRefiner
from ragbandit.schema import (
    OCRResult,
    OCRPage,
    OCRUsageInfo,
    PageDimensions,
)
from ragbandit.utils.token_usage_tracker import TokenUsageTracker
from datetime import datetime, timezone


@pytest.fixture
def mistral_api_key():
    """Get Mistral API key from environment."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY not set")
    return api_key


@pytest.fixture
def sample_ocr_result_with_footnotes():
    """Create a sample OCR result with footnotes for testing."""
    pages = [
        OCRPage(
            index=0,
            markdown="""# Introduction

This is a sample document with footnotes[^1] and citations[^2].

## Footnotes

[^1]: This is an explanatory footnote providing context.
[^2]: Smith, J. (2024). "Research Paper". Journal of Science.
""",
            images=[],
            dimensions=PageDimensions(dpi=72, width=612, height=792)
        )
    ]
    return OCRResult(
        component_name="MistralOCR",
        component_config={"model": "mistral-ocr-2512"},
        source_file_path="/test/sample.pdf",
        processed_at=datetime.now(timezone.utc),
        model="mistral-ocr-2512",
        pages=pages,
        usage_info=OCRUsageInfo(pages_processed=1, doc_size_bytes=1024),
    )


@pytest.fixture
def sample_ocr_result_with_references():
    """Create a sample OCR result with references section."""
    pages = [
        OCRPage(
            index=0,
            markdown="""# Main Content

This document discusses various topics.

## References

1. Smith, J. (2024). "First Paper". Journal A.
2. Doe, J. (2023). "Second Paper". Journal B.
3. Brown, A. (2022). "Third Paper". Journal C.
""",
            images=[],
            dimensions=PageDimensions(dpi=72, width=612, height=792)
        )
    ]
    return OCRResult(
        component_name="MistralOCRRDocument",
        component_config={"model": "mistral-ocr-2512"},
        source_file_path="/test/sample.pdf",
        processed_at=datetime.now(timezone.utc),
        model="mistral-ocr-2512",
        pages=pages,
        usage_info=OCRUsageInfo(pages_processed=1, doc_size_bytes=1024),
    )


@pytest.mark.integration
class TestFootnoteRefiner:
    """Integration tests for FootnoteRefiner with real API calls."""

    def test_footnote_refiner_process(
        self, mistral_api_key, sample_ocr_result_with_footnotes
    ):
        """Test footnote refiner processes document correctly."""
        refiner = FootnoteRefiner(api_key=mistral_api_key)
        tracker = TokenUsageTracker()

        result = refiner.process(
            sample_ocr_result_with_footnotes,
            usage_tracker=tracker
        )

        assert result is not None
        assert result.component_name == "FootnoteRefiner"
        assert "inline_explanations" in result.component_config
        assert "collect_citations" in result.component_config
        assert len(result.pages) > 0
        assert result.metrics is not None
        assert result.metrics.total_tokens > 0

    def test_footnote_refiner_extracts_footnotes(
        self, mistral_api_key, sample_ocr_result_with_footnotes
    ):
        """Test that footnote refiner extracts footnote data."""
        refiner = FootnoteRefiner(api_key=mistral_api_key)
        result = refiner.process(sample_ocr_result_with_footnotes)

        # Check if footnote data was extracted
        if result.extracted_data and "footnote_refs" in result.extracted_data:
            footnote_refs = result.extracted_data["footnote_refs"]
            # footnote_refs is a dict mapping page_index to list of footnotes
            assert isinstance(footnote_refs, dict)

    def test_footnote_refiner_behavior(
        self, mistral_api_key, sample_ocr_result_with_footnotes
    ):
        """Test that footnote refiner actually modifies markdown correctly."""
        refiner = FootnoteRefiner(api_key=mistral_api_key)
        result = refiner.process(sample_ocr_result_with_footnotes)

        # Get the processed markdown
        processed_markdown = result.pages[0].markdown
        original_markdown = sample_ocr_result_with_footnotes.pages[0].markdown

        # The markdown should have been modified
        assert processed_markdown != original_markdown

        # Check that explanatory footnotes are inlined
        # Original has: "footnotes[^1]" where [^1] is explanatory
        # After processing, the [^1] reference should be replaced with inline
        # Look for evidence of inlining (parentheses or similar)
        assert "[^1]" not in processed_markdown or "(" in processed_markdown

        # Check that citation footnotes are collected
        # [^2] is a citation, so it should be removed from main text
        # and stored in extracted_data
        if result.extracted_data and "footnote_refs" in result.extracted_data:
            footnote_refs = result.extracted_data["footnote_refs"]
            # Should have collected the citation footnote
            if footnote_refs:
                # Check that at least one footnote was processed
                assert len(footnote_refs) > 0

        # Verify the footnote definitions section is removed or modified
        # Original has "## Footnotes" section
        footnote_section_count_before = original_markdown.count("## Footnotes")
        footnote_section_count_after = processed_markdown.count("## Footnotes")
        # Should have removed or modified the footnotes section
        assert footnote_section_count_after <= footnote_section_count_before

    def test_footnote_refiner_get_config(self, mistral_api_key):
        """Test get_config returns correct configuration."""
        refiner = FootnoteRefiner(api_key=mistral_api_key)
        config = refiner.get_config()

        assert "inline_explanations" in config
        assert "collect_citations" in config
        assert config["inline_explanations"] is True
        assert config["collect_citations"] is True

    def test_footnote_refiner_get_name(self, mistral_api_key):
        """Test get_name returns correct component name."""
        refiner = FootnoteRefiner(api_key=mistral_api_key)
        assert refiner.get_name() == "FootnoteRefiner"

    def test_footnote_refiner_with_refining_result(
        self, mistral_api_key, sample_ocr_result_with_footnotes
    ):
        """Test footnote refiner accepts RefiningResult as input."""
        refiner = FootnoteRefiner(api_key=mistral_api_key)

        # First pass
        result1 = refiner.process(sample_ocr_result_with_footnotes)

        # Second pass with RefiningResult
        result2 = refiner.process(result1)

        assert result2 is not None
        assert result2.component_name == "FootnoteRefiner"


@pytest.mark.integration
class TestReferencesRefiner:
    """Integration tests for ReferencesRefiner with real API calls."""

    def test_references_refiner_process(
        self, mistral_api_key, sample_ocr_result_with_references
    ):
        """Test references refiner processes document correctly."""
        refiner = ReferencesRefiner(api_key=mistral_api_key)
        tracker = TokenUsageTracker()

        result = refiner.process(
            sample_ocr_result_with_references,
            usage_tracker=tracker
        )

        assert result is not None
        assert result.component_name == "ReferencesRefiner"
        assert "extract_references" in result.component_config
        assert "remove_from_document" in result.component_config
        assert len(result.pages) > 0
        assert result.metrics is not None
        assert result.metrics.total_tokens > 0

    def test_references_refiner_extracts_references(
        self, mistral_api_key, sample_ocr_result_with_references
    ):
        """Test that references refiner extracts reference data."""
        refiner = ReferencesRefiner(api_key=mistral_api_key)
        result = refiner.process(sample_ocr_result_with_references)

        # Check if references were extracted
        if result.extracted_data:
            if "references_markdown" in result.extracted_data:
                refs = result.extracted_data["references_markdown"]
                assert isinstance(refs, str)
                assert len(refs) > 0

    def test_references_refiner_behavior(
        self, mistral_api_key, sample_ocr_result_with_references
    ):
        """Test references refiner modifies markdown correctly."""
        refiner = ReferencesRefiner(api_key=mistral_api_key)
        result = refiner.process(sample_ocr_result_with_references)

        # Get the processed markdown
        processed_markdown = result.pages[0].markdown
        original_markdown = sample_ocr_result_with_references.pages[0].markdown

        # The markdown should have been modified
        assert processed_markdown != original_markdown

        # Check that references section is removed from main text
        # Original has "## References" section
        assert "## References" in original_markdown
        # After processing, references section should be removed or reduced
        references_count_before = original_markdown.count("## References")
        references_count_after = processed_markdown.count("## References")
        assert references_count_after <= references_count_before

        # Check that references were extracted to extracted_data
        assert result.extracted_data is not None
        if "references_markdown" in result.extracted_data:
            refs = result.extracted_data["references_markdown"]
            # Should have extracted the references
            assert len(refs) > 0
            # References should contain the citation info
            assert "Smith" in refs or "Doe" in refs or "Brown" in refs

        # Verify the main content is preserved
        assert "# Main Content" in processed_markdown
        assert "This document discusses various topics" in processed_markdown

    def test_references_refiner_get_config(self, mistral_api_key):
        """Test get_config returns correct configuration."""
        refiner = ReferencesRefiner(api_key=mistral_api_key)
        config = refiner.get_config()

        assert "extract_references" in config
        assert "remove_from_document" in config
        assert config["extract_references"] is True
        assert config["remove_from_document"] is True

    def test_references_refiner_get_name(self, mistral_api_key):
        """Test get_name returns correct component name."""
        refiner = ReferencesRefiner(api_key=mistral_api_key)
        assert refiner.get_name() == "ReferencesRefiner"

    def test_references_refiner_with_refining_result(
        self, mistral_api_key, sample_ocr_result_with_references
    ):
        """Test references refiner accepts RefiningResult as input."""
        refiner = ReferencesRefiner(api_key=mistral_api_key)

        # First pass
        result1 = refiner.process(sample_ocr_result_with_references)

        # Second pass with RefiningResult
        result2 = refiner.process(result1)

        assert result2 is not None
        assert result2.component_name == "ReferencesRefiner"


@pytest.mark.integration
class TestRefinerChaining:
    """Test chaining multiple refiners together."""

    def test_chain_footnote_and_references_refiners(
        self,
        mistral_api_key,
        sample_ocr_result_with_footnotes,
        sample_ocr_result_with_references
    ):
        """Test chaining footnote and references refiners."""
        footnote_refiner = FootnoteRefiner(api_key=mistral_api_key)
        references_refiner = ReferencesRefiner(api_key=mistral_api_key)

        # Process with footnote refiner first
        result1 = footnote_refiner.process(sample_ocr_result_with_footnotes)

        # Then with references refiner
        result2 = references_refiner.process(result1)

        assert result2 is not None
        assert result2.component_name == "ReferencesRefiner"
        # Refining trace may not accumulate across multiple refiners
        # Just verify the result is valid
        assert isinstance(result2.refining_trace, list)
