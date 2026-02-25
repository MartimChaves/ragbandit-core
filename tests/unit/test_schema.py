"""
Unit tests for ragbandit schema models.

Tests the Pydantic models defined in ragbandit.schema to ensure
proper validation and serialization.
"""

import pytest
from ragbandit.schema import (
    OCRResult,
    OCRPage,
    OCRUsageInfo,
    PageDimensions,
    RefiningResult,
    ChunkingResult,
    Chunk,
    ChunkMetadata,
    EmbeddingResult,
    ChunkWithEmbedding,
)


@pytest.mark.unit
class TestPageDimensions:
    """Tests for PageDimensions schema."""

    def test_create_page_dimensions(self):
        """Test creating a PageDimensions object."""
        dims = PageDimensions(dpi=72, width=612, height=792)
        assert dims.width == 612
        assert dims.height == 792
        assert dims.dpi == 72

    def test_page_dimensions_validation(self):
        """Test that PageDimensions validates numeric types."""
        with pytest.raises(Exception):  # Pydantic validation error
            PageDimensions(dpi=72, width="invalid", height=792)


@pytest.mark.unit
class TestOCRPage:
    """Tests for OCRPage schema."""

    def test_create_ocr_page(self):
        """Test creating an OCRPage object."""
        page = OCRPage(
            index=0,
            markdown="# Test Page",
            images=[],
            dimensions=PageDimensions(dpi=72, width=612, height=792)
        )
        assert page.index == 0
        assert page.markdown == "# Test Page"
        assert len(page.images) == 0
        assert page.dimensions.width == 612

    def test_ocr_page_serialization(self):
        """Test OCRPage can be serialized to dict."""
        page = OCRPage(
            index=0,
            markdown="# Test",
            images=[],
            dimensions=PageDimensions(dpi=72, width=612, height=792)
        )
        page_dict = page.model_dump()
        assert page_dict["index"] == 0
        assert page_dict["markdown"] == "# Test"


@pytest.mark.unit
class TestOCRResult:
    """Tests for OCRResult schema."""

    def test_create_ocr_result(self, sample_timestamp):
        """Test creating an OCRResult object."""
        result = OCRResult(
            component_name="MistralOCR",
            component_config={"model": "mistral-ocr-2512"},
            source_file_path="/path/to/file.pdf",
            processed_at=sample_timestamp,
            model="mistral-ocr-2512",
            pages=[],
            usage_info=OCRUsageInfo(pages_processed=0, doc_size_bytes=0),
        )
        assert result.component_name == "MistralOCR"
        assert result.component_config["model"] == "mistral-ocr-2512"
        assert result.source_file_path == "/path/to/file.pdf"
        assert len(result.pages) == 0

    def test_ocr_result_with_pages(self, sample_timestamp):
        """Test OCRResult with multiple pages."""
        pages = [
            OCRPage(
                index=i,
                markdown=f"# Page {i}",
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            )
            for i in range(3)
        ]
        result = OCRResult(
            component_name="MistralOCR",
            component_config={"model": "mistral-ocr-2512"},
            source_file_path="/path/to/file.pdf",
            processed_at=sample_timestamp,
            model="mistral-ocr-2512",
            pages=pages,
            usage_info=OCRUsageInfo(pages_processed=3, doc_size_bytes=1024),
        )
        assert len(result.pages) == 3
        assert result.pages[1].index == 1


@pytest.mark.unit
class TestRefiningResult:
    """Tests for RefiningResult schema."""

    def test_create_refining_result(self, sample_timestamp):
        """Test creating a RefiningResult object."""
        result = RefiningResult(
            component_name="FootnoteRefiner",
            component_config={"inline_explanations": True},
            processed_at=sample_timestamp,
            pages=[],
            refining_trace=[],
            extracted_data={},
        )
        assert result.component_name == "FootnoteRefiner"
        assert result.component_config["inline_explanations"] is True
        assert len(result.pages) == 0
        assert isinstance(result.refining_trace, list)

    def test_refining_result_with_extracted_data(self, sample_timestamp):
        """Test RefiningResult with extracted data."""
        result = RefiningResult(
            component_name="ReferencesRefiner",
            component_config={"extract_references": True},
            processed_at=sample_timestamp,
            pages=[],
            refining_trace=[],
            extracted_data={
                "references_markdown": "# References\n\n1. Paper 1"
            },
        )
        assert "references_markdown" in result.extracted_data
        refs_md = result.extracted_data["references_markdown"]
        assert refs_md.startswith("# References")


@pytest.mark.unit
class TestChunk:
    """Tests for Chunk schema."""

    def test_create_chunk(self):
        """Test creating a Chunk object."""
        chunk = Chunk(
            text="This is a chunk of text.",
            metadata=ChunkMetadata(page_index=0)
        )
        assert chunk.text == "This is a chunk of text."
        assert chunk.metadata.page_index == 0


@pytest.mark.unit
class TestChunkingResult:
    """Tests for ChunkingResult schema."""

    def test_create_chunking_result(self, sample_timestamp):
        """Test creating a ChunkingResult object."""
        chunks = [
            Chunk(
                text=f"Chunk {i}",
                metadata=ChunkMetadata(page_index=0)
            )
            for i in range(3)
        ]
        result = ChunkingResult(
            component_name="FixedSizeChunker",
            component_config={"chunk_size": 1000, "overlap": 200},
            processed_at=sample_timestamp,
            chunks=chunks,
        )
        assert result.component_name == "FixedSizeChunker"
        assert len(result.chunks) == 3
        assert result.component_config["chunk_size"] == 1000


@pytest.mark.unit
class TestChunkWithEmbedding:
    """Tests for ChunkWithEmbedding schema."""

    def test_create_chunk_with_embedding(self):
        """Test creating a ChunkWithEmbedding object."""
        chunk = ChunkWithEmbedding(
            text="Sample text",
            metadata=ChunkMetadata(page_index=0),
            embedding=[0.1, 0.2, 0.3],
            embedding_model="mistral-embed"
        )
        assert chunk.text == "Sample text"
        assert len(chunk.embedding) == 3
        assert chunk.embedding_model == "mistral-embed"


@pytest.mark.unit
class TestEmbeddingResult:
    """Tests for EmbeddingResult schema."""

    def test_create_embedding_result(self, sample_timestamp):
        """Test creating an EmbeddingResult object."""
        chunks = [
            ChunkWithEmbedding(
                text=f"Chunk {i}",
                metadata=ChunkMetadata(page_index=0),
                embedding=[0.1 * i, 0.2 * i, 0.3 * i],
                embedding_model="mistral-embed"
            )
            for i in range(3)
        ]
        result = EmbeddingResult(
            component_name="MistralEmbedder",
            component_config={"model": "mistral-embed"},
            processed_at=sample_timestamp,
            chunks_with_embeddings=chunks,
            model_name="mistral-embed",
        )
        assert result.component_name == "MistralEmbedder"
        assert len(result.chunks_with_embeddings) == 3
        assert result.model_name == "mistral-embed"
