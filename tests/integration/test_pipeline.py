"""
Integration tests for DocumentPipeline.

These tests validate the full pipeline execution with real API calls.
"""

import pytest
import os
from datetime import datetime, timezone
from ragbandit.documents.document_pipeline import DocumentPipeline
from ragbandit.documents.ocr.mistral_ocr import MistralOCRDocument
from ragbandit.documents.refiners.footnotes_refiner import FootnoteRefiner
from ragbandit.documents.refiners.references_refiner import ReferencesRefiner
from ragbandit.documents.chunkers.fixed_size_chunker import FixedSizeChunker
from ragbandit.documents.chunkers.semantic_chunker import SemanticChunker
from ragbandit.documents.embedders.mistral_embedder import MistralEmbedder
from ragbandit.schema import (
    OCRResult,
    OCRPage,
    OCRUsageInfo,
    PageDimensions,
    RefiningResult,
    ChunkingResult,
    Chunk,
    ChunkMetadata,
)


@pytest.fixture
def mistral_api_key():
    """Get Mistral API key from environment."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY not set")
    return api_key


@pytest.fixture
def sample_pdf_path():
    """Get path to sample PDF for testing."""
    import pathlib
    pdf_path = pathlib.Path(__file__).parent.parent / "fixtures" / "sample.pdf"
    if not pdf_path.exists():
        pytest.skip("sample.pdf not found in fixtures")
    return str(pdf_path)


@pytest.fixture
def sample_ocr_result():
    """Create a sample OCR result for testing."""
    pages = [
        OCRPage(
            index=0,
            markdown="""# Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience. Deep learning uses
neural networks with multiple layers[^1].

## Applications

Common applications include image recognition, natural language
processing, and recommendation systems.

## References

1. Smith, J. (2024). "Deep Learning Fundamentals". AI Journal.
2. Doe, A. (2024). "Neural Networks Explained". Tech Review.

## Footnotes

[^1]: Deep neural networks can have dozens or hundreds of layers.""",
            images=[],
            dimensions=PageDimensions(dpi=72, width=612, height=792)
        )
    ]
    return OCRResult(
        component_name="MistralOCRDocument",
        component_config={"model": "mistral-ocr-2512"},
        source_file_path="/test/sample.pdf",
        processed_at=datetime.now(timezone.utc),
        model="mistral-ocr-2512",
        pages=pages,
        usage_info=OCRUsageInfo(pages_processed=1, doc_size_bytes=2048),
    )


@pytest.mark.integration
class TestDocumentPipeline:
    """Integration tests for DocumentPipeline."""

    def test_pipeline_initialization(self, mistral_api_key):
        """Test pipeline can be initialized with all components."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        refiners = [
            FootnoteRefiner(api_key=mistral_api_key),
            ReferencesRefiner(api_key=mistral_api_key),
        ]
        chunker = FixedSizeChunker(chunk_size=500, overlap=100)
        embedder = MistralEmbedder(api_key=mistral_api_key)

        pipeline = DocumentPipeline(
            ocr_processor=ocr,
            refiners=refiners,
            chunker=chunker,
            embedder=embedder,
        )

        assert pipeline.ocr_processor is not None
        assert len(pipeline.refiners) == 2
        assert pipeline.chunker is not None
        assert pipeline.embedder is not None

    def test_pipeline_add_refiner(self, mistral_api_key):
        """Test adding refiners to pipeline."""
        pipeline = DocumentPipeline()
        refiner = FootnoteRefiner(api_key=mistral_api_key)

        pipeline.add_refiner(refiner)

        assert len(pipeline.refiners) == 1
        assert pipeline.refiners[0] == refiner

    def test_pipeline_run_refiners(
        self, mistral_api_key, sample_ocr_result
    ):
        """Test running refiners through pipeline."""
        refiners = [
            FootnoteRefiner(api_key=mistral_api_key),
            ReferencesRefiner(api_key=mistral_api_key),
        ]
        pipeline = DocumentPipeline(refiners=refiners)

        results = pipeline.run_refiners(sample_ocr_result)

        assert len(results) == 2
        assert all(isinstance(r, RefiningResult) for r in results)
        assert results[0].component_name == "FootnoteRefiner"
        assert results[1].component_name == "ReferencesRefiner"
        # Each refiner should have metrics
        assert all(r.metrics is not None for r in results)

    def test_pipeline_run_chunker(self, sample_ocr_result):
        """Test running chunker through pipeline."""
        chunker = FixedSizeChunker(chunk_size=300, overlap=50)
        pipeline = DocumentPipeline(chunker=chunker)

        result = pipeline.run_chunker(sample_ocr_result)

        assert isinstance(result, ChunkingResult)
        assert result.component_name == "FixedSizeChunker"
        assert len(result.chunks) > 0
        assert result.component_config["chunk_size"] == 300

    def test_pipeline_run_embedder(self, mistral_api_key):
        """Test running embedder through pipeline."""
        chunks = [
            Chunk(
                text="Machine learning is fascinating.",
                metadata=ChunkMetadata(page_index=0)
            ),
            Chunk(
                text="Deep learning uses neural networks.",
                metadata=ChunkMetadata(page_index=0)
            ),
        ]
        chunking_result = ChunkingResult(
            component_name="TestChunker",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            chunks=chunks,
        )

        embedder = MistralEmbedder(api_key=mistral_api_key)
        pipeline = DocumentPipeline(embedder=embedder)

        result = pipeline.run_embedder(chunking_result)

        assert result.component_name == "MistralEmbedder"
        assert len(result.chunks_with_embeddings) == 2
        assert all(
            len(c.embedding) > 0
            for c in result.chunks_with_embeddings
        )

    def test_pipeline_configuration_tracking(self, mistral_api_key):
        """Test that pipeline tracks component configurations."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        refiners = [FootnoteRefiner(api_key=mistral_api_key)]
        chunker = FixedSizeChunker(chunk_size=500, overlap=100)
        embedder = MistralEmbedder(
            api_key=mistral_api_key,
            model="mistral-embed"
        )

        pipeline = DocumentPipeline(
            ocr_processor=ocr,
            refiners=refiners,
            chunker=chunker,
            embedder=embedder,
        )

        # Verify configuration is tracked
        assert pipeline.ocr_processor.get_name() == "MistralOCRDocument"
        assert pipeline.refiners[0].get_name() == "FootnoteRefiner"
        assert pipeline.chunker.get_name() == "FixedSizeChunker"
        assert pipeline.embedder.get_name() == "MistralEmbedder"

        # Verify get_config works
        assert "model" in pipeline.ocr_processor.get_config()
        assert "inline_explanations" in pipeline.refiners[0].get_config()
        assert "chunk_size" in pipeline.chunker.get_config()
        assert "model" in pipeline.embedder.get_config()

    def test_pipeline_error_handling_no_ocr(self):
        """Test pipeline raises error when OCR is missing."""
        pipeline = DocumentPipeline()

        with pytest.raises(ValueError, match="ocr_processor is required"):
            pipeline.run_ocr("/test/sample.pdf")

    def test_pipeline_error_handling_no_chunker(self, sample_ocr_result):
        """Test pipeline raises error when chunker is missing."""
        pipeline = DocumentPipeline()

        with pytest.raises(ValueError, match="chunker is required"):
            pipeline.run_chunker(sample_ocr_result)

    def test_pipeline_error_handling_no_embedder(self):
        """Test pipeline raises error when embedder is missing."""
        chunks = [
            Chunk(
                text="Test",
                metadata=ChunkMetadata(page_index=0)
            )
        ]
        chunking_result = ChunkingResult(
            component_name="TestChunker",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            chunks=chunks,
        )

        pipeline = DocumentPipeline()

        with pytest.raises(ValueError, match="embedder is required"):
            pipeline.run_embedder(chunking_result)

    def test_pipeline_refiner_chaining(
        self, mistral_api_key, sample_ocr_result
    ):
        """Test that refiners are chained correctly."""
        refiner1 = FootnoteRefiner(api_key=mistral_api_key)
        refiner2 = ReferencesRefiner(api_key=mistral_api_key)

        pipeline = DocumentPipeline(refiners=[refiner1, refiner2])

        results = pipeline.run_refiners(sample_ocr_result)

        # Second refiner should receive output from first
        assert len(results) == 2
        # Both should have processed the document
        assert results[0].component_name == "FootnoteRefiner"
        assert results[1].component_name == "ReferencesRefiner"

    def test_pipeline_with_semantic_chunker(
        self, mistral_api_key, sample_ocr_result
    ):
        """Test pipeline with semantic chunker."""
        chunker = SemanticChunker(
            api_key=mistral_api_key,
            min_chunk_size=100
        )
        pipeline = DocumentPipeline(chunker=chunker)

        result = pipeline.run_chunker(sample_ocr_result)

        assert result.component_name == "SemanticChunker"
        assert len(result.chunks) > 0
        assert result.component_config["min_chunk_size"] == 100

    def test_pipeline_metrics_tracking(
        self, mistral_api_key, sample_ocr_result
    ):
        """Test that pipeline tracks metrics correctly."""
        refiners = [
            FootnoteRefiner(api_key=mistral_api_key),
            ReferencesRefiner(api_key=mistral_api_key),
        ]
        pipeline = DocumentPipeline(refiners=refiners)

        results = pipeline.run_refiners(sample_ocr_result)

        # Each refiner should have metrics
        for result in results:
            assert result.metrics is not None
            assert hasattr(result.metrics, 'total_tokens')
            assert result.metrics.total_tokens > 0

    def test_pipeline_timing_tracking(
        self, mistral_api_key, sample_ocr_result
    ):
        """Test that pipeline tracks timing for refiners."""
        refiners = [FootnoteRefiner(api_key=mistral_api_key)]
        pipeline = DocumentPipeline(refiners=refiners)

        results = pipeline.run_refiners(sample_ocr_result)

        # Should have timing information
        assert hasattr(results[0], 'refining_duration')
        assert results[0].refining_duration > 0

    def test_pipeline_empty_refiners(self, sample_ocr_result):
        """Test pipeline with no refiners."""
        pipeline = DocumentPipeline(refiners=[])

        results = pipeline.run_refiners(sample_ocr_result)

        assert len(results) == 0

    def test_pipeline_chunker_with_refining_result(
        self, mistral_api_key, sample_ocr_result
    ):
        """Test chunker accepts RefiningResult from refiners."""
        refiner = FootnoteRefiner(api_key=mistral_api_key)
        chunker = FixedSizeChunker(chunk_size=300, overlap=50)

        pipeline = DocumentPipeline(refiners=[refiner], chunker=chunker)

        # Run refiner first
        refining_results = pipeline.run_refiners(sample_ocr_result)

        # Then chunk the refined result
        chunking_result = pipeline.run_chunker(refining_results[-1])

        assert isinstance(chunking_result, ChunkingResult)
        assert len(chunking_result.chunks) > 0


@pytest.mark.integration
class TestDocumentPipelineFullExecution:
    """Tests for full pipeline execution with all components."""

    def test_pipeline_result_structure(self, mistral_api_key):
        """Test that pipeline result has correct structure."""
        # Create a minimal mock OCR result for testing
        pages = [
            OCRPage(
                index=0,
                markdown="# Test\n\nThis is test content for pipeline.",
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            )
        ]
        ocr_result = OCRResult(
            component_name="MistralOCRDocument",
            component_config={"model": "mistral-ocr-2512"},
            source_file_path="/test/sample.pdf",
            processed_at=datetime.now(timezone.utc),
            model="mistral-ocr-2512",
            pages=pages,
            usage_info=OCRUsageInfo(pages_processed=1, doc_size_bytes=1024),
        )

        # Create pipeline with all components
        refiners = [FootnoteRefiner(api_key=mistral_api_key)]
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        embedder = MistralEmbedder(api_key=mistral_api_key)

        pipeline = DocumentPipeline(
            refiners=refiners,
            chunker=chunker,
            embedder=embedder,
        )

        # Run individual steps to build result
        refining_results = pipeline.run_refiners(ocr_result)
        chunking_result = pipeline.run_chunker(refining_results[-1])
        embedding_result = pipeline.run_embedder(chunking_result)

        # Verify result structure
        assert len(refining_results) == 1
        assert refining_results[0].component_name == "FootnoteRefiner"
        assert chunking_result.component_name == "FixedSizeChunker"
        assert embedding_result.component_name == "MistralEmbedder"

        # Verify data flows correctly
        assert len(chunking_result.chunks) > 0
        assert len(embedding_result.chunks_with_embeddings) > 0
        assert (
            len(embedding_result.chunks_with_embeddings) ==
            len(chunking_result.chunks)
        )

    def test_pipeline_config_serialization(self, mistral_api_key):
        """Test that pipeline configuration is serializable."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        refiners = [
            FootnoteRefiner(api_key=mistral_api_key),
            ReferencesRefiner(api_key=mistral_api_key),
        ]
        chunker = FixedSizeChunker(chunk_size=500, overlap=100)
        embedder = MistralEmbedder(api_key=mistral_api_key)

        DocumentPipeline(
            ocr_processor=ocr,
            refiners=refiners,
            chunker=chunker,
            embedder=embedder,
        )

        # Get configuration from each component
        ocr_config = ocr.get_config()
        refiner_configs = [r.get_config() for r in refiners]
        chunker_config = chunker.get_config()
        embedder_config = embedder.get_config()

        # Verify all configs are dictionaries (serializable)
        assert isinstance(ocr_config, dict)
        assert all(isinstance(c, dict) for c in refiner_configs)
        assert isinstance(chunker_config, dict)
        assert isinstance(embedder_config, dict)

        # Verify configs have expected keys
        assert "model" in ocr_config
        assert "inline_explanations" in refiner_configs[0]
        assert "extract_references" in refiner_configs[1]
        assert "chunk_size" in chunker_config
        assert "model" in embedder_config

    def test_pipeline_metrics_aggregation(self, mistral_api_key):
        """Test that pipeline aggregates metrics from all steps."""
        pages = [
            OCRPage(
                index=0,
                markdown="# Test\n\nShort test content.",
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            )
        ]
        ocr_result = OCRResult(
            component_name="MistralOCRDocument",
            component_config={"model": "mistral-ocr-2512"},
            source_file_path="/test/sample.pdf",
            processed_at=datetime.now(timezone.utc),
            model="mistral-ocr-2512",
            pages=pages,
            usage_info=OCRUsageInfo(pages_processed=1, doc_size_bytes=512),
        )

        refiners = [
            FootnoteRefiner(api_key=mistral_api_key),
            ReferencesRefiner(api_key=mistral_api_key),
        ]
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        embedder = MistralEmbedder(api_key=mistral_api_key)

        pipeline = DocumentPipeline(
            refiners=refiners,
            chunker=chunker,
            embedder=embedder,
        )

        # Run pipeline steps
        refining_results = pipeline.run_refiners(ocr_result)
        chunking_result = pipeline.run_chunker(refining_results[-1])
        embedding_result = pipeline.run_embedder(chunking_result)

        # Verify each step has metrics
        assert all(r.metrics is not None for r in refining_results)
        assert embedding_result.metrics is not None

        # Verify metrics have token counts
        for result in refining_results:
            assert hasattr(result.metrics, 'total_tokens')
            assert result.metrics.total_tokens > 0

    def test_pipeline_component_names(self, mistral_api_key):
        """Test that all components report correct names."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        refiners = [
            FootnoteRefiner(api_key=mistral_api_key),
            ReferencesRefiner(api_key=mistral_api_key),
        ]
        chunker = FixedSizeChunker(chunk_size=500, overlap=100)
        embedder = MistralEmbedder(api_key=mistral_api_key)

        # Verify get_name() returns correct values
        assert ocr.get_name() == "MistralOCRDocument"
        assert refiners[0].get_name() == "FootnoteRefiner"
        assert refiners[1].get_name() == "ReferencesRefiner"
        assert chunker.get_name() == "FixedSizeChunker"
        assert embedder.get_name() == "MistralEmbedder"

        # Verify str() returns class name
        assert str(ocr) == "MistralOCRDocument"
        assert str(refiners[0]) == "FootnoteRefiner"
        assert str(chunker) == "FixedSizeChunker"
        assert str(embedder) == "MistralEmbedder"

    def test_pipeline_with_no_refiners(self, mistral_api_key):
        """Test pipeline execution without refiners."""
        pages = [
            OCRPage(
                index=0,
                markdown="# Test\n\nContent without refiners.",
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            )
        ]
        ocr_result = OCRResult(
            component_name="MistralOCRDocument",
            component_config={"model": "mistral-ocr-2512"},
            source_file_path="/test/sample.pdf",
            processed_at=datetime.now(timezone.utc),
            model="mistral-ocr-2512",
            pages=pages,
            usage_info=OCRUsageInfo(pages_processed=1, doc_size_bytes=512),
        )

        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        embedder = MistralEmbedder(api_key=mistral_api_key)

        pipeline = DocumentPipeline(
            refiners=[],
            chunker=chunker,
            embedder=embedder,
        )

        # Run without refiners
        refining_results = pipeline.run_refiners(ocr_result)
        chunking_result = pipeline.run_chunker(ocr_result)
        embedding_result = pipeline.run_embedder(chunking_result)

        assert len(refining_results) == 0
        assert len(chunking_result.chunks) > 0
        assert len(embedding_result.chunks_with_embeddings) > 0

    def test_pipeline_different_chunker_types(self, mistral_api_key):
        """Test pipeline with different chunker types."""
        pages = [
            OCRPage(
                index=0,
                markdown="""# Physics

Quantum mechanics describes atomic behavior.

# Cooking

Italian pasta is delicious.""",
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            )
        ]
        ocr_result = OCRResult(
            component_name="MistralOCRDocument",
            component_config={"model": "mistral-ocr-2512"},
            source_file_path="/test/sample.pdf",
            processed_at=datetime.now(timezone.utc),
            model="mistral-ocr-2512",
            pages=pages,
            usage_info=OCRUsageInfo(pages_processed=1, doc_size_bytes=512),
        )

        # Test with FixedSizeChunker
        fixed_chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        pipeline1 = DocumentPipeline(chunker=fixed_chunker)
        result1 = pipeline1.run_chunker(ocr_result)

        assert result1.component_name == "FixedSizeChunker"
        assert len(result1.chunks) > 0

        # Test with SemanticChunker
        semantic_chunker = SemanticChunker(
            api_key=mistral_api_key,
            min_chunk_size=50
        )
        pipeline2 = DocumentPipeline(chunker=semantic_chunker)
        result2 = pipeline2.run_chunker(ocr_result)

        assert result2.component_name == "SemanticChunker"
        assert len(result2.chunks) > 0

    def test_pipeline_embedder_model_config(self, mistral_api_key):
        """Test pipeline with different embedder model."""
        chunks = [
            Chunk(
                text="Test content",
                metadata=ChunkMetadata(page_index=0)
            )
        ]
        chunking_result = ChunkingResult(
            component_name="TestChunker",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            chunks=chunks,
        )

        embedder = MistralEmbedder(
            api_key=mistral_api_key,
            model="mistral-embed"
        )
        pipeline = DocumentPipeline(embedder=embedder)

        result = pipeline.run_embedder(chunking_result)

        assert result.component_name == "MistralEmbedder"
        assert result.model_name == "mistral-embed"
        assert "model" in result.component_config
        assert result.component_config["model"] == "mistral-embed"


@pytest.mark.integration
class TestDocumentPipelineProcess:
    """Tests for full pipeline.process() execution with real PDF."""

    def test_pipeline_process_full_execution(
        self, mistral_api_key, sample_pdf_path
    ):
        """Test full pipeline execution from PDF to embeddings."""
        # Create pipeline with all components
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        refiners = [
            FootnoteRefiner(api_key=mistral_api_key),
            ReferencesRefiner(api_key=mistral_api_key),
        ]
        chunker = FixedSizeChunker(chunk_size=500, overlap=100)
        embedder = MistralEmbedder(api_key=mistral_api_key)

        pipeline = DocumentPipeline(
            ocr_processor=ocr,
            refiners=refiners,
            chunker=chunker,
            embedder=embedder,
        )

        # Run full pipeline
        result = pipeline.process(sample_pdf_path)

        # Verify result structure
        assert result is not None
        assert result.source_file_path == sample_pdf_path
        assert result.processed_at is not None

        # Verify all steps completed
        assert result.ocr_result is not None
        assert result.refining_results is not None
        assert len(result.refining_results) == 2
        assert result.chunking_result is not None
        assert result.embedding_result is not None

        # Verify step status
        assert result.step_report.ocr == "success"
        assert result.step_report.refining == "success"
        assert result.step_report.chunking == "success"
        assert result.step_report.embedding == "success"

        # Verify pipeline config
        assert "ocr" in result.pipeline_config
        assert "refiners" in result.pipeline_config
        assert "chunker" in result.pipeline_config
        assert "embedder" in result.pipeline_config

        # Verify timings
        assert result.timings.ocr > 0
        assert result.timings.refining > 0
        assert result.timings.chunking > 0
        assert result.timings.embedding > 0
        assert result.timings.total_duration > 0

        # Verify metrics aggregation
        assert len(result.total_metrics) > 0
        assert result.total_cost_usd >= 0

        # Verify logs were captured
        assert result.logs is not None
        assert len(result.logs) > 0

    def test_pipeline_process_minimal_config(
        self, mistral_api_key, sample_pdf_path
    ):
        """Test pipeline with minimal configuration (no refiners)."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        chunker = FixedSizeChunker(chunk_size=300, overlap=50)
        embedder = MistralEmbedder(api_key=mistral_api_key)

        pipeline = DocumentPipeline(
            ocr_processor=ocr,
            refiners=[],
            chunker=chunker,
            embedder=embedder,
        )

        result = pipeline.process(sample_pdf_path)

        # Should complete successfully without refiners
        assert result.step_report.ocr == "success"
        assert result.step_report.refining == "success"
        assert result.step_report.chunking == "success"
        assert result.step_report.embedding == "success"

        # Refining results should be empty
        assert result.refining_results == []

        # Other steps should have results
        assert result.ocr_result is not None
        assert result.chunking_result is not None
        assert result.embedding_result is not None

    def test_pipeline_process_with_semantic_chunker(
        self, mistral_api_key, sample_pdf_path
    ):
        """Test pipeline with semantic chunker."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        chunker = SemanticChunker(
            api_key=mistral_api_key,
            min_chunk_size=200
        )
        embedder = MistralEmbedder(api_key=mistral_api_key)

        pipeline = DocumentPipeline(
            ocr_processor=ocr,
            chunker=chunker,
            embedder=embedder,
        )

        result = pipeline.process(sample_pdf_path)

        assert result.step_report.chunking == "success"
        assert result.chunking_result.component_name == "SemanticChunker"
        assert len(result.chunking_result.chunks) > 0

    def test_pipeline_process_error_missing_ocr(self, sample_pdf_path):
        """Test pipeline.process() raises error when OCR missing."""
        chunker = FixedSizeChunker(chunk_size=500, overlap=100)
        embedder = MistralEmbedder(api_key="test_key")

        pipeline = DocumentPipeline(
            chunker=chunker,
            embedder=embedder,
        )

        with pytest.raises(
            ValueError,
            match="ocr_processor is required for full pipeline execution"
        ):
            pipeline.process(sample_pdf_path)

    def test_pipeline_process_error_missing_chunker(
        self, mistral_api_key, sample_pdf_path
    ):
        """Test pipeline.process() raises error when chunker missing."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        embedder = MistralEmbedder(api_key=mistral_api_key)

        pipeline = DocumentPipeline(
            ocr_processor=ocr,
            embedder=embedder,
        )

        with pytest.raises(
            ValueError,
            match="chunker is required for full pipeline execution"
        ):
            pipeline.process(sample_pdf_path)

    def test_pipeline_process_error_missing_embedder(
        self, mistral_api_key, sample_pdf_path
    ):
        """Test pipeline.process() raises error when embedder missing."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        chunker = FixedSizeChunker(chunk_size=500, overlap=100)

        pipeline = DocumentPipeline(
            ocr_processor=ocr,
            chunker=chunker,
        )

        with pytest.raises(
            ValueError,
            match="embedder is required for full pipeline execution"
        ):
            pipeline.process(sample_pdf_path)

    def test_pipeline_process_result_serialization(
        self, mistral_api_key, sample_pdf_path
    ):
        """Test that pipeline result can be serialized."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        chunker = FixedSizeChunker(chunk_size=500, overlap=100)
        embedder = MistralEmbedder(api_key=mistral_api_key)

        pipeline = DocumentPipeline(
            ocr_processor=ocr,
            chunker=chunker,
            embedder=embedder,
        )

        result = pipeline.process(sample_pdf_path)

        # Verify result can be converted to dict (Pydantic model)
        result_dict = result.model_dump()

        assert isinstance(result_dict, dict)
        assert "source_file_path" in result_dict
        assert "processed_at" in result_dict
        assert "pipeline_config" in result_dict
        assert "timings" in result_dict
        assert "step_report" in result_dict

    def test_pipeline_process_component_configs(
        self, mistral_api_key, sample_pdf_path
    ):
        """Test that component configs are tracked in pipeline result."""
        ocr = MistralOCRDocument(
            api_key=mistral_api_key,
            model="mistral-ocr-2512"
        )
        refiners = [FootnoteRefiner(api_key=mistral_api_key)]
        chunker = FixedSizeChunker(chunk_size=500, overlap=100)
        embedder = MistralEmbedder(
            api_key=mistral_api_key,
            model="mistral-embed"
        )

        pipeline = DocumentPipeline(
            ocr_processor=ocr,
            refiners=refiners,
            chunker=chunker,
            embedder=embedder,
        )

        result = pipeline.process(sample_pdf_path)

        # Verify pipeline config contains component info
        assert result.pipeline_config["ocr"] == "MistralOCRDocument"
        assert "FootnoteRefiner" in result.pipeline_config["refiners"]
        assert result.pipeline_config["chunker"] == "FixedSizeChunker"
        assert result.pipeline_config["embedder"] == "MistralEmbedder"

        # Verify individual results have component configs
        ocr_config = result.ocr_result.component_config
        assert ocr_config["model"] == "mistral-ocr-2512"
        refiner_config = result.refining_results[0].component_config
        assert refiner_config["inline_explanations"]
        assert result.chunking_result.component_config["chunk_size"] == 500
        embed_config = result.embedding_result.component_config
        assert embed_config["model"] == "mistral-embed"

    def test_pipeline_process_logs_captured(
        self, mistral_api_key, sample_pdf_path
    ):
        """Test that pipeline captures logs during execution."""
        ocr = MistralOCRDocument(api_key=mistral_api_key)
        chunker = FixedSizeChunker(chunk_size=500, overlap=100)
        embedder = MistralEmbedder(api_key=mistral_api_key)

        pipeline = DocumentPipeline(
            ocr_processor=ocr,
            chunker=chunker,
            embedder=embedder,
        )

        result = pipeline.process(sample_pdf_path)

        # Verify logs field exists and is a string
        assert result.logs is not None
        assert isinstance(result.logs, str)

        # Verify logs contain the explicit message from process()
        assert "Starting full pipeline processing" in result.logs

        # Should also contain step information
        assert len(result.logs) > 0
