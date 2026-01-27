"""
Integration tests for embedders.

These tests make real API calls and require valid API keys.
Set MISTRAL_API_KEY and OPENAI_API_KEY environment variables before running.
"""

import pytest
import os
import numpy as np
from ragbandit.documents.embedders.mistral_embedder import MistralEmbedder
from ragbandit.documents.embedders.openai_embedder import OpenAIEmbedder
from ragbandit.schema import ChunkingResult, Chunk, ChunkMetadata
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
def openai_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return api_key


@pytest.fixture
def sample_chunking_result():
    """Create a sample chunking result for embedding."""
    chunks = [
        Chunk(
            text="Machine learning is a subset of artificial intelligence.",
            metadata=ChunkMetadata(page_index=0)
        ),
        Chunk(
            text="Deep learning uses neural networks with multiple layers.",
            metadata=ChunkMetadata(page_index=0)
        ),
        Chunk(
            text="Natural language processing enables computers to "
                 "understand human language.",
            metadata=ChunkMetadata(page_index=0)
        ),
        Chunk(
            text="Computer vision allows machines to interpret visual "
                 "information.",
            metadata=ChunkMetadata(page_index=0)
        ),
        Chunk(
            text="Reinforcement learning trains agents through rewards "
                 "and penalties.",
            metadata=ChunkMetadata(page_index=0)
        ),
    ]
    return ChunkingResult(
        component_name="FixedSizeChunker",
        component_config={"chunk_size": 500, "overlap": 100},
        processed_at=datetime.now(timezone.utc),
        chunks=chunks,
    )


@pytest.mark.integration
class TestMistralEmbedder:
    """Integration tests for MistralEmbedder with real API calls."""

    def test_embedder_with_default_model(
        self, mistral_api_key, sample_chunking_result
    ):
        """Test embedding with default model."""
        embedder = MistralEmbedder(api_key=mistral_api_key)
        result = embedder.embed_chunks(sample_chunking_result)

        assert result is not None
        assert result.component_name == "MistralEmbedder"
        assert "model" in result.component_config
        assert result.model_name == "mistral-embed"
        assert len(result.chunks_with_embeddings) == 5

        # Check embeddings are valid
        for chunk_with_emb in result.chunks_with_embeddings:
            assert len(chunk_with_emb.embedding) > 0
            assert chunk_with_emb.embedding_model == "mistral-embed"
            assert len(chunk_with_emb.text) > 0
            assert hasattr(chunk_with_emb.metadata, 'page_index')

    def test_embedder_with_specific_model(
        self, mistral_api_key, sample_chunking_result
    ):
        """Test embedding with specific model."""
        embedder = MistralEmbedder(
            api_key=mistral_api_key,
            model="mistral-embed"
        )
        result = embedder.embed_chunks(sample_chunking_result)

        assert result.component_name == "MistralEmbedder"
        assert result.component_config["model"] == "mistral-embed"
        assert result.model_name == "mistral-embed"
        assert len(result.chunks_with_embeddings) == 5

    def test_embedder_invalid_model_raises_error(self, mistral_api_key):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            MistralEmbedder(
                api_key=mistral_api_key,
                model="invalid-model"
            )

    def test_embedder_get_config(self, mistral_api_key):
        """Test get_config returns correct configuration."""
        embedder = MistralEmbedder(
            api_key=mistral_api_key,
            model="mistral-embed"
        )
        config = embedder.get_config()

        assert "model" in config
        assert config["model"] == "mistral-embed"

    def test_embedder_get_name(self, mistral_api_key):
        """Test get_name returns correct component name."""
        embedder = MistralEmbedder(api_key=mistral_api_key)
        assert embedder.get_name() == "MistralEmbedder"

    def test_embedder_result_has_metrics(
        self, mistral_api_key, sample_chunking_result
    ):
        """Test that embedding result includes token usage metrics."""
        embedder = MistralEmbedder(api_key=mistral_api_key)
        tracker = TokenUsageTracker()
        result = embedder.embed_chunks(sample_chunking_result, tracker)

        assert result.metrics is not None
        assert result.metrics.total_tokens > 0

    def test_embedder_embedding_dimensions(
        self, mistral_api_key, sample_chunking_result
    ):
        """Test that embeddings have consistent dimensions."""
        embedder = MistralEmbedder(api_key=mistral_api_key)
        result = embedder.embed_chunks(sample_chunking_result)

        # All embeddings should have the same dimension
        dimensions = [
            len(chunk.embedding)
            for chunk in result.chunks_with_embeddings
        ]
        assert len(set(dimensions)) == 1
        assert dimensions[0] > 0

    def test_embedder_cosine_similarity(self, mistral_api_key):
        """Test cosine similarity calculation."""
        embedder = MistralEmbedder(api_key=mistral_api_key)

        # Create two similar chunks
        chunks = [
            Chunk(
                text="Machine learning is a subset of AI.",
                metadata=ChunkMetadata(page_index=0)
            ),
            Chunk(
                text="Artificial intelligence includes machine learning.",
                metadata=ChunkMetadata(page_index=0)
            ),
        ]
        chunking_result = ChunkingResult(
            component_name="TestChunker",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            chunks=chunks,
        )

        result = embedder.embed_chunks(chunking_result)

        # Calculate similarity between the two embeddings
        emb1 = np.array(result.chunks_with_embeddings[0].embedding)
        emb2 = np.array(result.chunks_with_embeddings[1].embedding)

        similarity = embedder.cosine_similarity(emb1, emb2)

        # Similar texts should have high similarity
        assert 0 <= similarity <= 1
        assert similarity > 0.5

    def test_embedder_cosine_distance(self, mistral_api_key):
        """Test cosine distance calculation."""
        embedder = MistralEmbedder(api_key=mistral_api_key)

        chunks = [
            Chunk(
                text="Machine learning algorithms.",
                metadata=ChunkMetadata(page_index=0)
            ),
            Chunk(
                text="Cooking recipes for dinner.",
                metadata=ChunkMetadata(page_index=0)
            ),
        ]
        chunking_result = ChunkingResult(
            component_name="TestChunker",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            chunks=chunks,
        )

        result = embedder.embed_chunks(chunking_result)

        emb1 = np.array(result.chunks_with_embeddings[0].embedding)
        emb2 = np.array(result.chunks_with_embeddings[1].embedding)

        distance = embedder.cosine_distance(emb1, emb2)

        # Distance should be between 0 and 2
        assert 0 <= distance <= 2

    def test_embedder_empty_chunks(self, mistral_api_key):
        """Test embedder handles empty chunk list."""
        embedder = MistralEmbedder(api_key=mistral_api_key)

        empty_result = ChunkingResult(
            component_name="TestChunker",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            chunks=[],
        )

        result = embedder.embed_chunks(empty_result)

        assert result is not None
        assert len(result.chunks_with_embeddings) == 0

    def test_embedder_single_chunk(self, mistral_api_key):
        """Test embedder with single chunk."""
        embedder = MistralEmbedder(api_key=mistral_api_key)

        single_chunk_result = ChunkingResult(
            component_name="TestChunker",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            chunks=[
                Chunk(
                    text="Single chunk of text.",
                    metadata=ChunkMetadata(page_index=0)
                )
            ],
        )

        result = embedder.embed_chunks(single_chunk_result)

        assert result is not None
        assert len(result.chunks_with_embeddings) == 1
        assert len(result.chunks_with_embeddings[0].embedding) > 0

    def test_embedder_preserves_metadata(
        self, mistral_api_key, sample_chunking_result
    ):
        """Test that embedder preserves chunk metadata."""
        embedder = MistralEmbedder(api_key=mistral_api_key)
        result = embedder.embed_chunks(sample_chunking_result)

        # Verify metadata is preserved
        for i, chunk_with_emb in enumerate(result.chunks_with_embeddings):
            original_chunk = sample_chunking_result.chunks[i]
            assert chunk_with_emb.text == original_chunk.text
            assert (
                chunk_with_emb.metadata.page_index
                == original_chunk.metadata.page_index
            )


@pytest.mark.integration
class TestOpenAIEmbedder:
    """Integration tests for OpenAIEmbedder with real API calls."""

    def test_embedder_with_default_model(
        self, openai_api_key, sample_chunking_result
    ):
        """Test embedding with default model (text-embedding-3-small)."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)
        result = embedder.embed_chunks(sample_chunking_result)

        assert result is not None
        assert result.component_name == "OpenAIEmbedder"
        assert "model" in result.component_config
        assert result.model_name == "text-embedding-3-small"
        assert len(result.chunks_with_embeddings) == 5

        # Check embeddings are valid
        for chunk_with_emb in result.chunks_with_embeddings:
            assert len(chunk_with_emb.embedding) > 0
            assert chunk_with_emb.embedding_model == "text-embedding-3-small"
            assert len(chunk_with_emb.text) > 0
            assert hasattr(chunk_with_emb.metadata, 'page_index')

    def test_embedder_with_large_model(
        self, openai_api_key, sample_chunking_result
    ):
        """Test embedding with text-embedding-3-large model."""
        embedder = OpenAIEmbedder(
            api_key=openai_api_key,
            model="text-embedding-3-large"
        )
        result = embedder.embed_chunks(sample_chunking_result)

        assert result.component_name == "OpenAIEmbedder"
        assert result.component_config["model"] == "text-embedding-3-large"
        assert result.model_name == "text-embedding-3-large"
        assert len(result.chunks_with_embeddings) == 5

    def test_embedder_invalid_model_raises_error(self, openai_api_key):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            OpenAIEmbedder(
                api_key=openai_api_key,
                model="invalid-model"
            )

    def test_embedder_get_config(self, openai_api_key):
        """Test get_config returns correct configuration."""
        embedder = OpenAIEmbedder(
            api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        config = embedder.get_config()

        assert "model" in config
        assert config["model"] == "text-embedding-3-small"

    def test_embedder_get_name(self, openai_api_key):
        """Test get_name returns correct component name."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)
        assert embedder.get_name() == "OpenAIEmbedder"

    def test_embedder_result_has_metrics(
        self, openai_api_key, sample_chunking_result
    ):
        """Test that embedding result includes token usage metrics."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)
        tracker = TokenUsageTracker()
        result = embedder.embed_chunks(sample_chunking_result, tracker)

        assert result.metrics is not None
        assert result.metrics.total_tokens > 0

    def test_embedder_embedding_dimensions(
        self, openai_api_key, sample_chunking_result
    ):
        """Test that embeddings have consistent dimensions."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)
        result = embedder.embed_chunks(sample_chunking_result)

        # All embeddings should have the same dimension
        dimensions = [
            len(chunk.embedding)
            for chunk in result.chunks_with_embeddings
        ]
        assert len(set(dimensions)) == 1
        assert dimensions[0] > 0

    def test_embedder_cosine_similarity(self, openai_api_key):
        """Test cosine similarity calculation."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)

        # Create two similar chunks
        chunks = [
            Chunk(
                text="Machine learning is a subset of AI.",
                metadata=ChunkMetadata(page_index=0)
            ),
            Chunk(
                text="Artificial intelligence includes machine learning.",
                metadata=ChunkMetadata(page_index=0)
            ),
        ]
        chunking_result = ChunkingResult(
            component_name="TestChunker",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            chunks=chunks,
        )

        result = embedder.embed_chunks(chunking_result)

        # Calculate similarity between the two embeddings
        emb1 = np.array(result.chunks_with_embeddings[0].embedding)
        emb2 = np.array(result.chunks_with_embeddings[1].embedding)

        similarity = embedder.cosine_similarity(emb1, emb2)

        # Similar texts should have high similarity
        assert 0 <= similarity <= 1
        assert similarity > 0.5

    def test_embedder_cosine_distance(self, openai_api_key):
        """Test cosine distance calculation."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)

        chunks = [
            Chunk(
                text="Machine learning algorithms.",
                metadata=ChunkMetadata(page_index=0)
            ),
            Chunk(
                text="Cooking recipes for dinner.",
                metadata=ChunkMetadata(page_index=0)
            ),
        ]
        chunking_result = ChunkingResult(
            component_name="TestChunker",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            chunks=chunks,
        )

        result = embedder.embed_chunks(chunking_result)

        emb1 = np.array(result.chunks_with_embeddings[0].embedding)
        emb2 = np.array(result.chunks_with_embeddings[1].embedding)

        distance = embedder.cosine_distance(emb1, emb2)

        # Distance should be between 0 and 2
        assert 0 <= distance <= 2

    def test_embedder_empty_chunks(self, openai_api_key):
        """Test embedder handles empty chunk list."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)

        empty_result = ChunkingResult(
            component_name="TestChunker",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            chunks=[],
        )

        result = embedder.embed_chunks(empty_result)

        assert result is not None
        assert len(result.chunks_with_embeddings) == 0

    def test_embedder_single_chunk(self, openai_api_key):
        """Test embedder with single chunk."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)

        single_chunk_result = ChunkingResult(
            component_name="TestChunker",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            chunks=[
                Chunk(
                    text="Single chunk of text.",
                    metadata=ChunkMetadata(page_index=0)
                )
            ],
        )

        result = embedder.embed_chunks(single_chunk_result)

        assert result is not None
        assert len(result.chunks_with_embeddings) == 1
        assert len(result.chunks_with_embeddings[0].embedding) > 0

    def test_embedder_preserves_metadata(
        self, openai_api_key, sample_chunking_result
    ):
        """Test that embedder preserves chunk metadata."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)
        result = embedder.embed_chunks(sample_chunking_result)

        # Verify metadata is preserved
        for i, chunk_with_emb in enumerate(result.chunks_with_embeddings):
            original_chunk = sample_chunking_result.chunks[i]
            assert chunk_with_emb.text == original_chunk.text
            assert (
                chunk_with_emb.metadata.page_index
                == original_chunk.metadata.page_index
            )

    def test_behavior_embedding_values_are_normalized(
        self, openai_api_key, sample_chunking_result
    ):
        """Behavior test: Verify embeddings are normalized vectors."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)
        result = embedder.embed_chunks(sample_chunking_result)

        # OpenAI embeddings should be normalized (L2 norm ≈ 1)
        for chunk_with_emb in result.chunks_with_embeddings:
            embedding = np.array(chunk_with_emb.embedding)
            norm = np.linalg.norm(embedding)
            # Check that norm is close to 1 (normalized vector)
            assert 0.99 <= norm <= 1.01, (
                f"Embedding not normalized: norm={norm}"
            )

    def test_behavior_embedding_dimensions_correct(
        self, openai_api_key, sample_chunking_result
    ):
        """Behavior test: Verify embedding dimensions match model specs."""
        # text-embedding-3-small should have 1536 dimensions
        embedder = OpenAIEmbedder(
            api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        result = embedder.embed_chunks(sample_chunking_result)

        for chunk_with_emb in result.chunks_with_embeddings:
            assert len(chunk_with_emb.embedding) == 1536, (
                f"Expected 1536 dimensions for text-embedding-3-small, "
                f"got {len(chunk_with_emb.embedding)}"
            )

    def test_behavior_embedding_values_reasonable_range(
        self, openai_api_key, sample_chunking_result
    ):
        """Behavior test: Verify embedding values are in reasonable range."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)
        result = embedder.embed_chunks(sample_chunking_result)

        # Check that embedding values are reasonable (typically -1 to 1)
        for chunk_with_emb in result.chunks_with_embeddings:
            embedding = np.array(chunk_with_emb.embedding)
            min_val = embedding.min()
            max_val = embedding.max()

            # Values should be in a reasonable range for normalized vectors
            assert -1.5 <= min_val <= 1.5, (
                f"Embedding min value out of range: {min_val}"
            )
            assert -1.5 <= max_val <= 1.5, (
                f"Embedding max value out of range: {max_val}"
            )

    def test_behavior_similar_texts_high_similarity(self, openai_api_key):
        """Behavior test: Similar texts should have high cosine similarity."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)

        # Create very similar chunks
        chunks = [
            Chunk(
                text="The quick brown fox jumps over the lazy dog.",
                metadata=ChunkMetadata(page_index=0)
            ),
            Chunk(
                text="A quick brown fox jumps over a lazy dog.",
                metadata=ChunkMetadata(page_index=0)
            ),
        ]
        chunking_result = ChunkingResult(
            component_name="TestChunker",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            chunks=chunks,
        )

        result = embedder.embed_chunks(chunking_result)

        emb1 = np.array(result.chunks_with_embeddings[0].embedding)
        emb2 = np.array(result.chunks_with_embeddings[1].embedding)

        similarity = embedder.cosine_similarity(emb1, emb2)

        # Very similar texts should have very high similarity (>0.9)
        assert similarity > 0.9, (
            f"Similar texts should have high similarity, got {similarity}"
        )

    def test_behavior_dissimilar_texts_low_similarity(self, openai_api_key):
        """Behavior test: Dissimilar texts should have low similarity."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)

        # Create completely unrelated chunks
        chunks = [
            Chunk(
                text="Quantum mechanics describes subatomic particles.",
                metadata=ChunkMetadata(page_index=0)
            ),
            Chunk(
                text="Chocolate cake recipe with vanilla frosting.",
                metadata=ChunkMetadata(page_index=0)
            ),
        ]
        chunking_result = ChunkingResult(
            component_name="TestChunker",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            chunks=chunks,
        )

        result = embedder.embed_chunks(chunking_result)

        emb1 = np.array(result.chunks_with_embeddings[0].embedding)
        emb2 = np.array(result.chunks_with_embeddings[1].embedding)

        similarity = embedder.cosine_similarity(emb1, emb2)

        # Dissimilar texts should have lower similarity (<0.5)
        assert similarity < 0.5, (
            f"Dissimilar texts should have low similarity, got {similarity}"
        )

    def test_behavior_processed_timestamp_set(
        self, openai_api_key, sample_chunking_result
    ):
        """Behavior test: Verify processed_at timestamp is set correctly."""
        before = datetime.now(timezone.utc)
        embedder = OpenAIEmbedder(api_key=openai_api_key)
        result = embedder.embed_chunks(sample_chunking_result)
        after = datetime.now(timezone.utc)

        # Check timestamp is within reasonable range
        assert result.processed_at is not None
        assert before <= result.processed_at <= after

    def test_behavior_token_usage_tracking(
        self, openai_api_key, sample_chunking_result
    ):
        """Behavior test: Verify token usage is tracked correctly."""
        embedder = OpenAIEmbedder(api_key=openai_api_key)
        tracker = TokenUsageTracker()

        result = embedder.embed_chunks(sample_chunking_result, tracker)

        # Check that tokens were tracked
        assert result.metrics is not None
        assert result.metrics.total_tokens > 0
        # For embeddings, embedding tokens should be > 0
        assert result.metrics.total_embedding_tokens > 0
        # Input/output tokens should be 0 for embeddings
        assert result.metrics.total_input_tokens == 0
        assert result.metrics.total_output_tokens == 0
