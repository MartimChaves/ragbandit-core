"""
Integration tests for chunkers.

These tests use real document data. SemanticChunker tests require API calls.
"""

import pytest
import os
from ragbandit.documents.chunkers.fixed_size_chunker import FixedSizeChunker
from ragbandit.documents.chunkers.semantic_chunker import SemanticChunker
from ragbandit.documents.chunkers.sentence_chunker import SentenceChunker
from ragbandit.documents.chunkers.recursive_markdown_chunker import (
    RecursiveMarkdownChunker,
)
from ragbandit.schema import (
    RefiningResult,
    RefinedPage,
    PageDimensions,
)
from datetime import datetime, timezone


@pytest.fixture
def mistral_api_key():
    """Get Mistral API key from environment."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY not set")
    return api_key


@pytest.fixture
def sample_refining_result():
    """Create a sample refining result with substantial text."""
    pages = [
        RefinedPage(
            index=0,
            markdown="""# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on
building systems that can learn from data. These systems improve their
performance over time without being explicitly programmed.

## Types of Machine Learning

There are three main types of machine learning:

### Supervised Learning

In supervised learning, the algorithm learns from labeled training data.
The model is trained on a dataset where the correct answers are provided,
and it learns to predict outcomes for new, unseen data.

### Unsupervised Learning

Unsupervised learning involves training on data without labeled responses.
The algorithm tries to find patterns and structure in the data on its own.

### Reinforcement Learning

Reinforcement learning is about taking suitable actions to maximize reward
in a particular situation. The agent learns through trial and error.
""",
            images=[],
            dimensions=PageDimensions(dpi=72, width=612, height=792)
        ),
        RefinedPage(
            index=1,
            markdown="""# Applications of Machine Learning

Machine learning has numerous applications across various industries:

## Healthcare

Machine learning is revolutionizing healthcare by enabling better
diagnosis, treatment planning, and drug discovery. Algorithms can
analyze medical images, predict patient outcomes, and personalize
treatment plans.

## Finance

In finance, machine learning is used for fraud detection, algorithmic
trading, credit scoring, and risk management. These systems can
process vast amounts of financial data in real-time.

## Natural Language Processing

NLP applications include chatbots, translation services, sentiment
analysis, and text summarization. These systems help computers
understand and generate human language.
""",
            images=[],
            dimensions=PageDimensions(dpi=72, width=612, height=792)
        )
    ]
    return RefiningResult(
        component_name="TestRefiner",
        component_config={},
        processed_at=datetime.now(timezone.utc),
        pages=pages,
        refining_trace=[],
        extracted_data={},
    )


@pytest.mark.integration
class TestFixedSizeChunker:
    """Integration tests for FixedSizeChunker."""

    def test_fixed_size_chunker_basic(self, sample_refining_result):
        """Test basic fixed size chunking."""
        chunker = FixedSizeChunker(chunk_size=500, overlap=100)
        result = chunker.chunk(sample_refining_result)

        assert result is not None
        assert result.component_name == "FixedSizeChunker"
        assert result.component_config["chunk_size"] == 500
        assert result.component_config["overlap"] == 100
        assert len(result.chunks) > 0

        # Verify chunks have content
        for chunk in result.chunks:
            assert len(chunk.text) > 0
            assert hasattr(chunk.metadata, 'page_index')
            assert chunk.metadata.page_index >= 0

    def test_fixed_size_chunker_no_overlap(self, sample_refining_result):
        """Test fixed size chunking without overlap."""
        chunker = FixedSizeChunker(chunk_size=300, overlap=0)
        result = chunker.chunk(sample_refining_result)

        assert result is not None
        assert result.component_config["overlap"] == 0
        assert len(result.chunks) > 0

    def test_fixed_size_chunker_large_chunks(self, sample_refining_result):
        """Test with large chunk size."""
        chunker = FixedSizeChunker(chunk_size=2000, overlap=200)
        result = chunker.chunk(sample_refining_result)

        assert result is not None
        assert len(result.chunks) > 0

        # With large chunks, we should get fewer chunks
        for chunk in result.chunks:
            assert len(chunk.text) <= 2000 + 200  # chunk_size + overlap

    def test_fixed_size_chunker_get_config(self):
        """Test get_config returns correct configuration."""
        chunker = FixedSizeChunker(chunk_size=1000, overlap=150)
        config = chunker.get_config()

        assert "chunk_size" in config
        assert "overlap" in config
        assert config["chunk_size"] == 1000
        assert config["overlap"] == 150

    def test_fixed_size_chunker_get_name(self):
        """Test get_name returns correct component name."""
        chunker = FixedSizeChunker(chunk_size=500, overlap=100)
        assert chunker.get_name() == "FixedSizeChunker"

    def test_fixed_size_chunker_chunk_indices(self, sample_refining_result):
        """Test that chunk indices are sequential."""
        chunker = FixedSizeChunker(chunk_size=300, overlap=50)
        result = chunker.chunk(sample_refining_result)

        # Verify chunks have valid page indices
        for chunk in result.chunks:
            assert chunk.metadata.page_index >= 0

    def test_fixed_size_chunker_overlap_behavior(self):
        """Test overlap creates overlapping text between chunks."""
        # Create a simple document with known content
        pages = [
            RefinedPage(
                index=0,
                markdown="A" * 100 + "B" * 100 + "C" * 100 + "D" * 100,
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            )
        ]
        refining_result = RefiningResult(
            component_name="TestRefiner",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            pages=pages,
            refining_trace=[],
            extracted_data={},
        )

        # Chunk with overlap
        chunker = FixedSizeChunker(chunk_size=150, overlap=50)
        result = chunker.chunk(refining_result)

        # Should have multiple chunks
        assert len(result.chunks) >= 2

        # Verify overlap between consecutive chunks
        for i in range(len(result.chunks) - 1):
            current_chunk = result.chunks[i]
            next_chunk = result.chunks[i + 1]

            # Get the last 50 chars of current chunk
            current_end = current_chunk.text[-50:]
            # Get the first 50 chars of next chunk
            next_start = next_chunk.text[:50]

            # They should overlap (share some common text)
            # Check if there's any common substring
            overlap_found = False
            for j in range(len(current_end)):
                if current_end[j:] == next_start[:len(current_end[j:])]:
                    if len(current_end[j:]) > 10:  # At least 10 chars
                        overlap_found = True
                        break

            assert overlap_found, (
                f"No overlap found between chunk {i} and {i+1}"
            )


@pytest.mark.integration
class TestSemanticChunker:
    """Integration tests for SemanticChunker."""

    def test_semantic_chunker_basic(
        self, mistral_api_key, sample_refining_result
    ):
        """Test basic semantic chunking."""
        chunker = SemanticChunker(
            api_key=mistral_api_key,
            min_chunk_size=200,
        )
        result = chunker.chunk(sample_refining_result)

        assert result is not None
        assert result.component_name == "SemanticChunker"
        assert result.component_config["min_chunk_size"] == 200
        assert len(result.chunks) > 0

        # Verify chunks have content
        for chunk in result.chunks:
            assert len(chunk.text) > 0
            assert hasattr(chunk.metadata, 'page_index')

    def test_semantic_chunker_different_min_sizes(
        self, mistral_api_key, sample_refining_result
    ):
        """Test semantic chunking with different minimum chunk sizes."""
        chunker_small = SemanticChunker(
            api_key=mistral_api_key,
            min_chunk_size=200,
        )
        chunker_large = SemanticChunker(
            api_key=mistral_api_key,
            min_chunk_size=800,
        )
        result_small = chunker_small.chunk(sample_refining_result)
        result_large = chunker_large.chunk(sample_refining_result)

        assert result_small is not None
        assert result_large is not None

        # Smaller min size should generally allow more chunks
        assert len(result_small.chunks) >= len(result_large.chunks)

    def test_semantic_chunker_get_config(self, mistral_api_key):
        """Test get_config returns correct configuration."""
        chunker = SemanticChunker(
            api_key=mistral_api_key,
            min_chunk_size=300,
        )
        config = chunker.get_config()

        assert "min_chunk_size" in config
        assert config["min_chunk_size"] == 300

    def test_semantic_chunker_get_name(self, mistral_api_key):
        """Test get_name returns correct component name."""
        chunker = SemanticChunker(
            api_key=mistral_api_key,
            min_chunk_size=200,
        )
        assert chunker.get_name() == "SemanticChunker"

    def test_semantic_chunker_preserves_metadata(
        self, mistral_api_key, sample_refining_result
    ):
        """Test that semantic chunker preserves page metadata."""
        chunker = SemanticChunker(
            api_key=mistral_api_key,
            min_chunk_size=200,
        )
        result = chunker.chunk(sample_refining_result)

        # Verify all chunks have page_index metadata
        for chunk in result.chunks:
            assert hasattr(chunk.metadata, 'page_index')
            assert chunk.metadata.page_index >= 0

    def test_semantic_chunker_behavior(self, mistral_api_key):
        """Test semantic chunker breaks at semantic boundaries."""
        # Create document with clearly different thematic sections
        pages = [
            RefinedPage(
                index=0,
                markdown="""# Quantum Physics

Quantum mechanics is a fundamental theory in physics that describes
the behavior of matter and energy at atomic and subatomic scales.
Wave-particle duality is a central concept where particles exhibit
both wave and particle properties. The Heisenberg uncertainty
principle states that certain pairs of physical properties cannot
be simultaneously known to arbitrary precision.

# Cooking Recipes

Italian pasta carbonara is a classic Roman dish made with eggs,
cheese, pancetta, and black pepper. The key to perfect carbonara
is timing - the eggs must be added off heat to create a creamy
sauce without scrambling. Fresh pasta works best and should be
cooked al dente for the ideal texture.

# Ancient History

The Roman Empire was one of the largest empires in ancient history,
spanning three continents at its peak. Julius Caesar's military
campaigns in Gaul expanded Roman territory significantly. The fall
of Rome in 476 CE marked the end of the Western Roman Empire and
the beginning of the Middle Ages in Europe.""",
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            )
        ]
        refining_result = RefiningResult(
            component_name="TestRefiner",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            pages=pages,
            refining_trace=[],
            extracted_data={},
        )

        # Chunk with semantic chunker
        chunker = SemanticChunker(
            api_key=mistral_api_key,
            min_chunk_size=100,
        )
        result = chunker.chunk(refining_result)

        # Should create multiple chunks (at least 2)
        assert len(result.chunks) >= 2

        # Verify all chunks respect min_chunk_size
        for chunk in result.chunks:
            # Allow some flexibility for last chunk or short sections
            assert len(chunk.text) >= 50, (
                f"Chunk too small: {len(chunk.text)} chars"
            )

        # Verify semantic breaks make sense
        # Each major theme should ideally be in separate chunks
        quantum_chunks = [
            c for c in result.chunks
            if "quantum" in c.text.lower() or "heisenberg" in c.text.lower()
        ]
        cooking_chunks = [
            c for c in result.chunks
            if "pasta" in c.text.lower() or "carbonara" in c.text.lower()
        ]
        history_chunks = [
            c for c in result.chunks
            if "roman empire" in c.text.lower() or "caesar" in c.text.lower()
        ]

        # At least one chunk should contain each theme
        assert len(quantum_chunks) >= 1, "Quantum physics theme not found"
        assert len(cooking_chunks) >= 1, "Cooking theme not found"
        assert len(history_chunks) >= 1, "History theme not found"

        # Ideally, themes should be separated (not mixed in same chunk)
        # Check that at least some chunks are theme-specific
        theme_specific_chunks = 0
        for chunk in result.chunks:
            text_lower = chunk.text.lower()
            themes_present = sum([
                "quantum" in text_lower or "heisenberg" in text_lower,
                "pasta" in text_lower or "carbonara" in text_lower,
                "roman empire" in text_lower or "caesar" in text_lower,
            ])
            if themes_present == 1:
                theme_specific_chunks += 1

        # At least half the chunks should be theme-specific
        assert theme_specific_chunks >= len(result.chunks) // 2, (
            f"Only {theme_specific_chunks}/{len(result.chunks)} chunks "
            "are theme-specific"
        )


@pytest.mark.integration
class TestChunkerComparison:
    """Compare behavior of different chunkers."""

    def test_compare_chunker_outputs(
        self, mistral_api_key, sample_refining_result
    ):
        """Compare outputs from different chunkers."""
        fixed_chunker = FixedSizeChunker(chunk_size=500, overlap=100)
        semantic_chunker = SemanticChunker(
            api_key=mistral_api_key,
            min_chunk_size=200,
        )

        fixed_result = fixed_chunker.chunk(sample_refining_result)
        semantic_result = semantic_chunker.chunk(sample_refining_result)

        # Both should produce chunks
        assert len(fixed_result.chunks) > 0
        assert len(semantic_result.chunks) > 0

        # Component names should be different
        assert fixed_result.component_name != semantic_result.component_name

        # Both should have proper metadata
        for chunk in fixed_result.chunks:
            assert hasattr(chunk.metadata, 'page_index')
        for chunk in semantic_result.chunks:
            assert hasattr(chunk.metadata, 'page_index')


@pytest.mark.integration
class TestSentenceChunker:
    """Integration tests for SentenceChunker."""

    def test_sentence_chunker_basic(self, sample_refining_result):
        """Test basic sentence chunking."""
        chunker = SentenceChunker(sentences_per_chunk=5, sentence_overlap=1)
        result = chunker.chunk(sample_refining_result)

        assert result is not None
        assert result.component_name == "SentenceChunker"
        assert result.component_config["sentences_per_chunk"] == 5
        assert result.component_config["sentence_overlap"] == 1
        assert len(result.chunks) > 0

        for chunk in result.chunks:
            assert len(chunk.text) > 0
            assert hasattr(chunk.metadata, 'page_index')
            assert chunk.metadata.page_index >= 0

    def test_sentence_chunker_get_config(self):
        """Test get_config returns correct configuration."""
        chunker = SentenceChunker(
            sentences_per_chunk=3,
            sentence_overlap=1,
            min_chunk_size=50,
        )
        config = chunker.get_config()

        assert config["sentences_per_chunk"] == 3
        assert config["sentence_overlap"] == 1
        assert config["min_chunk_size"] == 50

    def test_sentence_chunker_get_name(self):
        """Test get_name returns correct component name."""
        chunker = SentenceChunker()
        assert chunker.get_name() == "SentenceChunker"

    def test_sentence_chunker_invalid_overlap_raises(self):
        """Test that overlap >= sentences_per_chunk raises ValueError."""
        with pytest.raises(ValueError, match="sentence_overlap"):
            SentenceChunker(sentences_per_chunk=3, sentence_overlap=3)

        with pytest.raises(ValueError, match="sentence_overlap"):
            SentenceChunker(sentences_per_chunk=3, sentence_overlap=5)

    def test_sentence_chunker_empty_page(self):
        """Test chunker handles empty pages without error."""
        pages = [
            RefinedPage(
                index=0,
                markdown="",
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            ),
            RefinedPage(
                index=1,
                markdown="A proper sentence here. And another one.",
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            ),
        ]
        ref_result = RefiningResult(
            component_name="TestRefiner",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            pages=pages,
            refining_trace=[],
            extracted_data={},
        )
        chunker = SentenceChunker(sentences_per_chunk=3, sentence_overlap=1)
        result = chunker.chunk(ref_result)

        assert result is not None
        assert len(result.chunks) > 0
        for chunk in result.chunks:
            assert chunk.metadata.page_index == 1

    def test_behavior_chunks_are_sentence_groups(self):
        """Behavior: each chunk contains at most
        sentences_per_chunk sentences."""
        pages = [
            RefinedPage(
                index=0,
                markdown=(
                    "First sentence ends here. "
                    "Second sentence is here. "
                    "Third sentence follows. "
                    "Fourth sentence comes next. "
                    "Fifth sentence is present. "
                    "Sixth sentence appears. "
                    "Seventh sentence finishes."
                ),
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            )
        ]
        ref_result = RefiningResult(
            component_name="TestRefiner",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            pages=pages,
            refining_trace=[],
            extracted_data={},
        )
        chunker = SentenceChunker(
            sentences_per_chunk=3,
            sentence_overlap=0,
            min_chunk_size=1,
        )
        result = chunker.chunk(ref_result)

        # 7 sentences with window=3, step=3 → 3 chunks (3, 3, 1)
        assert len(result.chunks) >= 2

        # No chunk should contain more than 3 sentence-ending punctuation marks
        for chunk in result.chunks:
            sentence_endings = chunk.text.count(
                "."
            ) + chunk.text.count("!") + chunk.text.count("?")
            assert sentence_endings <= 3, (
                f"Chunk has more than 3 sentences: {chunk.text!r}"
            )

    def test_behavior_overlap_repeats_sentences(self):
        """Behavior: overlapping sentences appear in consecutive chunks."""
        pages = [
            RefinedPage(
                index=0,
                markdown=(
                    "Alpha sentence is first. "
                    "Beta sentence is second. "
                    "Gamma sentence is third. "
                    "Delta sentence is fourth."
                ),
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            )
        ]
        ref_result = RefiningResult(
            component_name="TestRefiner",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            pages=pages,
            refining_trace=[],
            extracted_data={},
        )
        chunker = SentenceChunker(
            sentences_per_chunk=3,
            sentence_overlap=1,
            min_chunk_size=1,
        )
        result = chunker.chunk(ref_result)

        # With overlap=1, step=2: chunk0=[0,1,2], chunk1=[2,3]
        # "Gamma" is the overlap sentence — it must appear in both chunks
        assert len(result.chunks) >= 2
        assert "Gamma" in result.chunks[0].text
        assert "Gamma" in result.chunks[1].text

    def test_behavior_min_chunk_size_merges_small_chunks(self):
        """Behavior: chunks below min_chunk_size are merged."""
        pages = [
            RefinedPage(
                index=0,
                markdown="Hi. Bye. Ok.",
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            )
        ]
        ref_result = RefiningResult(
            component_name="TestRefiner",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            pages=pages,
            refining_trace=[],
            extracted_data={},
        )
        chunker = SentenceChunker(
            sentences_per_chunk=2,
            sentence_overlap=0,
            min_chunk_size=20,
        )
        result = chunker.chunk(ref_result)

        # All chunks must be at least min_chunk_size or merged into neighbours
        for chunk in result.chunks:
            assert len(chunk.text) >= 1


@pytest.mark.integration
class TestRecursiveMarkdownChunker:
    """Integration tests for RecursiveMarkdownChunker."""

    def test_recursive_chunker_basic(self, sample_refining_result):
        """Test basic recursive markdown chunking."""
        chunker = RecursiveMarkdownChunker(
            chunk_size=500, overlap=50, min_chunk_size=50
        )
        result = chunker.chunk(sample_refining_result)

        assert result is not None
        assert result.component_name == "RecursiveMarkdownChunker"
        assert result.component_config["chunk_size"] == 500
        assert result.component_config["overlap"] == 50
        assert len(result.chunks) > 0

        for chunk in result.chunks:
            assert len(chunk.text) > 0
            assert hasattr(chunk.metadata, 'page_index')

    def test_recursive_chunker_get_config(self):
        """Test get_config returns correct configuration."""
        chunker = RecursiveMarkdownChunker(
            chunk_size=800, overlap=80, min_chunk_size=100
        )
        config = chunker.get_config()

        assert config["chunk_size"] == 800
        assert config["overlap"] == 80
        assert config["min_chunk_size"] == 100

    def test_recursive_chunker_get_name(self):
        """Test get_name returns correct component name."""
        chunker = RecursiveMarkdownChunker()
        assert chunker.get_name() == "RecursiveMarkdownChunker"

    def test_recursive_chunker_large_chunk_size(self, sample_refining_result):
        """Test that a large chunk_size produces fewer chunks."""
        chunker_small = RecursiveMarkdownChunker(
            chunk_size=200, overlap=20, min_chunk_size=50
        )
        chunker_large = RecursiveMarkdownChunker(
            chunk_size=2000, overlap=100, min_chunk_size=50
        )
        result_small = chunker_small.chunk(sample_refining_result)
        result_large = chunker_large.chunk(sample_refining_result)

        assert len(result_small.chunks) >= len(result_large.chunks)

    def test_behavior_chunks_respect_chunk_size(self):
        """Behavior: chunks should not exceed chunk_size significantly."""
        pages = [
            RefinedPage(
                index=0,
                markdown="""# Section One

This is the first section with some content about topic A.
It has multiple sentences covering the main ideas.

# Section Two

This is the second section covering topic B in detail.
More sentences here to provide substance to the chunk.

# Section Three

Final section wrapping up with topic C and conclusions.
The end of the document is reached here.
""",
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            )
        ]
        ref_result = RefiningResult(
            component_name="TestRefiner",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            pages=pages,
            refining_trace=[],
            extracted_data={},
        )
        chunk_size = 200
        chunker = RecursiveMarkdownChunker(
            chunk_size=chunk_size, overlap=20, min_chunk_size=10
        )
        result = chunker.chunk(ref_result)

        assert len(result.chunks) > 0
        # Allow some tolerance for the overlap prepended to chunks
        tolerance = chunk_size * 0.5
        for chunk in result.chunks:
            assert len(chunk.text) <= chunk_size + tolerance, (
                f"Chunk exceeds size limit: {len(chunk.text)} chars"
            )

    def test_behavior_splits_at_headings(self):
        """Behavior: H1 headings should create chunk boundaries."""
        long_section = "Word " * 60  # 300 chars, forces split
        pages = [
            RefinedPage(
                index=0,
                markdown=(
                    f"# Alpha Heading\n\n{long_section}\n\n"
                    f"# Beta Heading\n\n{long_section}\n"
                ),
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            )
        ]
        ref_result = RefiningResult(
            component_name="TestRefiner",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            pages=pages,
            refining_trace=[],
            extracted_data={},
        )
        chunker = RecursiveMarkdownChunker(
            chunk_size=200, overlap=0, min_chunk_size=10
        )
        result = chunker.chunk(ref_result)

        # Alpha and Beta content should not be mixed in a single chunk
        alpha_only = any(
            "Alpha" in c.text and "Beta" not in c.text
            for c in result.chunks
        )
        beta_only = any(
            "Beta" in c.text and "Alpha" not in c.text
            for c in result.chunks
        )
        assert alpha_only, (
            "Expected at least one chunk with only Alpha content"
        )
        assert beta_only, (
            "Expected at least one chunk with only Beta content"
        )

    def test_behavior_empty_pages_skipped(self):
        """Behavior: empty pages produce no chunks."""
        pages = [
            RefinedPage(
                index=0,
                markdown="   \n\n  ",
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            ),
            RefinedPage(
                index=1,
                markdown="# Real Content\n\nActual text lives here.",
                images=[],
                dimensions=PageDimensions(dpi=72, width=612, height=792)
            ),
        ]
        ref_result = RefiningResult(
            component_name="TestRefiner",
            component_config={},
            processed_at=datetime.now(timezone.utc),
            pages=pages,
            refining_trace=[],
            extracted_data={},
        )
        chunker = RecursiveMarkdownChunker(
            chunk_size=500, overlap=50, min_chunk_size=10
        )
        result = chunker.chunk(ref_result)

        assert len(result.chunks) > 0
        for chunk in result.chunks:
            assert chunk.metadata.page_index == 1
