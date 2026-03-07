"""
Sentence-based chunker for splitting documents into sliding-window chunks.

Groups sentences into fixed-size windows with configurable overlap.
Uses regex sentence splitting with no external dependencies.
"""

import re
from datetime import datetime, timezone

from ragbandit.schema import (
    RefiningResult,
    Chunk,
    ChunkMetadata,
    ChunkingResult,
)
from ragbandit.utils.token_usage_tracker import TokenUsageTracker
from ragbandit.documents.chunkers.base_chunker import BaseChunker


class SentenceChunker(BaseChunker):
    """
    Chunks documents by grouping sentences into sliding-window chunks.

    Splits each page into sentences using a regex boundary pattern, then
    groups them into windows of ``sentences_per_chunk`` sentences, advancing
    by ``sentences_per_chunk - sentence_overlap`` sentences each step so that
    adjacent chunks share ``sentence_overlap`` sentences of context.

    Uses no external dependencies (pure-Python regex split).
    """

    # Sentence boundary: period/!/? followed by whitespace or end of string
    _SENT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        sentences_per_chunk: int = 5,
        sentence_overlap: int = 1,
        min_chunk_size: int = 100,
        max_chunk_size: int | None = None,
    ):
        """
        Initialize the sentence chunker.

        Args:
            sentences_per_chunk: Number of sentences per chunk.
            sentence_overlap: Number of sentences to overlap between chunks.
            min_chunk_size: Minimum character length; smaller chunks
                are merged with adjacent ones.
            max_chunk_size: Hard upper limit on chunk size in characters.
                Chunks exceeding this are split further. None means no limit.

        Raises:
            ValueError: If sentence_overlap >= sentences_per_chunk.
        """
        if sentence_overlap >= sentences_per_chunk:
            raise ValueError(
                f"sentence_overlap ({sentence_overlap}) must be less than "
                f"sentences_per_chunk ({sentences_per_chunk})"
            )
        super().__init__(max_chunk_size=max_chunk_size)
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
        self.min_chunk_size = min_chunk_size

    def get_config(self) -> dict:
        """Return the configuration for this chunker.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            "sentences_per_chunk": self.sentences_per_chunk,
            "sentence_overlap": self.sentence_overlap,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
        }

    def chunk(
        self,
        ref_result: RefiningResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> ChunkingResult:
        """
        Chunk the document into sentence-window chunks.

        Args:
            ref_result: The RefiningResult containing document content
                to chunk.
            usage_tracker: Optional tracker for token usage (not used
                by this chunker).

        Returns:
            A ChunkingResult containing Chunk objects.
        """
        chunks = self._sentence_chunk_pages(ref_result)
        chunks = self._split_oversized_chunks(chunks)
        chunks = self.attach_images(chunks, ref_result)
        chunks = self.process_chunks(chunks)
        return ChunkingResult(
            component_name=self.get_name(),
            component_config=self.get_config(),
            processed_at=datetime.now(timezone.utc),
            chunks=chunks,
            metrics=None,
        )

    def process_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Merge chunks that are smaller than min_chunk_size.

        Args:
            chunks: The initial chunks produced by the chunk method.

        Returns:
            Processed chunks with small chunks merged if needed.
        """
        if not chunks:
            return chunks
        min_len = min(len(c.text) for c in chunks)
        if min_len < self.min_chunk_size:
            self.logger.info(
                f"Found chunks smaller than {self.min_chunk_size} chars. "
                "Merging..."
            )
            chunks = self.merge_small_chunks(
                chunks, min_size=self.min_chunk_size
            )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using a punctuation-boundary regex.

        Args:
            text: Raw page text.

        Returns:
            List of non-empty sentence strings.
        """
        parts = self._SENT_RE.split(text.strip())
        return [s.strip() for s in parts if s.strip()]

    def _sentence_chunk_pages(self, ref_result: RefiningResult) -> list[Chunk]:
        """Build sentence-window chunks for every page in the document.

        Each window spans ``sentences_per_chunk`` sentences and advances
        by ``sentences_per_chunk - sentence_overlap`` sentences.

        Args:
            ref_result: RefiningResult containing the document pages.

        Returns:
            List of Chunk objects with page_index metadata set.
        """
        self.logger.info(
            f"Starting sentence chunking: sentences_per_chunk="
            f"{self.sentences_per_chunk}, overlap={self.sentence_overlap}"
        )
        chunks: list[Chunk] = []
        step = self.sentences_per_chunk - self.sentence_overlap

        for page_index, page in enumerate(ref_result.pages):
            if not page.markdown.strip():
                continue

            sentences = self._split_sentences(page.markdown)
            if not sentences:
                continue

            i = 0
            while i < len(sentences):
                window = sentences[i: i + self.sentences_per_chunk]
                chunk_text = " ".join(window)
                meta = ChunkMetadata(
                    page_index=page_index, images=[], extra={}
                )
                chunks.append(Chunk(text=chunk_text, metadata=meta))
                i += step

        self.logger.info(
            f"Sentence chunking complete. Created {len(chunks)} chunks."
        )
        return chunks
