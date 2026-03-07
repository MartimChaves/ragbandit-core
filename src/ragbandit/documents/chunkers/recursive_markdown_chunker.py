"""
Recursive markdown chunker for splitting documents hierarchically.

Splits text by heading level (H1 → H2 → H3 → H4), then by paragraph,
then by sentence, falling back to a hard character split as a last resort.
Adjacent short segments are merged with overlap to preserve context.
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


class RecursiveMarkdownChunker(BaseChunker):
    """
    Splits documents hierarchically by heading level (H1→H2→H3→H4),
    then by paragraph, then by sentence, falling back to character split.

    Each page is split recursively: the first splitter that produces more
    than one part is used, and any part still larger than ``chunk_size``
    is passed to the next splitter in the hierarchy. After splitting,
    short adjacent segments are merged (with overlap) to avoid tiny chunks.
    """

    # Ordered list of splitters tried in sequence: coarser splits first.
    _SPLITTERS = [
        re.compile(r"(?=^# )", re.MULTILINE),     # H1
        re.compile(r"(?=^## )", re.MULTILINE),    # H2
        re.compile(r"(?=^### )", re.MULTILINE),   # H3
        re.compile(r"(?=^#### )", re.MULTILINE),  # H4
        re.compile(r"\n\n+"),                     # paragraph break
        re.compile(r"(?<=[.!?])\s+"),             # sentence boundary
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
        min_chunk_size: int = 100,
        max_chunk_size: int | None = None,
    ):
        """
        Initialize the recursive markdown chunker.

        Args:
            chunk_size: Target maximum chunk size in characters. Segments
                larger than this are split further down the hierarchy.
            overlap: Characters of context prepended from the end of the
                previous chunk when merging segments.
            min_chunk_size: Chunks smaller than this are merged with
                adjacent ones after splitting.
            max_chunk_size: Hard upper limit on chunk size in characters.
                Chunks exceeding this are split further. None means no limit.
        """
        super().__init__(max_chunk_size=max_chunk_size)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def get_config(self) -> dict:
        """Return the configuration for this chunker.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
        }

    def chunk(
        self,
        ref_result: RefiningResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> ChunkingResult:
        """
        Chunk the document using recursive markdown splitting.

        Args:
            ref_result: The RefiningResult containing document content
                to chunk.
            usage_tracker: Optional tracker for token usage (not used
                by this chunker).

        Returns:
            A ChunkingResult containing Chunk objects.
        """
        chunks = self._recursive_chunk_pages(ref_result)
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

    def _split_text(self, text: str, splitter_index: int = 0) -> list[str]:
        """Recursively split text using the splitter hierarchy.

        Tries each splitter in ``_SPLITTERS`` in order. If a splitter
        produces only one part (i.e. it did nothing), the next splitter is
        tried. Parts still larger than ``chunk_size`` are split again with
        the next splitter. Falls back to ``_char_split`` when all splitters
        are exhausted.

        Args:
            text: Text to split.
            splitter_index: Index into ``_SPLITTERS`` to try next.

        Returns:
            List of text segments each at most ``chunk_size`` characters.
        """
        if len(text) <= self.chunk_size:
            return [text]

        if splitter_index >= len(self._SPLITTERS):
            return self._char_split(text)

        splitter = self._SPLITTERS[splitter_index]
        parts = splitter.split(text)
        parts = [p for p in parts if p.strip()]

        if len(parts) <= 1:
            # This splitter didn't divide the text; try the next one
            return self._split_text(text, splitter_index + 1)

        result: list[str] = []
        for part in parts:
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._split_text(part, splitter_index + 1))

        return result

    def _char_split(self, text: str) -> list[str]:
        """Hard character split with overlap as a last resort.

        Used when no splitter in the hierarchy can divide the text further.

        Args:
            text: Text to split by raw character count.

        Returns:
            List of segments each at most ``chunk_size`` characters, with
            ``overlap`` characters of context carried over between them.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - self.overlap
            if start <= 0:
                break
        return chunks

    def _merge_with_overlap(self, segments: list[str]) -> list[str]:
        """Merge adjacent short segments and add overlap between chunks.

        Short segments that together fit within ``chunk_size`` are joined
        into a single chunk. When a new segment would exceed the limit, the
        current chunk is finalised and the next chunk begins with
        ``overlap`` characters of context from the end of the previous one.

        Args:
            segments: List of text segments from ``_split_text``.

        Returns:
            Merged list of chunks with overlap context prepended.
        """
        if not segments:
            return []

        merged: list[str] = []
        current = segments[0]

        for seg in segments[1:]:
            if len(current) + len(seg) + 1 <= self.chunk_size:
                current = current.rstrip() + "\n\n" + seg.lstrip()
            else:
                merged.append(current)
                overlap_text = current[-self.overlap:] if self.overlap else ""
                current = (
                    (overlap_text + " " + seg).strip()
                    if overlap_text
                    else seg
                )

        merged.append(current)
        return merged

    def _recursive_chunk_pages(
        self, ref_result: RefiningResult
    ) -> list[Chunk]:
        """Build recursively-split chunks for every page in the document.

        Each page is split with ``_split_text``, short segments are merged
        with ``_merge_with_overlap``, and the result is wrapped in Chunk
        objects with the originating page index.

        Args:
            ref_result: RefiningResult containing the document pages.

        Returns:
            List of Chunk objects with page_index metadata set.
        """
        self.logger.info(
            f"Starting recursive markdown chunking: "
            f"chunk_size={self.chunk_size}, overlap={self.overlap}"
        )
        chunks: list[Chunk] = []

        for page_index, page in enumerate(ref_result.pages):
            if not page.markdown.strip():
                continue

            segments = self._split_text(page.markdown)
            segments = self._merge_with_overlap(segments)

            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue
                meta = ChunkMetadata(
                    page_index=page_index, images=[], extra={}
                )
                chunks.append(Chunk(text=seg, metadata=meta))

        self.logger.info(
            "Recursive markdown chunking complete. "
            f"Created {len(chunks)} chunks."
        )
        return chunks
