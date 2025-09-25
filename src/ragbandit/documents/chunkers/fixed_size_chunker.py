from datetime import datetime, timezone

from ragbandit.schema import (
    ProcessingResult,
    Chunk,
    ChunkMetadata,
    ChunkingResult,
)
from ragbandit.utils.token_usage_tracker import TokenUsageTracker
from ragbandit.documents.chunkers.base_chunker import BaseChunker


class FixedSizeChunker(BaseChunker):
    """
    A document chunker that splits documents into fixed-size chunks
    with optional overlap between chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        name: str | None = None,
    ):
        """
        Initialize the fixed size chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            overlap: Number of characters to overlap between chunks
            name: Optional name for the chunker
        """
        super().__init__(name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        proc_result: ProcessingResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> ChunkingResult:
        """
        Chunk the document into fixed-size chunks.

        Args:
            proc_result: The ProcessingResult containing
                      document content to chunk
            usage_tracker: Optional tracker for token usage
                           (not used in this chunker)

        Returns:
            A ChunkingResult containing Chunk objects
        """
        # 1. Generate raw chunks for each page
        chunks = self._fixed_size_chunk_pages(proc_result)

        # 2. Attach any inline images using BaseChunker helper
        chunks = self.attach_images(chunks, proc_result)

        # 3. Merge small chunks if needed
        chunks = self.process_chunks(chunks)

        # 4. Wrap in ChunkingResult
        return ChunkingResult(
            processed_at=datetime.now(timezone.utc),
            chunks=chunks,
            metrics=None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    def _fixed_size_chunk_pages(
        self, proc_result: ProcessingResult
    ) -> list[Chunk]:
        """Split each page into fixed-size chunks with optional overlap."""

        self.logger.info(
            f"Starting fixed-size chunking with size={self.chunk_size}, "
            f"overlap={self.overlap}"
        )

        chunks: list[Chunk] = []

        # Process each page
        for page_index, page in enumerate(proc_result.pages):
            page_text = page.markdown

            # Skip empty pages
            if not page_text.strip():
                continue

            # Create chunks from this page
            start = 0
            while start < len(page_text):
                # Determine end position for this chunk
                end = min(start + self.chunk_size, len(page_text))

                # If we're not at the end of the text,
                # try to find a good break point
                if end < len(page_text):
                    # Look for a period, question mark, or exclamation mark
                    # followed by whitespace
                    # within the last 100 characters of the chunk
                    search_start = max(end - 100, start)
                    for i in range(end, search_start, -1):
                        # Check if we're at a valid position to examine
                        if i <= 0 or i >= len(page_text):
                            continue

                        # Check if the previous character is punctuation
                        # and the current character is whitespace
                        if (
                            page_text[i - 1] in [".", "!", "?"]
                            and page_text[i].isspace()
                        ):
                            end = i
                            break

                # Create the chunk
                chunk_text = page_text[start:end]
                meta = ChunkMetadata(
                    page_index=page_index, images=[], extra={}
                    )
                chunks.append(Chunk(text=chunk_text, metadata=meta))

                # Check if we've reached the end of the page text
                if end >= len(page_text):
                    # We've processed the entire page, exit the loop
                    break

                # Move to next chunk start position, accounting for overlap
                start = end - self.overlap

                # Make sure we're making progress
                if start <= 0 or start >= len(page_text):
                    break

        self.logger.info(
            f"Fixed-size chunking complete. Created {len(chunks)} chunks."
        )

        return chunks

    def process_chunks(
        self, chunks: list[Chunk]
    ) -> list[Chunk]:
        """
        Process chunks after initial chunking - merge small chunks if needed.

        Args:
            chunks: The initial chunks produced by the chunk method

        Returns:
            Processed chunks with small chunks merged if needed
        """
        if not chunks:
            return chunks

        # Calculate minimum chunk size as a fraction of the target chunk size
        min_chunk_size = self.chunk_size // 2

        # Check if any chunks are too small
        min_len = min(len(c.text) for c in chunks)

        # Merge small chunks if needed
        if min_len < min_chunk_size:
            self.logger.info(
                f"Found chunks smaller than {min_chunk_size} characters. "
                "Merging..."
            )
            chunks = self.merge_small_chunks(chunks, min_size=min_chunk_size)
            self.logger.info(f"After merging: {len(chunks)} chunks")

        return chunks
