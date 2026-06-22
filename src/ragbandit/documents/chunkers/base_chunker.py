# ----------------------------------------------------------------------
# Standard library
import logging
import re
from abc import ABC, abstractmethod

# Project
from ragbandit.schema import (
    RefiningResult,
    Chunk,
    ChunkMetadata,
    ChunkingResult,
    Image,
)
from ragbandit.utils.token_usage_tracker import TokenUsageTracker


class BaseChunker(ABC):
    """
    Base class for document chunking strategies.
    Subclasses should implement the `chunk()` method to
    provide specific chunking logic.
    """

    def __init__(self, max_chunk_size: int | None = None):
        """Initialize the chunker.

        Args:
            max_chunk_size: Optional hard upper limit on chunk size in
                characters. Any chunk exceeding this is split further.
                None means no limit.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(
        self,
        document: RefiningResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> ChunkingResult:
        """
        Chunk the document content from a RefiningResult.

        Args:
            document: The RefiningResult containing
                      document content to chunk
            usage_tracker: Optional tracker for token usage

        Returns:
            A ChunkingResult containing the chunks
        """
        raise NotImplementedError

    @abstractmethod
    def get_config(self) -> dict:
        """Return the configuration for this chunker.

        Returns:
            dict: Configuration dictionary
        """
        raise NotImplementedError(
            "Subclasses must implement get_config method"
        )

    def get_name(self) -> str:
        """Return the component name.

        Returns:
            str: The class name of this component
        """
        return self.__class__.__name__

    def merge_small_chunks(
        self, chunks: list[Chunk], min_size: int
    ) -> list[Chunk]:
        """
        Merge small chunks with adjacent chunks to ensure minimum chunk size.

        Args:
            chunks: The chunks to process
            min_size: Minimum size for chunks (smaller chunks will be merged)

        Returns:
            Processed chunks with small chunks merged
        """
        if not chunks:
            return []

        merged = []
        i = 0
        n = len(chunks)

        while i < n:
            current_chunk = chunks[i]
            current_text = current_chunk.text

            # Check if this chunk is "small"
            if len(current_text) < min_size:
                # 1) Try to merge with the NEXT chunk if same page_index
                next_chunk_exists = (i + 1) < n
                if next_chunk_exists:
                    next_chunk_same_page = (
                        chunks[i + 1].metadata.page_index
                        == current_chunk.metadata.page_index
                    )
                else:
                    next_chunk_same_page = False

                if i < n - 1 and next_chunk_same_page:
                    # Merge current with the next chunk
                    current_chunk.text += (" " + chunks[i + 1].text)

                    # Carry over the absorbed chunk's images. This must run
                    # whenever the next chunk has images, even if the surviving
                    # chunk had none — otherwise images whose markers now live
                    # in this chunk's text would be dropped.
                    if chunks[i + 1].metadata.images:
                        current_chunk.metadata.images = (
                            (current_chunk.metadata.images or [])
                            + chunks[i + 1].metadata.images
                        )

                    # We've used chunk i+1, so skip it
                    i += 2

                    # Now this newly merged chunk is complete; add to 'merged'
                    merged.append(current_chunk)
                else:
                    # 2) Otherwise, try to merge with
                    # PREVIOUS chunk in 'merged'
                    if merged:
                        # Merge current chunk into the last chunk in 'merged'
                        merged[-1].text += (" " + current_chunk.text)

                        # Carry over the absorbed chunk's images even if the
                        # surviving chunk had none (see note above).
                        if current_chunk.metadata.images:
                            merged[-1].metadata.images = (
                                (merged[-1].metadata.images or [])
                                + current_chunk.metadata.images
                            )
                    else:
                        # If there's no previous chunk in 'merged', just add it
                        merged.append(current_chunk)

                    i += 1
            else:
                # If it's not "small," just add it as-is
                merged.append(current_chunk)
                i += 1

        return merged

    def _split_oversized_chunks(
        self, chunks: list[Chunk]
    ) -> list[Chunk]:
        """Split chunks that exceed max_chunk_size with a hard character split.

        Called before attach_images so image markers remain in the correct
        sub-chunk and get properly assigned when attach_images runs.

        Args:
            chunks: Chunks to check and split if necessary.

        Returns:
            New chunk list where every chunk is <= max_chunk_size chars.
        """
        if self.max_chunk_size is None:
            return chunks

        result: list[Chunk] = []
        for chunk in chunks:
            if len(chunk.text) <= self.max_chunk_size:
                result.append(chunk)
                continue

            self.logger.warning(
                f"Chunk on page {chunk.metadata.page_index} has "
                f"{len(chunk.text)} chars, exceeding max_chunk_size="
                f"{self.max_chunk_size}. Splitting."
            )
            text = chunk.text
            start = 0
            while start < len(text):
                end = min(start + self.max_chunk_size, len(text))
                result.append(Chunk(
                    text=text[start:end],
                    metadata=ChunkMetadata(
                        page_index=chunk.metadata.page_index,
                        images=[],
                        extra=chunk.metadata.extra or {},
                    ),
                ))
                if end >= len(text):
                    break
                start = end

        return result

    def process_chunks(
        self, chunks: list[Chunk]
    ) -> list[Chunk]:
        """
        Optional post-processing of chunks after initial chunking.
        This can be overridden by subclasses to
        implement additional processing.

        Args:
            chunks: The initial chunks produced by the chunk method

        Returns:
            Processed chunks
        """
        return chunks

    # ------------------------------------------------------------------
    # Shared helpers
    def attach_images(
        self,
        chunks: list[Chunk],
        ref_result: RefiningResult,
    ) -> list[Chunk]:
        """Populate each Chunk's metadata.images with inlined image data.

        Looks for markdown image markers in the chunk text and copies
        the matching ``image_base64`` from the corresponding page's
        images collection.

        Supported filename patterns:
        - ``img-<digits>.jpg/jpeg``  (e.g. ``img-01.jpeg``)
        - ``<hash>_img.jpg``         (e.g. ``1b7d539e_img.jpg``)
        """

        img_pattern = re.compile(
            r"!\[[^\]]*\]"
            r"\((img-\d+\.jpe?g|[a-f0-9]+_img\.jpe?g)\)"
        )

        for chunk in chunks:
            images_in_chunk = img_pattern.findall(chunk.text)
            if not images_in_chunk:
                # No image markers, ensure empty list and continue
                chunk.metadata.images = []
                continue

            page_idx = chunk.metadata.page_index
            rel_images = ref_result.pages[page_idx].images or []
            chunk.metadata.images = []

            for img_id in images_in_chunk:
                for ocr_img in rel_images:
                    if ocr_img.id == img_id:
                        chunk.metadata.images.append(
                            Image(id=img_id, image_base64=ocr_img.image_base64)
                        )
                        break

        return chunks

    def __str__(self) -> str:
        """Return a string representation of the chunker."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return a string representation of the chunker."""
        return f"{self.__class__.__name__}()"
