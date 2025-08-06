import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from document_data_models import ExtendedOCRResponse
from document_utils.cost_tracking import TokenUsageTracker


class BaseChunker(ABC):
    """
    Base class for document chunking strategies.
    Subclasses should implement the `chunk()` method to
    provide specific chunking logic.
    """

    def __init__(self, name: str | None = None):
        # Hierarchical names make it easy to filter later:
        #   chunker.semantic, chunker.fixed_size, etc.
        base = "chunker"
        self.logger = logging.getLogger(
            f"{base}.{name or self.__class__.__name__}"
        )

    @abstractmethod
    def chunk(
        self,
        response: ExtendedOCRResponse,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Chunk the document content from an ExtendedOCRResponse.

        Args:
            response: The ExtendedOCRResponse containing
                      document content to chunk
            usage_tracker: Optional tracker for token usage during chunking

        Returns:
            A list of chunk dictionaries, where each chunk contains at minimum:
            - chunk_text: The text content of the chunk
            - page_index: The index of the page this chunk comes from

        Additional fields may be added by specific chunker implementations.
        """
        raise NotImplementedError

    def merge_small_chunks(
        self, chunks: List[Dict[str, Any]], min_size: int
    ) -> List[Dict[str, Any]]:
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
            current_text = current_chunk["chunk_text"]

            # Check if this chunk is "small"
            if len(current_text) < min_size:
                # 1) Try to merge with the NEXT chunk if same page_index
                next_chunk_exists = (i + 1) < n
                if next_chunk_exists:
                    next_chunk_same_page = (
                        chunks[i + 1]["page_index"]
                        == current_chunk["page_index"]
                    )
                else:
                    next_chunk_same_page = False

                if i < n - 1 and next_chunk_same_page:
                    # Merge current with the next chunk
                    current_chunk["chunk_text"] += (
                        " " + chunks[i + 1]["chunk_text"]
                    )

                    # Merge images if they exist
                    if "images" in current_chunk and "images" in chunks[i + 1]:
                        current_chunk["images"].extend(chunks[i + 1]["images"])

                    # We've used chunk i+1, so skip it
                    i += 2

                    # Now this newly merged chunk is complete; add to 'merged'
                    merged.append(current_chunk)
                else:
                    # 2) Otherwise, try to merge with
                    # PREVIOUS chunk in 'merged'
                    if merged:
                        # Merge current chunk into the last chunk in 'merged'
                        merged[-1]["chunk_text"] += (
                            " " + current_chunk["chunk_text"]
                        )

                        # Merge images if they exist
                        if (
                            "images" in merged[-1]
                            and "images" in current_chunk
                        ):
                            merged[-1]["images"].extend(
                                current_chunk["images"]
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

    def process_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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

    def extend_response(
        self, response: ExtendedOCRResponse, chunks: List[Dict[str, Any]]
    ) -> ExtendedOCRResponse:
        """
        Extend the response with chunk metadata.

        Args:
            response: ExtendedOCRResponse to update
            chunks: The chunks produced by this chunker

        Returns:
            Updated ExtendedOCRResponse
        """
        if response.processing_metadata is None:
            response.processing_metadata = {}

        # Store chunking results in processing_metadata
        response.processing_metadata[self.__repr__()] = {
            "chunk_count": len(chunks),
            "chunks": chunks,
        }

        return response

    def __str__(self) -> str:
        """Return a string representation of the chunker."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return a string representation of the chunker."""
        return f"{self.__class__.__name__}()"
