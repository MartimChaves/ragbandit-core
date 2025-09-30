from datetime import datetime, timezone
from pydantic import BaseModel
from ragbandit.schema import (
    ProcessingResult,
    Chunk,
    ChunkMetadata,
    ChunkingResult,
)
from ragbandit.utils.token_usage_tracker import TokenUsageTracker

from ragbandit.prompt_tools.semantic_chunker_tools import (
    find_semantic_break_tool,
)

from ragbandit.documents.chunkers.base_chunker import BaseChunker


class SemanticBreak(BaseModel):
    semantic_break: str


class SemanticChunker(BaseChunker):
    """
    A document chunker that uses semantic understanding to split documents
    into coherent chunks based on content.
    """

    def __init__(
        self,
        min_chunk_size: int = 500,
        name: str | None = None,
        api_key: str | None = None
    ):
        """
        Initialize the semantic chunker.

        Args:
            min_chunk_size: Minimum size for chunks
                            (smaller chunks will be merged)
            name: Optional name for the chunker
            api_key: Mistral API Key
        """
        super().__init__(name, api_key)
        self.min_chunk_size = min_chunk_size

    def semantic_chunk_pages(
        self, pages: list, usage_tracker: TokenUsageTracker | None = None
    ) -> list[Chunk]:
        """
        Chunk pages semantically using LLM-based semantic breaks.

        Args:
            pages: List of page dictionaries with markdown content
            usage_tracker: Optional tracker for token usage

        Returns:
            A list of chunk dictionaries
        """
        i = 0
        full_text = pages[i].markdown
        chunks: list[Chunk] = []

        while i < len(pages):
            # If we have "remainder" from the last iteration,
            # it might be appended here
            break_lead = find_semantic_break_tool(
                api_key=self.api_key,
                text=full_text,
                usage_tracker=usage_tracker
            )

            if break_lead == "NO_BREAK":
                # This means the LLM found no break;
                # treat the entire `full_text` as one chunk
                meta = ChunkMetadata(page_index=i, images=[], extra={})
                chunks.append(Chunk(text=full_text, metadata=meta))
                # Move to the next page
                i += 1
                if i < len(pages):
                    full_text = pages[i].markdown
                else:
                    break
            else:
                # Attempt to find the snippet in the text
                idx = full_text.find(break_lead)

                # If exact match fails, try progressively shorter versions
                if idx == -1 and len(break_lead) > 0:
                    current_break_lead = break_lead
                    min_length = 10  # Minimum characters to try matching

                    # Try progressively shorter versions
                    # of the break_lead until we find a match
                    # or reach the minimum length
                    while idx == -1 and len(current_break_lead) >= min_length:
                        # Cut the break_lead in half and try again
                        current_break_lead = current_break_lead[
                            : len(current_break_lead) // 2
                        ]
                        idx = full_text.find(current_break_lead)

                if idx == -1:
                    # If we still can't find the snippet after
                    # trying shorter versions,
                    # fallback: chunk everything as is
                    meta = ChunkMetadata(page_index=i, images=[], extra={})
                    chunks.append(Chunk(text=full_text, metadata=meta))
                    i += 1
                    if i < len(pages):
                        full_text = pages[i].markdown
                    else:
                        break
                else:
                    # We found a break
                    chunk_text = full_text[:idx]
                    remainder = full_text[idx:]
                    meta = ChunkMetadata(page_index=i, images=[], extra={})
                    chunks.append(Chunk(text=chunk_text, metadata=meta))

                    # Now we see if remainder is too small
                    if len(remainder) < 1500:  # ~some threshold
                        i += 1
                        if i < len(pages):
                            # Combine remainder with next page
                            remainder += "\n" + pages[i].markdown
                    # remainder becomes the new full_text
                    full_text = remainder

                    # If we used up the last page, break
                    if i >= len(pages):
                        # Possibly chunk the remainder if it's not empty
                        if len(full_text.strip()) > 0:
                            meta = ChunkMetadata(
                                page_index=min(i, len(pages) - 1),
                                images=[],
                                extra={},
                            )
                            chunks.append(Chunk(text=full_text, metadata=meta))
                        break

        return chunks

    def chunk(
        self,
        proc_result: ProcessingResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> ChunkingResult:
        """
        Chunk the document using semantic chunking.

        Args:
            proc_result: The ProcessingResult containing
                      document content to chunk
            usage_tracker: Tracker for token usage during chunking

        Returns:
            A list of chunk dictionaries
        """
        self.logger.info("Starting semantic chunking")

        # Get the pages from the response
        pages = proc_result.pages

        # Perform semantic chunking
        chunks = self.semantic_chunk_pages(pages, usage_tracker)

        # Attach image data to chunks using shared helper
        chunks = self.attach_images(chunks, proc_result)

        # Merge small chunks if needed
        chunks = self.process_chunks(chunks)

        return ChunkingResult(
            processed_at=datetime.now(timezone.utc),
            chunks=chunks,
            metrics=usage_tracker.get_summary() if usage_tracker else None,
        )

    def process_chunks(
        self, chunks: list[Chunk]
    ) -> list[Chunk]:
        """
        Process chunks after initial chunking - merge small chunks.

        Args:
            chunks: The initial chunks produced by the chunk method

        Returns:
            Processed chunks with small chunks merged
        """
        # Check if any chunks are too small
        min_len = min([len(c.text) for c in chunks]) if chunks else 0

        # Merge small chunks if needed
        if min_len < self.min_chunk_size:
            self.logger.info(
                f"Found chunks smaller than {self.min_chunk_size} characters. "
                "Merging..."
            )
            chunks = self.merge_small_chunks(
                chunks, min_size=self.min_chunk_size
            )
            self.logger.info(f"After merging: {len(chunks)} chunks")

        return chunks
