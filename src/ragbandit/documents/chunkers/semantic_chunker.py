import re

from pydantic import BaseModel
from ragbandit.schema import ExtendedOCRResponse
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

    def __init__(self, min_chunk_size: int = 500, name: str | None = None):
        """
        Initialize the semantic chunker.

        Args:
            min_chunk_size: Minimum size for chunks
                            (smaller chunks will be merged)
            name: Optional name for the chunker
        """
        super().__init__(name)
        self.min_chunk_size = min_chunk_size

    def semantic_chunk_pages(
        self, pages: list[dict], usage_tracker: TokenUsageTracker | None = None
    ) -> list[dict[str, any]]:
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
        chunks = []

        while i < len(pages):
            # If we have "remainder" from the last iteration,
            # it might be appended here
            break_lead = find_semantic_break_tool(
                text=full_text, usage_tracker=usage_tracker
            )

            if break_lead == "NO_BREAK":
                # This means the LLM found no break;
                # treat the entire `full_text` as one chunk
                chunk_dict = {"chunk_text": full_text, "page_index": i}
                chunks.append(chunk_dict)
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
                    chunk_dict = {"chunk_text": full_text, "page_index": i}
                    chunks.append(chunk_dict)
                    i += 1
                    if i < len(pages):
                        full_text = pages[i].markdown
                    else:
                        break
                else:
                    # We found a break
                    chunk_text = full_text[:idx]
                    remainder = full_text[idx:]
                    chunk_dict = {"chunk_text": chunk_text, "page_index": i}
                    chunks.append(chunk_dict)

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
                            chunk_dict = {
                                "chunk_text": full_text,
                                "page_index": min(i, len(pages) - 1),
                            }
                            chunks.append(chunk_dict)
                        break

        return chunks

    def chunk(
        self,
        response: ExtendedOCRResponse,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> list[dict[str, any]]:
        """
        Chunk the document using semantic chunking.

        Args:
            response: The ExtendedOCRResponse containing
                      document content to chunk
            usage_tracker: Tracker for token usage during chunking

        Returns:
            A list of chunk dictionaries
        """
        self.logger.info("Starting semantic chunking")

        # Get the pages from the response
        pages = response.pages

        # Perform semantic chunking
        chunks = self.semantic_chunk_pages(pages, usage_tracker)

        # Process images in chunks
        chunks = self._process_images(chunks, response)

        # Process the chunks (merge small chunks)
        chunks = self.process_chunks(chunks)

        self.logger.info(
            f"Semantic chunking complete. Created {len(chunks)} chunks."
        )

        return chunks

    def process_chunks(
        self, chunks: list[dict[str, any]]
    ) -> list[dict[str, any]]:
        """
        Process chunks after initial chunking - merge small chunks.

        Args:
            chunks: The initial chunks produced by the chunk method

        Returns:
            Processed chunks with small chunks merged
        """
        # Check if any chunks are too small
        min_len = min([len(c["chunk_text"]) for c in chunks]) if chunks else 0

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

    def _process_images(
        self, chunks: list[dict[str, any]], response: ExtendedOCRResponse
    ) -> list[dict[str, any]]:
        """
        Process images in chunks and add them to chunk info.

        Args:
            chunks: The chunks to process
            response: The ExtendedOCRResponse containing image data

        Returns:
            Chunks with image information added
        """
        # If chunk has images in them, add it to chunk info
        img_pattern = re.compile(r"!\[img-\d+\.jpeg\]\(img-\d+\.jpeg\)")

        for chunk in chunks:
            images_in_chunk = img_pattern.findall(chunk["chunk_text"])
            has_image = len(images_in_chunk) > 0
            page_index = chunk["page_index"]
            chunk["images"] = []

            if has_image:
                for img in images_in_chunk:
                    id_of_img = img.split("[")[1].split("]")[0]
                    base64_of_img = None
                    rel_images = response.pages[page_index].images

                    for ocr_img in rel_images:
                        if ocr_img.id == id_of_img:
                            base64_of_img = ocr_img.image_base64
                            break

                    if base64_of_img:
                        img_info = {
                            "img_id": id_of_img,
                            "image_base64": base64_of_img,
                        }
                        chunk["images"].append(img_info)

        return chunks
