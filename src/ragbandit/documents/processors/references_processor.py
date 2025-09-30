"""
References processor for detecting and removing reference
sections from documents.

This processor identifies the references section header in a document and
extracts the references content, removing it from the main document text.
"""

import re
from difflib import SequenceMatcher

from ragbandit.documents.processors.base_processor import BaseProcessor
from ragbandit.utils.token_usage_tracker import TokenUsageTracker
from ragbandit.prompt_tools.references_processor_tools import (
    detect_references_header_tool,
)
from ragbandit.schema import OCRResult, ProcessingResult


class ReferencesProcessor(BaseProcessor):
    """Processor for detecting and removing references sections from documents.

    This processor:
    1. Extracts headers from the OCR pages
    2. Identifies the references section header using an LLM
    3. Removes the references section from the document
    4. Returns the modified document and the extracted references as markdown
    """

    def __init__(self, name: str | None = None, api_key: str | None = None):
        """Initialize the references processor.

        Args:
            name: Optional name for the processor
            api_key: API key for LLM services
        """
        super().__init__(name, api_key)

    def process(
        self,
        document: OCRResult | ProcessingResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> ProcessingResult:
        """Process OCR pages to detect and remove references.

        Args:
            document: OCRResult or ProcessingResult to process
            usage_tracker: Token usage tracker for LLM calls

        Returns:
            Tuple containing:
            - Modified ProcessingResult with references removed
            - Extracted references as markdown
        """

        # Normalize input once
        proc_input = self.ensure_processing_result(
                        document, processor_name=str(self)
                    )

        proc_result, references_markdown = self.remove_refs(
            proc_input, usage_tracker
        )

        # Save extracted references into processing result metadata
        if references_markdown:
            if proc_result.extracted_data is None:
                proc_result.extracted_data = {}

            proc_result.extracted_data["references_markdown"] = (
                references_markdown
            )

        return proc_result

    def find_best_match(
        self, target: str, string_list: list[str]
    ) -> tuple[str, int]:
        """
        Find the string in string_list that best contains the target string.

        Args:
            target: The string to search for
            string_list: List of strings to search through

        Returns:
            A tuple containing (best matching string, index of best match)
            If list is empty, returns ("", -1)
        """
        if not string_list or not target:
            return "", -1

        def similarity_ratio(s1: str, s2: str) -> float:
            return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

        best_idx = max(
            range(len(string_list)),
            key=lambda i: similarity_ratio(target, string_list[i]),
        )
        return string_list[best_idx], best_idx

    def remove_refs(
        self,
        proc_result: ProcessingResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> tuple[ProcessingResult, str]:
        """Remove references section from document and extract as markdown.

        This method identifies the references section in a document,
        extracts it, and removes it from the original document.

        Args:
            proc_result: The document to process (ProcessingResult)
            usage_tracker: Optional tracker for token usage in LLM calls

        Returns:
            Tuple containing:
            - Modified ProcessingResult with references removed
            - Extracted references as markdown
        """
        # Extract headers and identify references section
        headers = self._extract_headers(proc_result)
        refs_header, refs_header_index = self._identify_references_header(
            headers, usage_tracker
        )

        # If no references header found, return original document unchanged
        if not refs_header:
            return proc_result, ""

        # Find next header (if any) after references
        next_header = self._find_next_header(headers, refs_header_index)

        # Find page boundaries of references section
        boundaries = self._find_reference_boundaries(
            proc_result, refs_header, next_header
        )

        # If boundaries couldn't be determined, return original document
        if not boundaries:
            return proc_result, ""

        # Extract references and modify document
        return self._extract_references(proc_result, boundaries)

    def _extract_headers(self, proc_result: ProcessingResult) -> list[str]:
        """Extract all headers from document.

        Args:
            proc_result: ProcessingResult containing document pages

        Returns:
            List of headers found in the document
        """
        # Define header regular expression -
        # looks for header symbols (# to ######)
        header_regex = re.compile(
            r"(?im)(\s*#{1,6}\s*(?![^\n]*\|)[^\n]+(?:\n|$))"
        )

        # Search for headers in complete markdown string
        full_markdown = ""
        for page in proc_result.pages:
            full_markdown += page.markdown

        return header_regex.findall(full_markdown)

    def _identify_references_header(
        self,
        headers: list[str],
        usage_tracker: TokenUsageTracker | None = None,
    ) -> tuple[str, int]:
        """Identify the references header from a list of headers.

        Args:
            headers: List of headers to search through
            usage_tracker: Optional tracker for token usage in LLM calls

        Returns:
            Tuple containing the references header and its index
        """
        if not headers:
            return "", -1

        # Use LLM to identify the most likely references header
        refs = detect_references_header_tool(
            api_key=self.api_key,
            usage_tracker=usage_tracker,
            headers_list=headers
        )

        # Find the best match for the identified header
        return self.find_best_match(refs.references_header, headers)

    def _find_next_header(
        self, headers: list[str], refs_header_index: int
    ) -> str | None:
        """Find the next header after the references header.

        Args:
            headers: List of all headers
            refs_header_index: Index of the references header

        Returns:
            Next header if it exists, None otherwise
        """
        if refs_header_index < 0 or (refs_header_index + 1) >= len(headers):
            return None
        return headers[refs_header_index + 1]

    def _find_reference_boundaries(
        self,
        proc_result: ProcessingResult,
        refs_header: str,
        next_header: str | None,
    ) -> dict | None:
        """Find the boundaries of the references section.

        Args:
            proc_result: ProcessingResult containing document pages
            refs_header: The identified references header
            next_header: The next header after references (if any)

        Returns:
            Dictionary containing boundary information or None if not found
        """
        refs_page = -1
        next_header_page = -1

        # Find the pages where references start and end
        for page in proc_result.pages:
            if refs_header in page.markdown:
                refs_page = page.index
            if next_header is not None and next_header in page.markdown:
                next_header_page = page.index

        # If references header wasn't found in any page, return None
        if refs_page == -1:
            return None

        # Get the location (page, index) where references start
        refs_page_markdown = proc_result.pages[refs_page].markdown
        references_start_index = refs_page_markdown.find(refs_header)
        references_start = (refs_page, references_start_index)

        # Determine where references end
        references_end = None
        if next_header is not None and next_header_page != -1:
            next_header_page_markdown = proc_result.pages[
                next_header_page
            ].markdown
            references_end_index = next_header_page_markdown.find(next_header)
            if references_end_index is not None:
                references_end = (next_header_page, references_end_index)

        return {
            "start": references_start,
            "end": references_end,
            "refs_header": refs_header,
            "next_header": next_header,
        }

    def _extract_references(
        self, proc_result: ProcessingResult, boundaries: dict
    ) -> tuple[ProcessingResult, str]:
        """Extract references from document based on boundaries.

        Args:
            proc_result: ProcessingResult containing document pages
            boundaries: Dictionary with reference section boundaries

        Returns:
            Tuple containing modified document and extracted references
        """
        references_start = boundaries["start"]
        references_end = boundaries["end"]

        # If references end at the end of document
        if references_end is None:
            return self._extract_references_at_end(
                proc_result, references_start
            )

        # If references are contained within a single page
        if references_end[0] == references_start[0]:
            return self._extract_references_same_page(
                proc_result, references_start, references_end
            )

        # If references span multiple pages
        return self._extract_references_multi_page(
            proc_result, references_start, references_end
        )

    def _extract_references_at_end(
        self, proc_result: ProcessingResult, references_start: tuple[int, int]
    ) -> tuple[ProcessingResult, str]:
        """Extract references when they are the last section in the document.

        Args:
            proc_result: ProcessingResult containing document pages
            references_start: Tuple (page_index, char_index) where
                              references start

        Returns:
            Tuple containing modified document and extracted references
        """
        references_markdown = ""
        start_page = True

        for page_index in range(references_start[0], len(proc_result.pages)):
            if start_page:
                # Extract references text from first page,
                # preserve text before references
                references_markdown += proc_result.pages[page_index].markdown[
                    references_start[1]:
                ]
                proc_result.pages[page_index].markdown = proc_result.pages[
                    page_index
                ].markdown[0:references_start[1]]
                start_page = False
                continue

            # For subsequent pages, extract all content
            # (assumed to be references)
            references_markdown += proc_result.pages[page_index].markdown
            proc_result.pages[page_index].markdown = ""

        return proc_result, references_markdown

    def _extract_references_same_page(
        self,
        proc_result: ProcessingResult,
        references_start: tuple[int, int],
        references_end: tuple[int, int],
    ) -> tuple[ProcessingResult, str]:
        """Extract references when they start and end on the same page.

        Args:
            proc_result: ProcessingResult containing document pages
            references_start: Tuple (page_index, char_index) where
                              references start
            references_end: Tuple (page_index, char_index) where references end

        Returns:
            Tuple containing modified document and extracted references
        """
        page_idx = references_start[0]

        # Extract the references section
        references_markdown = proc_result.pages[page_idx].markdown[
            references_start[1]:references_end[1]
        ]

        # Remove references section from the page
        proc_result.pages[page_idx].markdown = (
            proc_result.pages[page_idx].markdown[0:references_start[1]]
            + proc_result.pages[page_idx].markdown[references_end[1]:]
        )

        return proc_result, references_markdown

    def _extract_references_multi_page(
        self,
        proc_result: ProcessingResult,
        references_start: tuple[int, int],
        references_end: tuple[int, int],
    ) -> tuple[ProcessingResult, str]:
        """Extract references when they span multiple pages.

        Args:
            proc_result: ProcessingResult containing document pages
            references_start: Tuple (page_index, char_index) where
                              references start
            references_end: Tuple (page_index, char_index) where references end

        Returns:
            Tuple containing modified document and extracted references
        """
        references_markdown = ""

        # Process each page in the range
        for page_index in range(references_start[0], references_end[0] + 1):
            # First page with references
            if page_index == references_start[0]:
                references_markdown += proc_result.pages[page_index].markdown[
                    references_start[1]:
                ]
                proc_result.pages[page_index].markdown = proc_result.pages[
                    page_index
                ].markdown[0:references_start[1]]
                continue

            # Last page with references
            if page_index == references_end[0]:
                references_markdown += proc_result.pages[page_index].markdown[
                    0:references_end[1]
                ]
                proc_result.pages[page_index].markdown = proc_result.pages[
                    page_index
                ].markdown[references_end[1]:]
                continue

            # Middle pages (contain only references)
            references_markdown += proc_result.pages[page_index].markdown
            proc_result.pages[page_index].markdown = ""

        return proc_result, references_markdown
