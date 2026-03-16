"""
Table of contents refiner for detecting and removing TOC sections
from documents.

This refiner identifies the TOC header in a document and extracts the TOC
content, removing it from the main document text and storing it in
extracted_data["toc_markdown"].
"""

import re
from difflib import SequenceMatcher
from datetime import datetime, timezone

from ragbandit.documents.refiners.base_refiner import BaseRefiner
from ragbandit.utils.token_usage_tracker import TokenUsageTracker
from ragbandit.prompt_tools.toc_refiner_tools import detect_toc_header_tool
from ragbandit.schema import OCRResult, RefiningResult


class TableOfContentsRefiner(BaseRefiner):
    """Refiner for detecting and removing table of contents sections.

    This refiner:
    1. Extracts headers from the OCR pages
    2. Identifies the TOC section header using an LLM
    3. Removes the TOC section from the document
    4. Returns the modified document and the extracted TOC as markdown

    If the detected TOC header appears in the last 20% of pages, removal
    is skipped (a TOC is always near the start of a document).
    """

    def __init__(self, api_key: str):
        """Initialize the table of contents refiner.

        Args:
            api_key: API key for LLM services.
        """
        super().__init__()
        self.api_key = api_key

    def get_config(self) -> dict:
        """Return the configuration for this refiner.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            "extract_toc": True,
            "remove_from_document": True,
        }

    def process(
        self,
        document: OCRResult | RefiningResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> RefiningResult:
        """Process OCR pages to detect and remove the table of contents.

        Args:
            document: OCRResult or RefiningResult to process.
            usage_tracker: Token usage tracker for LLM calls.

        Returns:
            RefiningResult with the TOC section removed from page markdown
            and stored in ``extracted_data["toc_markdown"]``.
        """
        ref_input = self.ensure_refining_result(
            document, refiner_name=str(self)
        )

        ref_result, toc_markdown = self._remove_toc(ref_input, usage_tracker)

        extracted_data = ref_result.extracted_data or {}
        if toc_markdown:
            extracted_data["toc_markdown"] = toc_markdown

        return RefiningResult(
            component_name=self.get_name(),
            component_config=self.get_config(),
            processed_at=datetime.now(timezone.utc),
            pages=ref_result.pages,
            refining_trace=ref_result.refining_trace,
            extracted_data=extracted_data,
            metrics=usage_tracker.get_summary() if usage_tracker else None,
        )

    def find_best_match(
        self, target: str, string_list: list[str]
    ) -> tuple[str, int]:
        """Find the string in string_list that best matches the target string.

        Uses ``SequenceMatcher`` similarity to rank candidates, which is
        robust to minor formatting differences between the LLM output and
        the actual header text extracted from the document.

        Args:
            target: The string to search for.
            string_list: List of strings to search through.

        Returns:
            A tuple containing (best matching string, index of best match).
            If the list is empty or the target is empty, returns ("", -1).
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

    # ------------------------------------------------------------------
    # Internal helpers

    def _remove_toc(
        self,
        ref_result: RefiningResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> tuple[RefiningResult, str]:
        """Orchestrate TOC detection and extraction.

        Args:
            ref_result: The document to refine.
            usage_tracker: Optional tracker for LLM token usage.

        Returns:
            Tuple of (modified RefiningResult, extracted TOC markdown).
            If no TOC is found, the original document and an empty string
            are returned.
        """
        headers = self._extract_headers(ref_result)
        toc_header, toc_header_index = self._identify_toc_header(
            headers, usage_tracker
        )

        if not toc_header:
            return ref_result, ""

        # The section immediately after the TOC header marks the TOC's end
        next_header = self._find_next_header(
            headers, toc_header_index, ref_result
        )

        boundaries = self._find_toc_boundaries(
            ref_result, toc_header, next_header
        )

        if not boundaries:
            return ref_result, ""

        # Guard: skip if TOC header is in the last 20% of pages
        total_pages = len(ref_result.pages)
        toc_page = boundaries["start"][0]
        if total_pages > 0 and toc_page >= int(total_pages * 0.8):
            self.logger.info(
                f"TOC header found near end of document (page {toc_page} / "
                f"{total_pages}). Skipping removal."
            )
            return ref_result, ""

        return self._extract_toc(ref_result, boundaries)

    def _extract_headers(self, ref_result: RefiningResult) -> list[str]:
        """Extract all markdown headers from the document.

        Args:
            ref_result: RefiningResult containing document pages.

        Returns:
            List of header strings (including the leading ``#`` symbols)
            found across all pages in document order.
        """
        header_regex = re.compile(
            r"(?im)(\s*#{1,6}\s*(?![^\n]*\|)[^\n]+(?:\n|$))"
        )
        full_markdown = "".join(page.markdown for page in ref_result.pages)
        return header_regex.findall(full_markdown)

    def _identify_toc_header(
        self,
        headers: list[str],
        usage_tracker: TokenUsageTracker | None = None,
    ) -> tuple[str, int]:
        """Identify the TOC section header from a list of headers.

        Calls the LLM tool to find the most likely TOC header, then
        uses fuzzy matching to locate the best match in the actual header
        list (tolerating minor formatting differences).

        Args:
            headers: List of headers extracted from the document.
            usage_tracker: Optional tracker for LLM token usage.

        Returns:
            Tuple of (matched header string, index in headers list).
            Returns ("", -1) if no headers are provided.
        """
        if not headers:
            return "", -1

        toc = detect_toc_header_tool(
            api_key=self.api_key,
            usage_tracker=usage_tracker,
            headers_list=headers,
        )
        return self.find_best_match(toc.toc_header, headers)

    def _find_next_header(
        self,
        headers: list[str],
        toc_header_index: int,
        ref_result: RefiningResult,
    ) -> str | None:
        """Return the first header after the TOC that starts real content.

        Unlike the simple ``headers[toc_header_index + 1]`` approach, this
        skips sub-headers that are still part of the TOC structure (e.g.
        ``## General``, ``## Section I``) by checking whether the content
        immediately following each candidate header looks like TOC entries
        (lines with dot-leaders or trailing page numbers).

        Args:
            headers: List of all headers in the document.
            toc_header_index: Index of the identified TOC header.
            ref_result: RefiningResult used to inspect page content.

        Returns:
            The first header whose following content is not TOC-like,
            or None if the TOC runs to the end of the document.
        """
        if toc_header_index < 0:
            return None

        toc_entry_pattern = re.compile(
            r"(?:\.{2,}\s*\d+|\b\d+\s*$)", re.MULTILINE
        )

        for i in range(toc_header_index + 1, len(headers)):
            candidate = headers[i].strip()
            for page in ref_result.pages:
                pos = page.markdown.find(candidate)
                if pos == -1:
                    continue
                after = page.markdown[
                    pos + len(candidate): pos + len(candidate) + 300
                ]
                toc_matches = toc_entry_pattern.findall(after)
                if len(toc_matches) < 3:
                    return headers[i]  # This header starts real content
                break  # Found the header but it's TOC-like; try next header

        return None

    def _find_toc_boundaries(
        self,
        ref_result: RefiningResult,
        toc_header: str,
        next_header: str | None,
    ) -> dict | None:
        """Find the page and character positions that bound the TOC section.

        Headers extracted from concatenated page text may carry leading
        whitespace from page boundaries, so both headers are stripped before
        searching within individual page markdown strings.

        Args:
            ref_result: RefiningResult containing document pages.
            toc_header: The identified TOC section header.
            next_header: The header following the TOC (marks its end),
                or None if the TOC extends to the end of the document.

        Returns:
            Dictionary with keys:
                - ``"start"``: (page_index, char_index) of the TOC header.
                - ``"end"``: (page_index, char_index) of the next header,
                  or None if the TOC runs to the end of the document.
                - ``"toc_header"``: stripped TOC header string.
                - ``"next_header"``: stripped next header string or None.
            Returns None if the TOC header cannot be located in any page.
        """
        toc_page = -1
        next_header_page = -1

        # Strip leading/trailing whitespace so headers captured from
        # concatenated page text (which may have a leading \n) can be
        # matched against individual page markdown strings.
        toc_header_s = toc_header.strip()
        next_header_s = next_header.strip() if next_header else None

        for page in ref_result.pages:
            if toc_header_s in page.markdown:
                toc_page = page.index
            if next_header_s is not None and next_header_s in page.markdown:
                next_header_page = page.index

        if toc_page == -1:
            return None

        toc_page_markdown = ref_result.pages[toc_page].markdown
        toc_start_index = toc_page_markdown.find(toc_header_s)
        toc_start = (toc_page, toc_start_index)

        toc_end = None
        if next_header_s is not None and next_header_page != -1:
            next_header_markdown = ref_result.pages[next_header_page].markdown
            toc_end_index = next_header_markdown.find(next_header_s)
            if toc_end_index != -1:
                toc_end = (next_header_page, toc_end_index)

        return {
            "start": toc_start,
            "end": toc_end,
            "toc_header": toc_header_s,
            "next_header": next_header_s,
        }

    def _extract_toc(
        self, ref_result: RefiningResult, boundaries: dict
    ) -> tuple[RefiningResult, str]:
        """Dispatch to the appropriate extraction method based on boundaries.

        Args:
            ref_result: RefiningResult containing document pages.
            boundaries: Dictionary returned by ``_find_toc_boundaries``.

        Returns:
            Tuple of (modified RefiningResult, extracted TOC markdown).
        """
        toc_start = boundaries["start"]
        toc_end = boundaries["end"]

        if toc_end is None:
            return self._extract_at_end(ref_result, toc_start)

        if toc_end[0] == toc_start[0]:
            return self._extract_same_page(ref_result, toc_start, toc_end)

        return self._extract_multi_page(ref_result, toc_start, toc_end)

    def _extract_at_end(
        self, ref_result: RefiningResult, toc_start: tuple[int, int]
    ) -> tuple[RefiningResult, str]:
        """Extract a TOC that runs to the end of the document.

        All content from ``toc_start`` to the final page is extracted as
        TOC markdown and removed from the document pages.

        Args:
            ref_result: RefiningResult containing document pages.
            toc_start: (page_index, char_index) where the TOC begins.

        Returns:
            Tuple of (modified RefiningResult, extracted TOC markdown).
        """
        toc_markdown = ""
        start_page = True

        for page_index in range(toc_start[0], len(ref_result.pages)):
            if start_page:
                page_md = ref_result.pages[page_index].markdown
                toc_markdown += page_md[toc_start[1]:]
                ref_result.pages[page_index].markdown = (
                    page_md[0: toc_start[1]]
                )
                start_page = False
                continue
            toc_markdown += ref_result.pages[page_index].markdown
            ref_result.pages[page_index].markdown = ""

        return ref_result, toc_markdown

    def _extract_same_page(
        self,
        ref_result: RefiningResult,
        toc_start: tuple[int, int],
        toc_end: tuple[int, int],
    ) -> tuple[RefiningResult, str]:
        """Extract a TOC that begins and ends on the same page.

        Args:
            ref_result: RefiningResult containing document pages.
            toc_start: (page_index, char_index) where the TOC begins.
            toc_end: (page_index, char_index) where the TOC ends.

        Returns:
            Tuple of (modified RefiningResult, extracted TOC markdown).
        """
        page_idx = toc_start[0]
        toc_markdown = ref_result.pages[page_idx].markdown[
            toc_start[1]: toc_end[1]
        ]
        ref_result.pages[page_idx].markdown = (
            ref_result.pages[page_idx].markdown[0: toc_start[1]]
            + ref_result.pages[page_idx].markdown[toc_end[1]:]
        )
        return ref_result, toc_markdown

    def _extract_multi_page(
        self,
        ref_result: RefiningResult,
        toc_start: tuple[int, int],
        toc_end: tuple[int, int],
    ) -> tuple[RefiningResult, str]:
        """Extract a TOC that spans multiple pages.

        The first page is partially cleared from ``toc_start`` onwards,
        intermediate pages are fully cleared, and the last page is cleared
        up to ``toc_end``.

        Args:
            ref_result: RefiningResult containing document pages.
            toc_start: (page_index, char_index) where the TOC begins.
            toc_end: (page_index, char_index) where the TOC ends.

        Returns:
            Tuple of (modified RefiningResult, extracted TOC markdown).
        """
        toc_markdown = ""

        for page_index in range(toc_start[0], toc_end[0] + 1):
            if page_index == toc_start[0]:
                page_md = ref_result.pages[page_index].markdown
                toc_markdown += page_md[toc_start[1]:]
                ref_result.pages[page_index].markdown = (
                    page_md[0: toc_start[1]]
                )
                continue

            if page_index == toc_end[0]:
                page_md = ref_result.pages[page_index].markdown
                toc_markdown += page_md[0: toc_end[1]]
                ref_result.pages[page_index].markdown = page_md[toc_end[1]:]
                continue

            # Middle pages belong entirely to the TOC section
            toc_markdown += ref_result.pages[page_index].markdown
            ref_result.pages[page_index].markdown = ""

        return ref_result, toc_markdown
