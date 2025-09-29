"""
Footnote processor for detecting, processing,
and handling footnotes in documents.

This processor identifies footnotes in a document, categorizes them as either
references or explanations, and processes them accordingly:
- Explanation footnotes are inlined where they are referenced
- Citation/reference footnotes are collected and returned separately
"""

from difflib import SequenceMatcher

from ragbandit.documents.processors.base_processor import BaseProcessor
from ragbandit.utils.token_usage_tracker import TokenUsageTracker

from ragbandit.prompt_tools.footnotes_processor_tools import (
    detect_footnote_section_tool,
    FootnoteStart,
    detect_footnote_start_tool,
    classify_footnote_tool,
    replace_footnote_inline_operation
)
from ragbandit.schema import (
    OCRResult,
    ProcessingResult,
)


class FootnoteProcessor(BaseProcessor):
    """Processor for detecting and handling footnotes in documents.

    This processor:
    1. Detects footnote sections at the bottom of each page
    2. Processes each footnote to determine if it's a citation or explanation
    3. Inlines explanation footnotes where they are referenced
    4. Collects citation footnotes for inclusion in references
    5. Returns the modified document and the extracted footnote references
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
        """Process OCR pages to detect and handle footnotes.

        Args:
            document: OCR response or ProcessingResult to process
            usage_tracker: Token usage tracker for LLM calls

        Returns:
            Tuple containing:
            - Modified ProcessingResult with footnotes processed
            - Dictionary of footnote references by page
        """
        # Normalise input to ProcessingResult once, then delegate
        proc_input = self.ensure_processing_result(
                        document, processor_name=str(self)
                    )

        proc_result, footnote_refs = self.process_footnotes(
            proc_input, usage_tracker
        )

        # Embed footnote references into extracted_data for downstream use
        if footnote_refs:
            if proc_result.extracted_data is None:
                proc_result.extracted_data = {}

            proc_result.extracted_data["footnote_refs"] = footnote_refs

        return proc_result

    def process_footnotes(
        self,
        proc_result: ProcessingResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> tuple[ProcessingResult, dict]:
        """Process footnotes in document pages.

        This method identifies footnote sections in each page, processes them,
        and handles them based on their category (explanation or citation).

        Args:
            proc_result: The document to process (already a ProcessingResult)
            usage_tracker: Optional tracker for token usage in LLM calls

        Returns:
            Tuple containing:
            - Modified ProcessingResult with footnotes processed
            - Dictionary of footnote references by page
        """
        footnote_sections: dict[int, str] = {}
        for page in proc_result.pages:
            page_footnote_section = detect_footnote_section_tool(
                api_key=self.api_key,
                ocr_response_page=page.markdown,
                usage_tracker=usage_tracker
            )
            footnote_sections[page.index] = (
                page_footnote_section.footnote_section
            )

        # Clean up footnote sections
        footnote_sections = self._clean_footnote_sections(footnote_sections)

        # Split footnote sections into individual footnotes
        footnotes_listed = self._split_footnote_sections(footnote_sections)

        # Process and categorize each footnote
        footnotes_explained = self._categorize_footnotes(
            footnotes_listed, usage_tracker
        )

        # Process footnotes based on their category and update document
        proc_result, footnote_refs = self._process_footnotes_by_category(
            proc_result, footnotes_explained, footnotes_listed, usage_tracker
        )

        return proc_result, footnote_refs

    def _clean_footnote_sections(self, footnote_sections: dict) -> None:
        """Clean up footnote sections by removing common junk characters.

        Args:
            footnote_sections: Dictionary mapping page index
                               to footnote section text
        """
        # Remove commonly occurring junk char [^0]
        remove_char = "[^0]:"
        remove_char_len = len(remove_char)

        for page_index, footnote in footnote_sections.items():
            if footnote:
                remove_char_index = footnote.find(remove_char)
                if remove_char_index >= 0:
                    footnote = (
                        footnote[0:remove_char_index]
                        + footnote[(remove_char_index + remove_char_len):]
                    )
                    footnote_sections[page_index] = footnote

        # Delete footnote sections without actual footnotes
        page_index_no_footnotes = [
            page_index
            for page_index in footnote_sections
            if len(footnote_sections[page_index]) == 0
        ]
        for page_index in page_index_no_footnotes:
            del footnote_sections[page_index]

        return footnote_sections

    def _split_footnote_sections(self, footnote_sections: dict) -> dict:
        """Split footnote sections into individual footnotes.

        Args:
            footnote_sections: Dictionary mapping page index
                               to footnote section text

        Returns:
            Dictionary mapping page index to list of individual footnotes
        """
        footnotes_listed = {}
        for page_index in footnote_sections:
            # Split footnote section into list
            footnotes_list = footnote_sections[page_index].split("\n")
            clean_footnote_list = []
            for footnote in footnotes_list:
                # Remove extra spaces and newlines
                stripped_footnote = footnote.strip()
                # If footnote less than 5 characters,
                # assume it's formatting junk
                # And only include footnotes longer than 5 characters
                if len(stripped_footnote) > 5:
                    clean_footnote_list.append(stripped_footnote)

            if clean_footnote_list:
                # Use clean footnote list
                footnotes_listed[page_index] = clean_footnote_list
            else:
                # No clean footnote list available
                # Assume cleaning process possibly removed vital information
                # If footnotes are longer than 5 chars, include them
                safe_footnote_list = []
                for footnote in footnotes_list:
                    if len(footnote) > 5:
                        safe_footnote_list.append(footnote)
                if safe_footnote_list:
                    footnotes_listed[page_index] = safe_footnote_list

        return footnotes_listed

    def _get_footnote_symbol(
        self,
        footnote_start: FootnoteStart,
        footnote: str,
    ) -> tuple[str, str]:
        fn_start = footnote_start.footnote_start

        footnote_start_index = footnote.find(fn_start)
        footnote_symbol = footnote[0:footnote_start_index].strip()
        footnote_text = footnote[footnote_start_index:].strip()

        return footnote_symbol, footnote_text

    def _categorize_footnotes(
        self,
        footnotes_listed: dict,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> dict:
        """Categorize footnotes as citations or explanations.

        Args:
            footnotes_listed: Dictionary mapping page index
                              to list of footnotes
            usage_tracker: Optional tracker for token usage in LLM calls

        Returns:
            Dictionary mapping page index to list of categorized footnotes
        """
        footnotes_explained = {}
        for page_index in footnotes_listed:
            footnotes_explained[page_index] = []
            for footnote in footnotes_listed[page_index]:
                # Get footnote symbol and text
                footnote_start = detect_footnote_start_tool(
                    api_key=self.api_key,
                    footnote=footnote,
                    usage_tracker=usage_tracker
                )
                footnote_symbol, footnote_text = self._get_footnote_symbol(
                    footnote_start, footnote
                )

                # Based on the text, classify the footnote
                # Depending on the text the footnote can either be
                # another reference OR an explanation.
                # We assume that explanations are to be inlined,
                # and references added to the references
                footnote_classification = classify_footnote_tool(
                    api_key=self.api_key,
                    footnote_text=footnote_text,
                    usage_tracker=usage_tracker
                )
                footnotes_explained[page_index].append(
                    {
                        "footnote_symbol": footnote_symbol,
                        "footnote_text": footnote_text,
                        "category": footnote_classification.category.value,
                    }
                )

        return footnotes_explained

    def _process_footnotes_by_category(
        self,
        proc_result: ProcessingResult,
        footnotes_explained: dict,
        footnotes_listed: dict,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> tuple[ProcessingResult, dict]:
        """Process footnotes based on their category and update document.

        Args:
            proc_result: ProcessingResult containing document pages
            footnotes_explained: Dictionary mapping page index to
                                 categorized footnotes
            footnotes_listed: Dictionary mapping page index to
                              original footnote text

        Returns:
            Tuple containing:
            - Modified ProcessingResult with footnotes processed
            - Dictionary of footnote references by page
        """
        footnote_refs: dict[int, list[dict]] = {}
        for page_index in footnotes_explained:
            page_markdown = proc_result.pages[page_index].markdown
            for footnote in footnotes_explained[page_index]:
                footnote_category = footnote.get("category", "")
                # If footnote is a citation, add it to footnote refs
                if (
                    footnote_category == "link"
                    or footnote_category == "citation"
                ):
                    footnote_ref = {
                        "symbol": footnote.get("footnote_symbol", ""),
                        "text": footnote.get("footnote_text", ""),
                    }
                    if page_index not in footnote_refs.keys():
                        footnote_refs[page_index] = []
                    footnote_refs[page_index].append(footnote_ref)
                else:
                    # If footnote is an explanation,
                    # inline it where the footnote is called
                    page_markdown = replace_footnote_inline_operation(
                        self.api_key,
                        footnote,
                        page_markdown,
                        usage_tracker
                    )

            # Delete footnote sections
            # Use footnotes as line for closer match,
            # instead of processed footnote
            for footnote_as_line in footnotes_listed[page_index]:
                page_markdown = self._remove_footnotes_by_line(
                    page_markdown, footnote_as_line
                )
            proc_result.pages[page_index].markdown = page_markdown

        return proc_result, footnote_refs

    def _remove_footnotes_by_line(
        self, markdown: str, target_header: str, threshold=0.95
    ) -> str:
        """Remove footnote lines from markdown text.

        Args:
            markdown: The markdown text to process
            target_header: The footnote line to remove
            threshold: Similarity threshold for matching lines

        Returns:
            Updated markdown with footnote lines removed
        """
        lines = markdown.splitlines()
        # Normalize the target header line.
        target_line = target_header.strip()

        for i, line in enumerate(lines):
            # Compare each line after stripping extra whitespace.
            clean_line = line.replace("[^0]:", "").replace("[^0]", "").strip()
            if (
                SequenceMatcher(None, clean_line, target_line).ratio()
                >= threshold
            ):
                # Remove the matched line from the markdown
                line_start_index = markdown.find(line)
                markdown = (
                    markdown[0:line_start_index]
                    + markdown[(line_start_index + len(line)):]
                )
        return markdown
