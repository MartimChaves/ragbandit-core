"""
Prompt tools for structured LLM interactions.

This module provides tools for creating and using structured prompts with LLMs.
"""

from ragbandit.prompt_tools.prompt_tool import create_prompt_tool
from ragbandit.prompt_tools.footnotes_processor_tools import (
    detect_footnote_section_tool,
    detect_footnote_start_tool,
    classify_footnote_tool,
    footnote_insertion_instruction_tool,
    replace_footnote_inline_operation
)
from ragbandit.prompt_tools.references_processor_tools import (
    detect_references_header_tool
)

__all__ = [
    "create_prompt_tool",
    "detect_footnote_section_tool",
    "detect_footnote_start_tool",
    "classify_footnote_tool",
    "footnote_insertion_instruction_tool",
    "replace_footnote_inline_operation",
    "detect_references_header_tool"
]
