from ragbandit.prompt_tools.prompt_tool import create_prompt_tool
from pydantic import BaseModel
from enum import Enum
from ragbandit.utils.token_usage_tracker import TokenUsageTracker

# Detect Footnote Section Tool
class FootnoteSection(BaseModel):  # noqa
    footnote_section: str


footnote_section_tool_prompt = (
    "You are an expert at identifying the footnotes section of a page. "
    "The footnotes section of a page appears at the bottom of the page "
    "and contains text with notes or references. "
    "Identify the footnote section of the following page of markdown. "
    "Return a JSON object with a 'footnote_section' key containing "
    "a string with the footnotes section. "
    "If there's no footnote section, then return an empty string. "
    "Include all of the text in the footnotes section."
    "Page (enclosed in <<<>>>):\n"
    "<<<\n"
    "{{ocr_response_page}}\n"
    ">>>"
)
detect_footnote_section_tool = create_prompt_tool(
    template=footnote_section_tool_prompt,
    output_schema=FootnoteSection,
    model="mistral-medium-latest",
    temperature=0
)


# Detect Footnote Symbol Tool
class FootnoteStart(BaseModel):
    footnote_start: str


footnote_start_tool_prompt = (
    "You will be given a footnote of a page. "
    "Your task is to extract the first word of the footnote text. "
    "Return a JSON object with a single key: 'footnote_start'. "
    "The value should be the first word of the footnote. "
    "Example:\n"
    "<<<\n"
    r"[{\/12}] This study explores the effects of climate change on marine biodiversity."  # noqa
    ">>>\n\n"
    "Output:\n"
    "{'footnote_start': 'This'}"
    "Footnote (enclosed in <<<>>>):\n"
    "<<<\n"
    "{{footnote}}\n"
    ">>>\n\n"
)
detect_footnote_start_tool = create_prompt_tool(
    template=footnote_start_tool_prompt,
    output_schema=FootnoteStart,
    model="mistral-medium-latest",
    temperature=0,
)


# Classify Footnote Tool
class Label(Enum):
    CITATION = "citation"  # noqa:E221
    EXPLANATION = "explanation"
    LINK = "link"  # noqa:E221
    EDITORIAL = "editorial"  # noqa:E221
    OTHER = "other"  # noqa:E221


class FootnoteLabel(BaseModel):
    category: Label
    reason: str


classify_footnote_tool_prompt = (
    "Classify the following Footnote.\n"
    "Use the following categories:\n"
    "- citation: Contains bibliographic information.\n"
    "- explanation: Provides additional context.\n"
    "- link: Includes URLs or references to online resources.\n"
    "- editorial: Contains subjective remarks or corrections.\n"
    "- other: The footnote does not fit into any of the above.\n"
    "Provide a 'reason' for the chosen 'category'.\n"
    "Here's the expected schema:\n"
    "{'category': [category], 'reason': [reason]} json format.\n"
    "Here's the footnote:\n"
    "<<<\n"
    "{{footnote_text}}\n"
    ">>>\n"
)
classify_footnote_tool = create_prompt_tool(
    template=classify_footnote_tool_prompt,
    output_schema=FootnoteLabel,
    model="mistral-small-latest",
    temperature=0,
)


# Footnote Replacement Tool
class SingleFootnoteChange(BaseModel):
    text_to_replace: str
    replacement_text: str


footnote_insertion_instruction_prompt = (
    "You are a text-cleaning assistant. "
    "We will provide you a markdown and details about a footnote. "
    "You must generate minimal edits to inline that footnote. "
    "Only output a JSON array of a single instruction in the form:\n"
    "{'text_to_replace': str, 'replacement_text': str}\n"
    "Do NOT encapsulate the JSON in a list."
    "Do NOT rewrite lines that do not contain the footnote. "
    "Do NOT provide any other text or commentary.\n\n"
    "Rules:\n"
    "1. Inline the footnote right after the usage text, "
    "replacing the footnote symbol.\n"
    "2. Keep everything else exactly as is.\n"
    "Example:\n"
    "Input: \n"
    "Footnote:\n"
    "- Footnote text: *Hej means hello in swedish\n"
    "Text:\n"
    "<<<"
    "Hej*, said the nice old lady. She was wearing an apron.\n"
    "*Hej means hello in swedish\n"
    ">>>"
    "Output: "
    "{'text_to_replace': 'Hej*', "
    "'replacement_text': 'Hej (Hej means hello in swedish)'},"
    "Now process this text:\n"
    "<<<\n"
    "Footnote:\n"
    "- Footnote text: {{footnote_text}}\n"
    "Text:\n"
    "<<<"
    "{{markdown}}\n"
    ">>>"
)
footnote_insertion_instruction_tool = create_prompt_tool(
    template=footnote_insertion_instruction_prompt,
    output_schema=SingleFootnoteChange,
    model="mistral-small-latest",
    temperature=0,
)


def replace_footnote_inline_operation(
    api_key: str,
    footnote: dict,
    markdown: str,
    usage_tracker: TokenUsageTracker | None = None
) -> str:
    """
    Given a footnote and the page's markdown text,
    perform an inline replacement using a 'diff/instructions' approach
    to ensure no unintended text changes occur.

    Steps:
      1) Prompt the LLM to output structured edit instructions
         (text_to_replace, replacement_text).
      2) Apply those instructions to the original text.

    Args:
        api_key: Mistral API Key
        footnote (dict): {
            'footnote_symbol': '*',
            'footnote_text': 'Corresponding author',
            'usage_text': 'James Andrews*',
            'category': 'other',
            'details': 'Footnote indicating that James Andrews
                        is a corresponding author.'
        }
        markdown str: OCRed page.

    Returns:
        str: The updated text, with the footnote properly inlined
             (and footnote lines removed) without altering other content.
    """

    footnote_symbol = footnote.get("footnote_symbol", "")
    footnote_text = f"{footnote_symbol}{footnote.get('footnote_text', '')}"

    replace_instruction = footnote_insertion_instruction_tool(
        api_key=api_key,
        footnote_text=footnote_text,
        markdown=markdown,
        usage_tracker=usage_tracker
    )
    markdown = markdown.replace(
        replace_instruction.text_to_replace,
        replace_instruction.replacement_text,
    )

    return markdown
