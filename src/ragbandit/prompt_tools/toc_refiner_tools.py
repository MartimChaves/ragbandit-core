from ragbandit.prompt_tools.prompt_tool import create_prompt_tool
from pydantic import BaseModel


class TOCHeader(BaseModel):
    toc_header: str


toc_tool_prompt = (
    "You are an expert at identifying the table of contents section "
    "of a document. You will be given a list of headers. "
    "Identify the header that represents the table of contents section "
    "(e.g., 'Table of Contents', 'Contents', 'TOC', etc.). "
    "Return a JSON object with a single key 'toc_header' "
    "containing the identified header. "
    "If no table of contents header is found, return an empty string.\n"
    "The available headers are provided below (enclosed in <<< and >>>):\n"
    "<<<\n"
    "{{headers}}"
    "\n>>>"
)

detect_toc_header_tool = create_prompt_tool(
    template=toc_tool_prompt,
    output_schema=TOCHeader,
    model="mistral-medium-latest",
    temperature=0,
    preprocess_fn=lambda kwargs: {
        "headers": "\n".join(kwargs["headers_list"])
    },
)
