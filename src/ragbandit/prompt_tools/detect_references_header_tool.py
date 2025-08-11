from ragbandit.prompt_tools.prompt_tool import create_prompt_tool
from pydantic import BaseModel


class ReferencesHeader(BaseModel):
    references_header: str


references_tool_prompt = (
    "You are an expert at identifying the references section "
    "of a document. You will be given a list of headers. "
    "Identify the header that represents the references section "
    "(e.g., 'References', 'Bibliography', 'Sources', etc.). "
    "Return a JSON object with a single key 'references_header' "
    "containing the identified header. "
    "If no references header is found, return an empty string.\n"
    "The available headers are provided below (enclosed in <<< and >>>):\n"
    "<<<\n"
    "{{headers}}"
    "\n>>>"
)
detect_references_header_tool = create_prompt_tool(
    template=references_tool_prompt,
    output_schema=ReferencesHeader,
    model="mistral-medium-latest",
    temperature=0,
    # Optional preprocessing function to join headers
    preprocess_fn=lambda kwargs: {
        "headers": "\n".join(kwargs["headers_list"])
    },
)
