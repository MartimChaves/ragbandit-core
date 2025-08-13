from ragbandit.prompt_tools.prompt_tool import create_prompt_tool
from pydantic import BaseModel


class SemanticBreak(BaseModel):
    semantic_break: str


semantic_break_tool_prompt = (
    "EXAMPLE TEXT:\n"
    "Once upon a time in a faraway land, a brave knight set forth "
    "on a quest to rescue the princess. \n"
    "He traveled through forests and mountains, encountering "
    "strange creatures along the way. \n"
    "Finally, he reached the dragon's lair. "
    "![img-0.jpeg](img-0.jpeg)(Image description: A large dragon "
    "perched on a rocky ledge.)\n"
    "The knight prepared for battle, sword in hand.\n"
    "\n"
    "Instruction:\n"
    "1. We want to split the text into coherent chunks. "
    "The first chunk begins at the start of the text.\n"
    "2. Identify where the next chunk should begin—that is, "
    "find the point at which the first chunk naturally ends "
    "(thematic break), and the second chunk begins.\n"
    "3. Return ONLY a short snippet of text (up to ~30 characters) "
    "that marks the beginning of the next chunk. "
    "For example, if the next chunk starts at the word 'Finally,' "
    "return 'Finally, he reached the dragon's lair.' "
    "(truncated if necessary).\n"
    "4. If the entire text above is just one cohesive chunk "
    "with no good break, return \"NO_BREAK\".\n"
    "5. Do not split inside any "
    "![img-0.jpeg](img-0.jpeg)(Image description: ...) text. "
    "Keep these intact.\n"
    "6. Do not output any additional commentary—"
    "just provide the snippet or NO_BREAK.\n"
    "7. Your output should be a JSON containing "
    "a single key 'semantic_break' with the text snippet "
    "for the semantic break.\n"
    "Now find the next semantic break in this text:\n"
    "{{text}}\n"
)


def return_break_string(result: SemanticBreak) -> str:
    return result.semantic_break


find_semantic_break_tool = create_prompt_tool(
    template=semantic_break_tool_prompt,
    output_schema=SemanticBreak,
    model="mistral-small-latest",
    temperature=0,
    postprocess_fn=return_break_string
)
