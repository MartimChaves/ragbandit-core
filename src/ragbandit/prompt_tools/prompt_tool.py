"""
Utilities for creating LLM-powered tools based on prompt templates.
"""

from typing import Generic, TypeVar, Callable

from pydantic import BaseModel
from ragbandit.utils.llm_utils import query_llm
from ragbandit.utils.token_usage_tracker import TokenUsageTracker

T = TypeVar("T", bound=BaseModel)


class PromptTool(Generic[T]):
    """A tool that uses a prompt template to query an
    LLM and return structured data."""

    def __init__(
        self,
        template: str,
        output_schema: type[T],
        model: str = "mistral-small-latest",
        temperature: float = 0,
        preprocess_fn: Callable[[dict[str, object]], dict[str, object]] = None,
        postprocess_fn: Callable[[T], object] = None,
    ):
        """Initialize a new prompt-based tool.

        Args:
            template: String template with {variable} placeholders
            output_schema: Pydantic model for response validation
            model: LLM model to use
            temperature: Sampling temperature
            preprocess_fn: Optional function to preprocess variables
                           before formatting
            postprocess_fn: Optional function to process the result
                            after LLM response
        """
        self.template = template
        self.output_schema = output_schema
        self.model = model
        self.temperature = temperature
        self.preprocess_fn = preprocess_fn or (lambda x: x)
        self.postprocess_fn = postprocess_fn or (lambda x: x)

    def format_prompt(self, **kwargs) -> str:
        """Format the template with the provided variables.

        This method handles variable substitution in the template.
        """
        processed_kwargs = self.preprocess_fn(kwargs)

        # Simple string replacement approach - more reliable than format()
        result = self.template
        for key, value in processed_kwargs.items():
            placeholder = "{{" + key + "}}"
            result = result.replace(placeholder, str(value))

        return result

    def __call__(
        self,
        api_key: str,
        usage_tracker: TokenUsageTracker | None = None,
        **kwargs
    ) -> object:
        """Execute the tool with the given variables.

        Args:
            api_key: Mistral API key for authentication
            usage_tracker: Optional token usage tracker
            **kwargs: Variables to substitute in the prompt template

        Returns:
            Processed result from the LLM

        This makes the tool callable like a function, e.g.:
        result = my_tool(api_key="your_api_key", var1="value", var2="value2")
        """
        # Format the prompt with variables
        prompt = self.format_prompt(**kwargs)

        # Query the LLM
        result = query_llm(
            prompt=prompt,
            output_schema=self.output_schema,
            api_key=api_key,
            usage_tracker=usage_tracker,
            model=self.model,
            temperature=self.temperature,
        )

        # Apply any post-processing
        return self.postprocess_fn(result)


# Helper function to create a tool more easily
def create_prompt_tool(
    template: str,
    output_schema: type[T],
    model: str = "mistral-small-latest",
    temperature: float = 0,
    preprocess_fn: Callable[[dict[str, object]], dict[str, object]] = None,
    postprocess_fn: Callable[[T], object] = None,
) -> PromptTool[T]:
    """Create a new prompt-based tool with the given template and schema.

    Note: When calling the returned tool,
    you must provide an api_key parameter.
    """
    return PromptTool(
        template=template,
        output_schema=output_schema,
        model=model,
        temperature=temperature,
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
    )
