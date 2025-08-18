import logging
from abc import ABC, abstractmethod
from ragbandit.schema import ExtendedOCRResponse
from ragbandit.utils.token_usage_tracker import TokenUsageTracker


class BaseProcessor(ABC):
    """
    Base class or mix-in for individual processors.
    Subclasses override `process()` and, optionally, `extend_response()`.
    """

    def __init__(self, name: str | None = None, api_key: str | None = None):
        """
        Initialize the processor.

        Args:
            name: Optional name for the processor
            api_key: API key for LLM services
        """
        # Hierarchical names make it easy to filter later:
        #   pipeline.text_cleaner, pipeline.language_model, …
        base = "pipeline"
        self.logger = logging.getLogger(
            f"{base}.{name or self.__class__.__name__}"
        )
        self.api_key = api_key

    @abstractmethod
    def process(
        self,
        response: ExtendedOCRResponse,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> tuple[ExtendedOCRResponse, dict[str, any]]:
        """
        Do one step of work and return:
          * a (possibly modified) ExtendedOCRResponse
          * a dict of metadata specific to this processor

        Args:
            response: The OCR response to process
            usage_tracker: Optional token usage tracker
        """
        raise NotImplementedError
    # ----------------------------------------------------------------------

    # default implementation
    def extend_response(
        self,
        response: ExtendedOCRResponse,
        metadata: dict[str, any],
    ) -> None:
        response.processing_metadata.update(metadata)
        return response

    def __str__(self) -> str:
        """Return a string representation of the processor."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return a string representation of the processor."""
        return f"{self.__class__.__name__}()"
