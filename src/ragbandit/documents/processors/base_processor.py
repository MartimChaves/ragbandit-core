import logging
from abc import ABC, abstractmethod
from document_data_models import ExtendedOCRResponse
from document_utils.cost_tracking import TokenUsageTracker


class BaseProcessor(ABC):
    """
    Base class or mix-in for individual processors.
    Subclasses override `process()` and, optionally, `extend_response()`.
    """

    def __init__(self, name: str | None = None):
        # Hierarchical names make it easy to filter later:
        #   pipeline.text_cleaner, pipeline.language_model, …
        base = "pipeline"
        self.logger = logging.getLogger(
            f"{base}.{name or self.__class__.__name__}"
        )

    @abstractmethod
    def process(
        self,
        response: ExtendedOCRResponse,
        usage_tracker: TokenUsageTracker,
    ) -> tuple[ExtendedOCRResponse, dict[str, any]]:
        """
        Do one step of work and return:
          * a (possibly modified) ExtendedOCRResponse
          * a dict of metadata specific to this processor
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
