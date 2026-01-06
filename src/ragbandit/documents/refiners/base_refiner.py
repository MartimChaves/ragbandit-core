import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from ragbandit.schema import OCRResult, RefiningResult, RefinedPage
from ragbandit.utils.token_usage_tracker import TokenUsageTracker


class BaseRefiner(ABC):
    """
    Base class or mix-in for individual refiners.
    Subclasses override `process()` and, optionally, `extend_response()`.
    """

    def __init__(self, name: str | None = None, api_key: str | None = None):
        """
        Initialize the refiner.

        Args:
            name: Optional name for the refiner
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
        document: OCRResult | RefiningResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> RefiningResult:
        """
        Do one step of work and return:
          * a (possibly modified) RefiningResult
          * a dict of metadata specific to this refiner

        Args:
            response: The OCR or intermediate refining result to process.
                This can be either an `OCRResult` (raw OCR output) or
                a `RefiningResult` (output of a previous refiner).
            usage_tracker: Optional token usage tracker
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        """Return a string representation of the refiner."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return a string representation of the refiner."""
        return f"{self.__class__.__name__}()"

    # ----------------------------------------------------------------------
    # Utility helpers
    @staticmethod
    def ensure_refining_result(
        document: OCRResult | RefiningResult,
        refiner_name: str = "bootstrap",
    ) -> RefiningResult:
        """Ensure the incoming `document` is a `RefiningResult`.

        If an `OCRResult` is supplied (as is the case for the first refiner
        in a pipeline), it will be converted to a shallow `RefiningResult` so
        that downstream logic can assume a consistent type.
        """

        # Always create a fresh RefiningResult so that timestamps, metrics,
        # and extracted data do not roll over between refiners.

        source_pages = document.pages if hasattr(document, "pages") else []

        pages_refined = [
            RefinedPage(**page.model_dump()) for page in source_pages
        ]

        return RefiningResult(
            refiner_name=refiner_name,
            processed_at=datetime.now(timezone.utc),
            pages=pages_refined,
            refining_trace=[],
            extracted_data={},
            metrics=None,
        )
