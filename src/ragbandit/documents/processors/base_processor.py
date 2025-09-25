import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from ragbandit.schema import OCRResult, ProcessingResult, ProcessedPage
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
        document: OCRResult | ProcessingResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> ProcessingResult:
        """
        Do one step of work and return:
          * a (possibly modified) ProcessingResult
          * a dict of metadata specific to this processor

        Args:
            response: The OCR or intermediate processing result to process.
                This can be either an `OCRResult` (raw OCR output) or
                a `ProcessingResult` (output of a previous processor).
            usage_tracker: Optional token usage tracker
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        """Return a string representation of the processor."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return a string representation of the processor."""
        return f"{self.__class__.__name__}()"

    # ----------------------------------------------------------------------
    # Utility helpers
    def _ensure_processing_result(
        self,
        document: OCRResult | ProcessingResult,
    ) -> ProcessingResult:
        """Ensure the incoming `document` is a `ProcessingResult`.

        If an `OCRResult` is supplied (as is the case for the first processor
        in a pipeline), it will be converted to a shallow `ProcessingResult` so
        that downstream logic can assume a consistent type.
        """

        if isinstance(document, ProcessingResult):
            # Ensure the processor name reflects the current processor
            document.processor_name = str(self)
            return document

        # Convert OCRResult → ProcessingResult (shallow conversion)
        pages_processed = [
            ProcessedPage(**page.model_dump()) for page in document.pages
        ]

        return ProcessingResult(
            processor_name=str(self),
            processed_at=datetime.now(timezone.utc),
            pages=pages_processed,
            processing_trace=[],
            extracted_data={},
            metrics=[],
        )
