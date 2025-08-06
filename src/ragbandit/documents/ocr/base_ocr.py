import logging
from abc import ABC, abstractmethod
from mistralai import OCRResponse


class BaseOCR(ABC):
    """Base class for OCR document processing.

    This class provides the interface for OCR processing and a default
    implementation using Mistral's OCR API.
    """

    def __init__(self, logger: logging.Logger = None):
        """Initialize the OCR processor.

        Args:
            logger: Optional logger for OCR events
        """
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def process(self, pdf_filepath: str) -> OCRResponse:
        """Process a PDF file through OCR.

        Args:
            pdf_filepath: Path to the PDF file to process

        Returns:
            OCRResponse: The OCR response from the processor
        """
        raise NotImplementedError("Subclasses must implement process method")
