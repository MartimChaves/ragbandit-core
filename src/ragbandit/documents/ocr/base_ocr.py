import logging
import os
from abc import ABC, abstractmethod
from io import BytesIO, BufferedReader
from ragbandit.schema import OCRResult


class BaseOCR(ABC):
    """Base class for OCR document processing.

    This abstract class defines the interface that all OCR implementations
    must follow. Subclasses should implement the process() method.
    """

    def __init__(self, logger: logging.Logger = None):
        """Initialize the OCR processor.

        Args:
            logger: Optional logger for OCR events
        """
        self.logger = logger or logging.getLogger(__name__)

    def validate_pdf(self, pdf_filepath: str) -> str:
        """Validate that a PDF file exists.

        Args:
            pdf_filepath: Path to the PDF file to validate

        Returns:
            str: The basename of the file

        Raises:
            ValueError: If the file does not exist
        """
        file_name = os.path.basename(pdf_filepath)
        pdf_file_exists = os.path.isfile(pdf_filepath)

        if not pdf_file_exists:
            self.logger.error(f"PDF file {pdf_filepath} not found")
            raise ValueError(f"PDF file {pdf_filepath} not found")

        return file_name

    def read_file(self, pdf_filepath: str) -> BufferedReader:
        """Read a PDF file and return a buffered reader.

        Args:
            pdf_filepath: Path to the PDF file

        Returns:
            BufferedReader: A buffered reader for the file content
        """
        self.logger.info("Reading file for OCR...")
        with open(pdf_filepath, 'rb') as f:
            content = f.read()

        raw = BytesIO(content)
        raw.seek(0)
        return BufferedReader(raw)

    def validate_and_prepare_file(
        self, pdf_filepath: str
    ) -> tuple[str, BufferedReader]:
        """Validate and prepare a PDF file for OCR processing.

        Args:
            pdf_filepath: Path to the PDF file to OCR

        Returns:
            tuple: (file_name, file_reader)

        Raises:
            ValueError: If the file does not exist
        """
        file_name = self.validate_pdf(pdf_filepath)
        reader = self.read_file(pdf_filepath)

        return file_name, reader

    @abstractmethod
    def process(self, pdf_filepath: str) -> OCRResult:
        """Process a PDF file through OCR.

        Args:
            pdf_filepath: Path to the PDF file to OCR

        Returns:
            OCRResult: The OCR result from the OCR processor
        """
        raise NotImplementedError("Subclasses must implement process method")

    @abstractmethod
    def get_config(self) -> dict:
        """Return the configuration for this OCR component.

        Returns:
            dict: Configuration dictionary
        """
        raise NotImplementedError(
            "Subclasses must implement get_config method"
        )

    def get_name(self) -> str:
        """Return the component name.

        Returns:
            str: The class name of this component
        """
        return self.__class__.__name__

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        """Return a string representation of the OCR processor."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return a string representation of the OCR processor."""
        return f"{self.__class__.__name__}()"
