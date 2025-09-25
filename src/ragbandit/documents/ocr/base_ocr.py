import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from io import BytesIO, BufferedReader
from ragbandit.documents.utils.secure_file_handler import SecureFileHandler
from ragbandit.schema import OCRResult


class BaseOCR(ABC):
    """Base class for OCR document processing.

    This class provides the interface for OCR processing and a default
    implementation using Mistral's OCR API.
    """

    def __init__(self, logger: logging.Logger = None, **kwargs):
        """Initialize the OCR processor.

        Args:
            logger: Optional logger for OCR events
            **kwargs: Additional keyword arguments (e.g., encryption_key)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.kwargs = kwargs

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

    def read_encrypted_file(self, pdf_filepath: str) -> BufferedReader:
        """Read an encrypted PDF file and return a buffered reader.

        Args:
            pdf_filepath: Path to the encrypted PDF file

        Returns:
            BufferedReader: A buffered reader for the decrypted file content

        Raises:
            ValueError: If encryption_key is not provided in kwargs
        """
        self.logger.info("Decrypting for OCR...")

        encryption_key = self.kwargs.get("encryption_key")
        if not encryption_key:
            raise ValueError(
                "encryption_key must be provided in kwargs "
                "for encrypted file operations. "
                "Pass encryption_key when initializing the OCR processor."
            )

        secure_handler = SecureFileHandler(encryption_key=encryption_key)
        decrypted = secure_handler.read_encrypted_file(Path(pdf_filepath))
        raw = BytesIO(decrypted)
        raw.seek(0)
        return BufferedReader(raw)

    def read_unencrypted_file(self, pdf_filepath: str) -> BufferedReader:
        """Read an unencrypted PDF file and return a buffered reader.

        Args:
            pdf_filepath: Path to the unencrypted PDF file

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
        self, pdf_filepath: str, encrypted: bool = True
    ) -> tuple[str, BufferedReader]:
        """Validate and prepare a PDF file for OCR processing.

        Args:
            pdf_filepath: Path to the PDF file to process
            encrypted: Whether the file is encrypted (default: True)

        Returns:
            tuple: (file_name, file_reader)

        Raises:
            ValueError: If the file does not exist
        """
        file_name = self.validate_pdf(pdf_filepath)

        if encrypted:
            reader = self.read_encrypted_file(pdf_filepath)
        else:
            reader = self.read_unencrypted_file(pdf_filepath)

        return file_name, reader

    @abstractmethod
    def process(self, pdf_filepath: str) -> OCRResult:
        """Process a PDF file through OCR.

        Args:
            pdf_filepath: Path to the PDF file to process

        Returns:
            OCRResult: The OCR result from the processor
        """
        raise NotImplementedError("Subclasses must implement process method")

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        """Return a string representation of the OCR processor."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return a string representation of the OCR processor."""
        return f"{self.__class__.__name__}()"
