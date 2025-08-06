from ragbandit.documents.ocr.base_ocr import BaseOCR

import logging
import os
from pathlib import Path
from io import BytesIO, BufferedReader
from mistralai import Mistral, OCRResponse
from document_utils.file_encryption import SecureFileHandler


class MistralOCRDocument(BaseOCR):
    """OCR document processor using Mistral's API."""

    def __init__(self, api_key: str = None, logger: logging.Logger = None):
        """Initialize the Mistral OCR processor.

        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env variable)
            logger: Optional logger for OCR events
        """
        super().__init__(logger)
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=api_key)
        self.secure_handler = SecureFileHandler()

    def process(self, pdf_filepath: str) -> OCRResponse:
        """Process a PDF file through Mistral's OCR API.

        Args:
            pdf_filepath: Path to the PDF file to process

        Returns:
            OCRResponse: The OCR response from Mistral
        """
        file_name = os.path.basename(pdf_filepath)
        pdf_file_exists = os.path.isfile(pdf_filepath)

        if not pdf_file_exists:
            self.logger.error(f"PDF file {pdf_filepath} not found")
            raise ValueError(f"PDF file {pdf_filepath} not found")

        self.logger.info("Decrypting for OCR...")
        decrypted = self.secure_handler.read_encrypted_file(Path(pdf_filepath))
        raw = BytesIO(decrypted)
        raw.seek(0)
        reader = BufferedReader(raw)

        self.logger.info("Creating OCR...")
        uploaded_pdf = self.client.files.upload(
            file={
                "file_name": file_name,
                "content": reader,
            },
            purpose="ocr",
        )

        del raw, reader

        signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)
        self.logger.info(
            "File uploaded to Mistral Cloud, making OCR request..."
        )

        ocr_response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            include_image_base64=True,
        )
        self.logger.info("OCR request received, deleting file on cloud...")

        file_deleted = False
        num_tries = 10
        while not file_deleted:
            delete_response = self.client.files.delete(file_id=uploaded_pdf.id)
            file_deleted = delete_response.deleted
            if num_tries == 0:
                self.logger.error(
                    f"Deleting unsuccessful. ID: {uploaded_pdf.id}"
                )
                break
            num_tries -= 1

        if num_tries > 0:
            self.logger.info("File deletion successful!")

        return ocr_response
