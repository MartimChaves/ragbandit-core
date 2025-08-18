from ragbandit.documents.ocr.base_ocr import BaseOCR

import logging
from mistralai import OCRResponse
from ragbandit.utils import mistral_client_manager


class MistralOCRDocument(BaseOCR):
    """OCR document processor using Mistral's API."""

    def __init__(
        self,
        api_key: str,
        logger: logging.Logger = None,
        **kwargs
    ):
        """
        Initialize the Mistral OCR processor.

        Args:
            api_key: Mistral API key
            logger: Optional logger for OCR events
            **kwargs: Additional keyword arguments
                - encryption_key: Optional key for encrypted file operations
        """
        # Pass all kwargs to the base class
        super().__init__(logger=logger, **kwargs)
        self.client = mistral_client_manager.get_client(api_key)

    def process(
        self, pdf_filepath: str, encrypted: bool = False
    ) -> OCRResponse:
        """Process a PDF file through Mistral's OCR API.

        Args:
            pdf_filepath: Path to the PDF file to process
            encrypted: Whether the file is encrypted (default: True)

        Returns:
            OCRResponse: The OCR response from Mistral
        """
        file_name, reader = self.validate_and_prepare_file(
                                pdf_filepath, encrypted
                            )

        self.logger.info("Creating OCR...")
        uploaded_pdf = self.client.files.upload(
            file={
                "file_name": file_name,
                "content": reader,
            },
            purpose="ocr",
        )

        del reader

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
