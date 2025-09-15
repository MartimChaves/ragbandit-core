from ragbandit.documents.ocr.base_ocr import BaseOCR
from ragbandit.schema import (
    OCRResult, OCRPage, PageDimensions, Image, OCRUsageInfo
)
from datetime import datetime

import logging
from ragbandit.utils import mistral_client_manager


class MistralOCRDocument(BaseOCR):
    """OCR document processor using Mistral's API."""

    def __init__(self, api_key: str, logger: logging.Logger = None, **kwargs):
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

    def process(self, pdf_filepath: str, encrypted: bool = False) -> OCRResult:
        """Process a PDF file through Mistral's OCR API.

        Args:
            pdf_filepath: Path to the PDF file to process
            encrypted: Whether the file is encrypted (default: True)

        Returns:
            OCRResult: The OCR result conforming to the schema
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

        # Transform the Mistral OCRResponse into the OCRResult schema
        pages = []
        for i, page in enumerate(ocr_response.pages):
            images = []
            if page.images:
                for img in page.images:
                    images.append(Image.model_validate(img))

            ocr_page = OCRPage(
                index=i,
                markdown=page.markdown,
                images=images,
                dimensions=PageDimensions.model_validate(page.dimensions),
            )
            pages.append(ocr_page)

        usage_info = OCRUsageInfo.model_validate(ocr_response.usage_info)

        result = OCRResult(
            source_file_path=pdf_filepath,
            processed_at=datetime.now(),
            model=ocr_response.model,
            document_annotation=ocr_response.document_annotation,
            pages=pages,
            usage_info=usage_info,
        )

        return result
