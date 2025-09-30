from ragbandit.documents.ocr.base_ocr import BaseOCR
from ragbandit.schema import (
    OCRResult, OCRPage, PageDimensions,
    Image, OCRUsageInfo, PagesProcessedMetrics
)
from ragbandit.config.pricing import OCR_MODEL_COSTS
from mistralai.models.ocrresponse import OCRResponse
from io import BufferedReader
from datetime import datetime

import logging
from ragbandit.utils import mistral_client_manager


class MistralOCRDocument(BaseOCR):
    """OCR document processor using Mistral's API."""

    # Explicit model identifier used for all Mistral OCR requests
    MODEL_NAME = "mistral-ocr-latest"

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

    # ----------------- Helper methods ----------------- #

    def _upload_file(self, file_name: str, content: BufferedReader):
        """Upload PDF to Mistral Cloud and return the uploaded file object."""
        return self.client.files.upload(
            file={"file_name": file_name, "content": content},
            purpose="ocr",
        )

    def _get_signed_url(self, file_id: str) -> str:
        """Retrieve a signed URL for the previously uploaded file."""
        return self.client.files.get_signed_url(file_id=file_id).url

    def _run_ocr(self, document_url: str) -> OCRResponse:
        """Run Mistral OCR on the provided document URL."""
        return self.client.ocr.process(
            model=self.MODEL_NAME,
            document={"type": "document_url", "document_url": document_url},
            include_image_base64=True,
        )

    def _delete_file_with_retries(
        self, file_id: str, max_tries: int = 10
    ) -> None:
        """Attempt to delete a file from Mistral Cloud with retries."""
        while max_tries > 0:
            resp = self.client.files.delete(file_id=file_id)
            if resp.deleted:
                self.logger.info("File deletion successful!")
                return
            max_tries -= 1
        self.logger.error(f"Deleting unsuccessful. ID: {file_id}")

    def _convert_pages(self, ocr_response: OCRResponse) -> list[OCRPage]:
        """Convert OCRResponse pages to internal OCRPage schema."""
        pages = []
        for i, page in enumerate(ocr_response.pages):
            images = [
                Image.model_validate(img, from_attributes=True)
                for img in (page.images or [])
            ]
            ocr_page = OCRPage(
                index=i,
                markdown=page.markdown,
                images=images,
                dimensions=PageDimensions.model_validate(
                    page.dimensions, from_attributes=True
                ),
            )
            pages.append(ocr_page)
        return pages

    def _build_usage_info(self, ocr_response: OCRResponse) -> OCRUsageInfo:
        """Extract usage information from the OCR response."""
        return OCRUsageInfo.model_validate(
            ocr_response.usage_info, from_attributes=True
        )

    def _build_metrics(self, pages: list[OCRPage]) -> PagesProcessedMetrics:
        """Create page-processing cost metrics."""
        cost_per_page = OCR_MODEL_COSTS.get(self.MODEL_NAME, 0.0)
        return PagesProcessedMetrics(
            pages_processed=len(pages),
            cost_per_page=cost_per_page,
            total_cost_usd=len(pages) * cost_per_page,
        )

    def _build_result(
        self,
        pdf_filepath: str,
        ocr_response: OCRResponse,
        pages: list[OCRPage],
        usage_info: OCRUsageInfo,
        metrics: list[PagesProcessedMetrics],
    ) -> OCRResult:
        """Assemble the OCRResult object."""
        return OCRResult(
            source_file_path=pdf_filepath,
            processed_at=datetime.now(),
            model=ocr_response.model,
            document_annotation=ocr_response.document_annotation,
            pages=pages,
            usage_info=usage_info,
            metrics=metrics,
        )

    def process(self, pdf_filepath: str, encrypted: bool = False) -> OCRResult:
        """High-level orchestration for running Mistral OCR on a PDF."""

        file_name, reader = self.validate_and_prepare_file(
                                pdf_filepath, encrypted
                            )

        uploaded = self._upload_file(file_name, reader)
        del reader  # free memory

        try:
            doc_url = self._get_signed_url(uploaded.id)
            ocr_resp = self._run_ocr(doc_url)
        finally:
            self._delete_file_with_retries(uploaded.id)

        pages = self._convert_pages(ocr_resp)
        usage_info = self._build_usage_info(ocr_resp)
        metrics = [self._build_metrics(pages)]

        return self._build_result(
            pdf_filepath=pdf_filepath,
            ocr_response=ocr_resp,
            pages=pages,
            usage_info=usage_info,
            metrics=metrics,
        )
