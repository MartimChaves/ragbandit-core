import os
import time
import requests
import logging
from pathlib import Path
from datetime import datetime
from ragbandit.documents.ocr.base_ocr import BaseOCR
from datalab_sdk.models import ConversionResult
from ragbandit.schema import (
    OCRResult,
    OCRPage,
    OCRUsageInfo,
    PageDimensions,
    Image,
    PagesProcessedMetrics,
)


class DatalabOCR(BaseOCR):
    """OCR component using Datalab API that returns OCRResult schema."""

    # Valid model names for Datalab OCR
    VALID_MODELS = [
        "marker",  # Currently the only available model
    ]

    # Valid mode names for Datalab OCR
    VALID_MODES = [
        "fast",
        "balanced",
        "accurate",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://www.datalab.to/api/v1",
        model: str = "marker",
        mode: str = "balanced",
        output_format: str = "markdown",
        max_pages: int | None = None,
        page_range: str | None = None,
        disable_image_extraction: bool = False,
        disable_image_captions: bool = False,
        poll_interval: int = 2,
        max_polls: int = 300,
        logger: logging.Logger = None,
        **kwargs
    ):
        """
        Initialize the Datalab OCR processor.

        Args:
            api_key: Datalab API key (or set DATALAB_API_KEY env var)
            base_url: Base URL for Datalab API
            model: OCR model to use (must be in VALID_MODELS)
            mode: Processing mode (fast, balanced, accurate)
            output_format: Output format (markdown, html, json, chunks)
            max_pages: Maximum pages to process
            page_range: Specific pages (e.g., "0-5,10", 0-indexed)
            disable_image_extraction: Don't extract images
            disable_image_captions: Don't generate image captions
            poll_interval: Seconds between status checks
            max_polls: Maximum number of polling attempts
            logger: Optional logger for OCR events
            **kwargs: Additional keyword arguments

        Raises:
            ValueError: If model is not in VALID_MODELS
            ValueError: If mode is not in VALID_MODES
        """
        if model not in self.VALID_MODELS:
            raise ValueError(
                f"Invalid model '{model}'. "
                f"Must be one of: {', '.join(self.VALID_MODELS)}"
            )

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. "
                f"Must be one of: {', '.join(self.VALID_MODES)}"
            )

        super().__init__(logger=logger, **kwargs)

        self.model = model
        self.api_key = api_key or os.getenv("DATALAB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in "
                "DATALAB_API_KEY environment variable"
            )
        self.base_url = base_url
        self.mode = mode
        self.output_format = output_format
        self.max_pages = max_pages
        self.page_range = page_range
        self.disable_image_extraction = disable_image_extraction
        self.disable_image_captions = disable_image_captions
        self.poll_interval = poll_interval
        self.max_polls = max_polls
        self.headers = {"X-API-Key": self.api_key}

    # ----------------- Helper methods ----------------- #

    def _build_request_data(self) -> dict:
        """Build the request data dictionary for API call."""
        data = {
            "output_format": self.output_format,
            "mode": self.mode,
            "paginate": True,
        }

        if self.max_pages is not None:
            data["max_pages"] = self.max_pages
        if self.page_range is not None:
            data["page_range"] = self.page_range
        if self.disable_image_extraction:
            data["disable_image_extraction"] = True
        if self.disable_image_captions:
            data["disable_image_captions"] = True

        return data

    def _submit_ocr_job(
        self, file_path: str, data: dict
    ) -> str:
        """Submit OCR job to Datalab API and return check URL."""
        url = f"{self.base_url}/marker"
        file_path_obj = Path(file_path)

        with open(file_path_obj, "rb") as f:
            response = requests.post(
                url,
                files={
                    "file": (
                        file_path_obj.name,
                        f,
                        "application/pdf",
                    )
                },
                data=data,
                headers=self.headers,
            )

        response.raise_for_status()
        result = response.json()
        return result["request_check_url"]

    def _poll_for_completion(
        self, check_url: str, poll_interval: int = 2, max_polls: int = 300
    ) -> ConversionResult:
        """Poll the check URL until OCR job completes.

        Returns:
            ConversionResult: The API response with OCR data
        """

        for _ in range(max_polls):
            response = requests.get(check_url, headers=self.headers)
            response.raise_for_status()
            result = response.json()

            status = result.get("status")
            if status == "complete":
                # Convert dict to ConversionResult object for strong typing
                # Get fields from dir() and manually add instance attributes
                valid_fields = [
                    attr for attr in dir(ConversionResult)
                    if not attr.startswith('_')
                ]
                # Add instance attributes that don't show up in dir()
                valid_fields.extend(['success', 'output_format'])

                filtered_result = {
                    k: v for k, v in result.items() if k in valid_fields
                }
                return ConversionResult(**filtered_result)
            elif status == "error":
                error_msg = result.get("error", "Unknown error")
                raise RuntimeError(f"OCR processing failed: {error_msg}")

            time.sleep(poll_interval)

        timeout_secs = max_polls * poll_interval
        raise TimeoutError(
            f"OCR processing timed out after {timeout_secs} seconds"
        )

    def _convert_pages(self, api_result: ConversionResult) -> list[OCRPage]:
        """Convert Datalab API response pages to OCRPage schema."""
        import re

        pages = []
        markdown_content = api_result.markdown

        if (
            api_result.output_format == "markdown"
            and markdown_content
        ):
            # Split by page delimiter: {page_number}----...
            page_pattern = r'\{(\d+)\}-{40,}'
            page_splits = re.split(page_pattern, markdown_content)

            # Remove empty first element if content starts with delimiter
            if page_splits and not page_splits[0].strip():
                page_splits = page_splits[1:]

            # Get images dict once (it's the same for all pages)
            images_dict = api_result.images or {}

            # Process pairs of (page_number, content)
            for i in range(0, len(page_splits), 2):
                if i + 1 >= len(page_splits):
                    break

                idx = int(page_splits[i])
                page_content = page_splits[i + 1].strip()
                images_list = []

                # Images are a dict: {filename: base64_string}
                # Only include images referenced in this page's content
                for img_key, img_base64 in images_dict.items():
                    # Check if this image key appears in the page content
                    if img_key in page_content:
                        # Add data URI prefix for browser compatibility
                        # Assume JPEG format (most common for PDFs)
                        if not img_base64.startswith('data:'):
                            img_base64 = f"data:image/jpeg;base64,{img_base64}"

                        images_list.append(
                            Image(
                                id=img_key,
                                top_left_x=None,
                                top_left_y=None,
                                bottom_right_x=None,
                                bottom_right_y=None,
                                image_base64=img_base64,
                                image_annotation=None,
                            )
                        )

                # Note: Datalab API doesn't provide page dimensions
                # metadata only contains page_stats with page_id and num_blocks
                # Using 0 to indicate unknown/not provided
                dimensions = PageDimensions(
                    dpi=0,
                    height=0,
                    width=0,
                )

                pages.append(
                    OCRPage(
                        index=idx,
                        markdown=page_content,
                        images=images_list,
                        dimensions=dimensions,
                    )
                )

        return pages

    def _build_usage_info(
        self, api_result: ConversionResult, source_file_path: str
    ) -> OCRUsageInfo:
        """Extract usage information from API response."""
        page_count = api_result.page_count or 0

        doc_size_bytes = 0
        if Path(source_file_path).exists():
            doc_size_bytes = Path(source_file_path).stat().st_size

        return OCRUsageInfo(
            pages_processed=page_count, doc_size_bytes=doc_size_bytes
        )

    def _build_metrics(
        self, api_result: ConversionResult
    ) -> list[PagesProcessedMetrics]:
        """Create page-processing cost metrics from API response."""
        page_count = api_result.page_count or 0

        total_cost = 0.0
        if api_result.cost_breakdown:
            cost_data = api_result.cost_breakdown
            if isinstance(cost_data, dict):
                total_cost = cost_data.get("final_cost_cents", 0.0) / 100

        cost_per_page = total_cost / page_count if page_count > 0 else 0.0

        return [
            PagesProcessedMetrics(
                pages_processed=page_count,
                cost_per_page=cost_per_page,
                total_cost_usd=total_cost,
            )
        ]

    def get_config(self) -> dict:
        """Return the configuration for this OCR component.

        Returns:
            dict: Configuration dictionary
        """
        return {
            "model": self.model,
            "mode": self.mode,
            "max_pages": self.max_pages,
            "page_range": self.page_range,
            "disable_image_extraction": self.disable_image_extraction,
            "disable_image_captions": self.disable_image_captions,
        }

    def _build_result(
        self,
        pdf_filepath: str,
        pages: list[OCRPage],
        usage_info: OCRUsageInfo,
        metrics: list[PagesProcessedMetrics],
    ) -> OCRResult:
        """Assemble the OCRResult object."""

        return OCRResult(
            component_name=self.get_name(),
            component_config=self.get_config(),
            source_file_path=pdf_filepath,
            processed_at=datetime.now(),
            model=self.model,
            document_annotation=None,
            pages=pages,
            usage_info=usage_info,
            metrics=metrics,
        )

    def process(
        self,
        pdf_filepath: str,
    ) -> OCRResult:
        """High-level orchestration for running Datalab OCR on a PDF.

        Args:
            pdf_filepath: Path to the PDF file to OCR

        Returns:
            OCRResult: The OCR result from Datalab
        """
        _, reader = self.validate_and_prepare_file(
            pdf_filepath
        )

        try:
            data = self._build_request_data()

            check_url = self._submit_ocr_job(pdf_filepath, data)
            api_result = self._poll_for_completion(
                check_url, self.poll_interval, self.max_polls
            )
        finally:
            del reader

        pages = self._convert_pages(api_result)
        usage_info = self._build_usage_info(api_result, pdf_filepath)
        metrics = self._build_metrics(api_result)

        return self._build_result(
            pdf_filepath=pdf_filepath,
            pages=pages,
            usage_info=usage_info,
            metrics=metrics,
        )
