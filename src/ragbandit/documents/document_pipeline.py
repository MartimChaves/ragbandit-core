"""
Document processing pipeline that orchestrates multiple document processors.

This module provides the main DocumentPipeline class that manages the execution
of document processors in sequence.
"""
import logging
import traceback
from datetime import datetime, timezone

from mistralai import OCRResponse

from ragbandit.documents.processors.base_processor import BaseProcessor
from ragbandit.utils.token_usage_tracker import TokenUsageTracker

from ragbandit.schema import ExtendedOCRResponse
from ragbandit.utils.in_memory_log_handler import InMemoryLogHandler


class DocumentPipeline:
    """Pipeline for processing documents through a
    sequence of document processors.

    The pipeline manages the execution of document processors in sequence,
    where each
    processor receives the output of the previous processor.
    The pipeline also tracks
    token usage and costs for each document.
    """

    def __init__(
        self,
        processors: list[BaseProcessor] = None,
        logger: logging.Logger = None,
        ocr_processor=None,
    ):
        """Initialize a new document processing pipeline.

        Args:
            processors: List of document processors to execute in sequence
            logger: Optional logger for pipeline events
            ocr_processor: Optional OCR processor to use
                           defaults to MistralOCRDocument)
        """
        if ocr_processor is None:
            raise ValueError(
                "An OCR processor must be explicitly provided"
                " to DocumentPipeline"
                )
        self.ocr_processor = ocr_processor

        self.processors = processors or []

        # Set up logging with more explicit configuration

        self.logger = logger or logging.getLogger(__name__)

        self._transcript = InMemoryLogHandler(level=logging.DEBUG)
        print("Adding InMemoryLogHandler to root logger")
        root_logger = logging.getLogger()
        root_logger.addHandler(self._transcript)

        # Ensure we're generating logs
        self.logger.info("DocumentPipeline initialized")

    def add_processor(self, processor: BaseProcessor) -> None:
        """Add a processor to the pipeline.

        Args:
            processor: The document processor to add
        """
        self.processors.append(processor)
        self.logger.info(f"Added processor: {processor}")

    def _fresh_buffer(self):
        print("Clearing log buffer")
        self._transcript.clear()
        # Ensure the handler is still attached
        root_logger = logging.getLogger()
        if self._transcript not in root_logger.handlers:
            print("Re-adding InMemoryLogHandler to root logger")
            root_logger.addHandler(self._transcript)

    def process(
        self,
        ocr_response: OCRResponse,
        document_id: str,
        metadata: dict[str, any] | None = None,
    ) -> ExtendedOCRResponse:
        """Process a document through the pipeline.

        Args:
            ocr_response: The initial OCR response to process
            document_id: Unique identifier for the document
            metadata: Optional metadata to include in the extended response

        Returns:
            An extended OCR response with additional metadata
            from all processors
        """
        extended_response = None

        try:
            # Create a document-specific usage tracker
            usage_tracker = TokenUsageTracker()

            # Initialize the extended response with the OCR response
            extended_response = ExtendedOCRResponse(
                **ocr_response.model_dump()
            )

            # Add processing metadata
            processing_metadata = metadata or {}
            processing_metadata.update(
                {
                    "document_id": document_id,
                    "processing_started": datetime.now(timezone.utc),
                    "processors": [str(p) for p in self.processors],
                }
            )

            # Add metadata to the extended response
            extended_response.processing_metadata = processing_metadata

            # Process the document through each processor in sequence
            for processor in self.processors:
                self.logger.info(f"Running processor: {processor}")
                try:
                    # Process the document
                    extended_response, processor_metadata = processor.process(
                        extended_response, usage_tracker
                    )

                    # Extend the response with processor-specific metadata
                    extended_response = processor.extend_response(
                        extended_response, processor_metadata
                    )

                    self.logger.info(
                        f"Processor {processor} completed successfully"
                    )
                except Exception as e:
                    error_traceback = traceback.format_exc()
                    self.logger.error(
                        f"Error in processor {processor}: {e}\n"
                        f"Traceback: {error_traceback}\n"
                    )
                    # Continue with the next processor
                    continue

            # Add token usage information to the extended response
            usage_summary = usage_tracker.get_summary()
            extended_response.processing_metadata["token_usage"] = (
                usage_summary
            )

            # Update processing metadata
            now = datetime.now(timezone.utc)
            extended_response.processing_metadata["processing_completed"] = now

            return extended_response
        finally:
            # Save logs
            if extended_response is None:
                extended_response = ExtendedOCRResponse(
                    **ocr_response.model_dump()
                )
            extended_response.processing_metadata["logs"] = (
                self._transcript.dump()
            )
            logging.getLogger().removeHandler(self._transcript)

    def perform_ocr(self, pdf_filepath: str) -> OCRResponse:
        """Perform OCR on a PDF file using the configured OCR processor.

        Args:
            pdf_filepath: Path to the PDF file to process

        Returns:
            OCRResponse: The OCR response from the processor
        """
        return self.ocr_processor.process(pdf_filepath)
