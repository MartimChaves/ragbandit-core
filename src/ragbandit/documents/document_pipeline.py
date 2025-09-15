"""
Document processing pipeline that orchestrates multiple document processors.

This module provides the main DocumentPipeline class that manages the execution
of document processors in sequence, chunking, and embedding.
"""

import logging
import traceback
from datetime import datetime, timezone

from ragbandit.schema import OCRResult, ExtendedOCRResponse

from ragbandit.documents.ocr import BaseOCR
from ragbandit.documents.processors.base_processor import BaseProcessor
from ragbandit.documents.chunkers.base_chunker import BaseChunker
from ragbandit.documents.embedders.base_embedder import BaseEmbedder
from ragbandit.utils.token_usage_tracker import TokenUsageTracker

from ragbandit.utils.in_memory_log_handler import InMemoryLogHandler


class DocumentPipeline:
    """Pipeline for processing documents through a
    sequence of document processors, chunkers, and embedders.

    The pipeline manages the execution of document processors in sequence,
    where each processor receives the output of the previous processor.
    The pipeline also tracks token usage and costs for each document.
    """

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        ocr_processor: BaseOCR,
        processors: list[BaseProcessor] = None,
        logger: logging.Logger = None,
    ):
        """Initialize a new document processing pipeline.

        Args:
            chunker: Chunker to use for document chunking
            embedder: Embedder to use for chunk embedding
            ocr_processor: OCR processor to use
            processors: List of document processors to execute in sequence
            logger: Optional logger for pipeline events
        """
        self.ocr_processor = ocr_processor
        self.processors = processors or []
        self.chunker = chunker
        self.embedder = embedder

        # Set up logging with more explicit configuration
        self.logger = logger or logging.getLogger(__name__)

        self._transcript = InMemoryLogHandler(level=logging.DEBUG)
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
        self._transcript.clear()
        # Ensure the handler is still attached
        root_logger = logging.getLogger()
        if self._transcript not in root_logger.handlers:
            root_logger.addHandler(self._transcript)

    def run_processors(
        self,
        ocr_result: OCRResult,
        document_id: str,
        metadata: dict[str, any] | None = None,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> ExtendedOCRResponse:
        """Process a document through the processors pipeline.

        Args:
            ocr_result: The initial OCR result to process
            document_id: Unique identifier for the document
            metadata: Optional metadata to include in the extended response
            usage_tracker: Optional token usage tracker

        Returns:
            An extended OCR response with additional metadata
            from all processors
        """
        extended_response = None

        try:
            # Use provided usage tracker or create a new one
            usage_tracker = usage_tracker or TokenUsageTracker()

            # Initialize the extended response with the OCR response
            extended_response = ExtendedOCRResponse(
                                    **ocr_result.model_dump()
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
            extended_response.processing_metadata["token_usage"] = \
                usage_summary

            # Update processing metadata
            now = datetime.now(timezone.utc)
            extended_response.processing_metadata["processing_completed"] = now

            return extended_response
        finally:
            # We don't save logs or remove handlers here since
            # that's handled by the process method
            if extended_response is None:
                extended_response = ExtendedOCRResponse(
                                        **ocr_result.model_dump()
                                    )

    def run_chunker(
        self,
        extended_response: ExtendedOCRResponse,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> list[dict[str, any]]:
        """Chunk the document using the configured chunker.

        Args:
            extended_response: The extended OCR response to chunk
            usage_tracker: Optional token usage tracker

        Returns:
            A list of chunk dictionaries
        """
        usage_tracker = usage_tracker or TokenUsageTracker()
        self.logger.info(f"Running chunker: {self.chunker}")

        try:
            # Generate initial chunks
            chunks = self.chunker.chunk(extended_response, usage_tracker)
            self.logger.info(
                f"Initial chunking completed, created {len(chunks)} chunks"
            )

            # Process chunks (e.g., merge small chunks)
            chunks = self.chunker.process_chunks(chunks)
            self.logger.info(
                f"Chunk processing completed, final chunk count: {len(chunks)}"
            )

            return chunks
        except Exception as e:
            error_traceback = traceback.format_exc()
            self.logger.error(
                f"Error in chunker {self.chunker}: {e}\n"
                f"Traceback: {error_traceback}\n"
            )
            raise

    def run_embedder(
        self,
        chunks: list[dict[str, any]],
        extended_response: ExtendedOCRResponse,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> ExtendedOCRResponse:
        """Embed chunks using the configured embedder.

        Args:
            chunks: The chunks to embed
            extended_response: The extended OCR response
                               to update with embeddings
            usage_tracker: Optional token usage tracker

        Returns:
            The updated extended OCR response with embeddings
        """
        usage_tracker = usage_tracker or TokenUsageTracker()
        self.logger.info(f"Running embedder: {self.embedder}")

        try:
            chunks_with_embeddings = self.embedder.embed_chunks(
                                        chunks, usage_tracker
                                    )
            self.logger.info(
                "Embedding completed successfully for "
                f"{len(chunks_with_embeddings)} chunks"
            )

            # Extend the response with embedding metadata
            extended_response = self.embedder.extend_response(
                extended_response, chunks_with_embeddings
            )
            return extended_response
        except Exception as e:
            error_traceback = traceback.format_exc()
            self.logger.error(
                f"Error in embedder {self.embedder}: {e}\n"
                f"Traceback: {error_traceback}\n"
            )
            raise

    def perform_ocr(self, pdf_filepath: str) -> OCRResult:
        """Perform OCR on a PDF file using the configured OCR processor.

        Args:
            pdf_filepath: Path to the PDF file to process

        Returns:
            OCRResult: The OCR result from the processor
        """
        return self.ocr_processor.process(pdf_filepath)

    def process(
        self,
        pdf_filepath: str,
        document_id: str,
        metadata: dict[str, any] | None = None,
        run_chunking: bool = True,
        run_embedding: bool = True,
    ) -> ExtendedOCRResponse:
        """Process a document through the complete pipeline.

        This method orchestrates the entire document processing pipeline:
        1. OCR processing
        2. Document processors (footnotes, references, etc.)
        3. Chunking (optional)
        4. Embedding (optional)

        Args:
            pdf_filepath: Path to the PDF file to process
            document_id: Unique identifier for the document
            metadata: Optional metadata to include in the extended response
            run_chunking: Whether to run the chunking step
            run_embedding: Whether to run the embedding step

        Returns:
            An extended OCR response with all processing results

        Raises:
            Exception: If any critical step in the pipeline fails
        """
        # Create a usage tracker for the entire pipeline
        usage_tracker = TokenUsageTracker()
        extended_response = None

        try:
            # Step 1: OCR
            self.logger.info(
                f"Starting OCR processing for document: {document_id}"
            )
            try:
                ocr_result = self.perform_ocr(pdf_filepath)
                self.logger.info("OCR processing completed")
            except Exception as e:
                self.logger.error(f"OCR processing failed: {e}")
                # OCR is a critical step - if it fails, we can't continue
                raise

            # Step 2: Run processors
            self.logger.info("Starting document processors")
            try:
                extended_response = self.run_processors(
                    ocr_result, document_id, metadata, usage_tracker
                )
                self.logger.info("Document processors completed")
            except Exception as e:
                self.logger.error(f"Document processors failed: {e}")
                # If processors fail, we still have the OCR response to return
                extended_response = ExtendedOCRResponse(
                                        **ocr_result.model_dump()
                                    )
                if metadata:
                    extended_response.processing_metadata = metadata
                return extended_response

            # Step 3: Chunking (if enabled)
            chunks = None
            if run_chunking:
                self.logger.info("Starting document chunking")
                try:
                    chunks = self.run_chunker(extended_response, usage_tracker)
                    # Store chunks in metadata
                    if extended_response.processing_metadata is None:
                        extended_response.processing_metadata = {}
                    extended_response.processing_metadata["chunks"] = {
                        "count": len(chunks),
                        "chunker": str(self.chunker),
                    }
                    self.logger.info(
                        f"Chunking completed with {len(chunks)} chunks"
                    )
                except Exception as e:
                    self.logger.error(f"Chunking failed: {e}")
                    # If chunking fails, we can't proceed with embedding
                    # but we can return the processed document
                    if extended_response.processing_metadata is None:
                        extended_response.processing_metadata = {}
                    extended_response.processing_metadata["chunking_error"] = \
                        str(e)
                    run_embedding = False

            # Step 4: Embedding (if enabled and chunks are available)
            if run_embedding and chunks:
                self.logger.info("Starting chunk embedding")
                try:
                    extended_response = self.run_embedder(
                        chunks, extended_response, usage_tracker
                    )
                    self.logger.info("Embedding completed successfully")
                except Exception as e:
                    self.logger.error(f"Embedding failed: {e}")
                    # If embedding fails,
                    # we still have the processed document and chunks
                    if extended_response.processing_metadata is None:
                        extended_response.processing_metadata = {}
                    extended_response\
                        .processing_metadata["embedding_error"] = str(e)

            # Update final token usage
            if extended_response.processing_metadata is None:
                extended_response.processing_metadata = {}
            extended_response.processing_metadata["total_token_usage"] = (
                usage_tracker.get_summary()
            )

            self.logger.info(
                f"Document processing completed for document: {document_id}"
            )
            return extended_response

        finally:
            # Ensure logs are captured even if an exception occurs
            if extended_response and extended_response.processing_metadata:
                extended_response.processing_metadata["logs"] = (
                    self._transcript.dump()
                )
            elif extended_response:
                # Create processing_metadata if it doesn't exist
                extended_response.processing_metadata = {
                    "logs": self._transcript.dump()
                }
            # Always remove the handler when done
            logging.getLogger().removeHandler(self._transcript)
