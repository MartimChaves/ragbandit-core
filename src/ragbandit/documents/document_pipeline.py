"""
Document processing pipeline that orchestrates multiple document processors.

This module provides the main DocumentPipeline class that manages the execution
of document processors in sequence, chunking, and embedding.
"""

import logging
import traceback
from datetime import datetime, timezone

from ragbandit.schema import (
    OCRResult,
    ProcessingResult,
    ChunkingResult,
    EmbeddingResult,
    DocumentPipelineResult,
    TimingMetrics,
    StepReport,
    StepStatus,
)

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
        usage_tracker: TokenUsageTracker | None = None,
    ) -> list[ProcessingResult]:
        """Process a document through the processors pipeline.

        Args:
            ocr_result: The initial OCR result to process
            usage_tracker: Optional token usage tracker

        Returns:
            A list of ProcessingResult with additional metadata
            from all processors
        """
        processing_results: list[ProcessingResult] = []

        # Start the processor chain with the raw OCRResult; each processor
        # is responsible for converting it to ProcessingResult if needed.
        prev_result = ocr_result

        try:
            # Process the document through each processor in sequence
            for processor in self.processors:
                self.logger.info(f"Running processor: {processor}")
                try:
                    # Give each processor its own usage tracker
                    proc_usage = TokenUsageTracker()
                    proc_result = processor.process(prev_result, proc_usage)
                    # Attach token usage summary to metrics
                    proc_result.metrics.append(proc_usage.get_summary())

                    processing_results.append(proc_result)
                    prev_result = proc_result
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

            return processing_results
        finally:
            # We don't save logs or remove handlers here since
            # that's handled by the process method
            pass  # No special fallback; let caller handle failures

    def run_chunker(
        self,
        proc_result: ProcessingResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> ChunkingResult:
        """Chunk the document using the configured chunker.

        Args:
            proc_result: The ProcessingResult to chunk
            usage_tracker: Optional token usage tracker

        Returns:
            A ChunkingResult object
        """
        usage_tracker = usage_tracker or TokenUsageTracker()
        self.logger.info(f"Running chunker: {self.chunker}")

        try:
            # Generate chunks via chunker -> returns ChunkingResult
            chunk_result = self.chunker.chunk(proc_result, usage_tracker)
            self.logger.info(
                "Chunking completed, created "
                f"{len(chunk_result.chunks)} chunks"
            )

            return chunk_result
        except Exception as e:
            error_traceback = traceback.format_exc()
            self.logger.error(
                f"Error in chunker {self.chunker}: {e}\n"
                f"Traceback: {error_traceback}\n"
            )
            raise

    def run_embedder(
        self,
        chunk_result: ChunkingResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> EmbeddingResult:
        """Embed chunks using the configured embedder.

        Args:
            chunk_result: The ChunkingResult to embed
            usage_tracker: Optional token usage tracker

        Returns:
            An EmbeddingResult containing embeddings for each chunk
        """
        usage_tracker = usage_tracker or TokenUsageTracker()
        self.logger.info(f"Running embedder: {self.embedder}")

        try:
            embedding_result = self.embedder.embed_chunks(
                chunk_result, usage_tracker
            )
            self.logger.info(
                "Embedding completed successfully for "
                f"{len(embedding_result.chunks_with_embeddings)} chunks"
            )

            return embedding_result
        except Exception as e:
            error_traceback = traceback.format_exc()
            self.logger.error(
                f"Error in embedder {self.embedder}: {e}\n"
                f"Traceback: {error_traceback}\n"
            )
            raise

    def run_ocr(self, pdf_filepath: str) -> OCRResult:
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
        run_chunking: bool = True,
        run_embedding: bool = True,
    ) -> DocumentPipelineResult:
        """Process a document through the complete pipeline.

        This method orchestrates the entire document processing pipeline:
        1. OCR processing
        2. Document processors (footnotes, references, etc.)
        3. Chunking (optional)
        4. Embedding (optional)

        Args:
            pdf_filepath: Path to the PDF file to process
            run_chunking: Whether to run the chunking step
            run_embedding: Whether to run the embedding step

        Returns:
            A DocumentPipelineResult with all processing results

        Raises:
            Exception: If any critical step in the pipeline fails
        """
        # Create trackers and the pipeline result object we will fill
        usage_tracker = TokenUsageTracker()

        dpr = DocumentPipelineResult(
            source_file_path=pdf_filepath,
            processed_at=datetime.now(timezone.utc),
            pipeline_config={
                "processors": [str(p) for p in self.processors],
                "chunker": str(self.chunker) if self.chunker else None,
                "embedder": str(self.embedder) if self.embedder else None,
            },
            timings=TimingMetrics(),
            total_metrics=[],
            step_report=StepReport(),
        )

        try:
            # Step 1: OCR
            self.logger.info("Starting OCR processing.")
            try:
                ocr_result = self.run_ocr(pdf_filepath)
                self.logger.info("OCR processing completed")
                dpr.ocr_result = ocr_result
                dpr.step_report.ocr = StepStatus.success
            except Exception as e:
                self.logger.error(f"OCR processing failed: {e}")
                dpr.step_report.ocr = StepStatus.failed
                return dpr

            # Step 2: Run processors
            self.logger.info("Starting document processors")
            try:
                processing_results = self.run_processors(
                    ocr_result, usage_tracker
                )
                self.logger.info("Document processors completed")
                processing_result = processing_results[-1]
                dpr.processing_result = processing_result
                dpr.step_report.processing = StepStatus.success
            except Exception as e:
                self.logger.error(f"Document processors failed: {e}")
                dpr.step_report.processing = StepStatus.failed
                return dpr

            # Step 3: Chunking (if enabled)
            if run_chunking:
                self.logger.info("Starting document chunking")
                try:
                    chunk_result = self.run_chunker(
                        processing_result, usage_tracker
                    )
                    self.logger.info(
                        "Chunking completed with "
                        f"{len(chunk_result.chunks)} chunks"
                    )
                    dpr.chunking_result = chunk_result
                    dpr.step_report.chunking = StepStatus.success
                except Exception as e:
                    self.logger.error(f"Chunking failed: {e}")
                    dpr.step_report.chunking = StepStatus.failed
                    dpr.chunking_result = None
                    run_embedding = False

            # Step 4: Embedding (if enabled and chunks are available)
            if run_embedding and dpr.chunking_result:
                self.logger.info("Starting chunk embedding")
                try:
                    embedding_result = self.run_embedder(
                        dpr.chunking_result, usage_tracker
                    )
                    self.logger.info("Embedding completed successfully")
                    dpr.embedding_result = embedding_result
                    dpr.step_report.embedding = StepStatus.success
                except Exception as e:
                    self.logger.error(f"Embedding failed: {e}")
                    dpr.step_report.embedding = StepStatus.failed

            dpr.total_metrics.append(usage_tracker.get_summary())

            self.logger.info("Document processing completed for document.")
            dpr.timings.total_duration = usage_tracker.stopwatch_total()
            return dpr

        finally:
            # Capture transcript logs regardless of success or early return
            try:
                logs_dump = self._transcript.dump()
                if getattr(dpr, 'extra', None) is None:
                    dpr.extra = {}
                dpr.extra["logs"] = logs_dump
            finally:
                # Always remove the handler when done
                logging.getLogger().removeHandler(self._transcript)
