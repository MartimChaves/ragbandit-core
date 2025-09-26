"""
Document processing pipeline that orchestrates multiple document processors.

This module provides the main DocumentPipeline class that manages the execution
of document processors in sequence, chunking, and embedding.
"""

import logging
import traceback
from datetime import datetime, timezone
import time

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
        # list to hold per-processor timings captured in run_processors()
        self._processing_timings: list[dict[str, float]] = []

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
    ) -> list[ProcessingResult]:
        """Process a document through the processors pipeline.

        Args:
            ocr_result: The initial OCR result to process

        Returns:
            A list of ProcessingResult with additional metadata
            from all processors
        """
        processing_results: list[ProcessingResult] = []
        timings: list[dict[str, float]] = []

        # Start the processor chain with the raw OCRResult; each processor
        # is responsible for converting it to ProcessingResult if needed.
        prev_result = ocr_result

        # Process the document through each processor in sequence
        for processor in self.processors:
            self.logger.info(f"Running processor: {processor}")
            start_step = time.perf_counter()
            try:
                # Give each processor its own usage tracker
                proc_usage = TokenUsageTracker()
                proc_result = processor.process(prev_result, proc_usage)
                # Attach token usage summary to metrics
                proc_result.metrics = proc_usage.get_summary()

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
            finally:
                timings.append(
                    {str(processor): time.perf_counter() - start_step}
                )

        # expose timings to be consumed by the outer process() function
        self._processing_timings = timings
        return processing_results

    def run_chunker(
        self,
        proc_result: ProcessingResult,
    ) -> ChunkingResult:
        """Chunk the document using the configured chunker.

        Args:
            proc_result: The ProcessingResult to chunk

        Returns:
            A ChunkingResult object
        """
        usage_tracker = TokenUsageTracker()
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
    ) -> EmbeddingResult:
        """Embed chunks using the configured embedder.

        Args:
            chunk_result: The ChunkingResult to embed
            usage_tracker: Optional token usage tracker

        Returns:
            An EmbeddingResult containing embeddings for each chunk
        """
        usage_tracker = TokenUsageTracker()
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
    ) -> DocumentPipelineResult:
        """Process a document through the complete pipeline.

        This method orchestrates the entire document processing pipeline:
        1. OCR processing
        2. Document processors (footnotes, references, etc.)
        3. Chunking
        4. Embedding

        Args:
            pdf_filepath: Path to the PDF file to process

        Returns:
            A DocumentPipelineResult with all processing results

        Raises:
            Exception: If any critical step in the pipeline fails
        """
        # Record start of whole pipeline
        start_total = time.perf_counter()
        # Create the pipeline result object we will fill
        dpr = DocumentPipelineResult(
            source_file_path=pdf_filepath,
            processed_at=datetime.now(timezone.utc),
            pipeline_config={
                "ocr": str(self.ocr_processor) if self.ocr_processor else None,
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
            start_ocr = time.perf_counter()
            try:
                ocr_result = self.run_ocr(pdf_filepath)
                self.logger.info("OCR processing completed")
                dpr.ocr_result = ocr_result
                dpr.step_report.ocr = StepStatus.success
                dpr.total_metrics.extend(ocr_result.metrics)
                dpr.timings.ocr = time.perf_counter() - start_ocr
            except Exception as e:
                self.logger.error(f"OCR processing failed: {e}")
                dpr.step_report.ocr = StepStatus.failed
                dpr.timings.ocr = time.perf_counter() - start_ocr
                dpr.timings.total_duration = time.perf_counter() - start_total
                return dpr

            # Step 2: Run processors
            self.logger.info("Starting document processors")
            try:
                processing_results = self.run_processors(ocr_result)
                self.logger.info("Document processors completed")
                dpr.processing_results = processing_results
                dpr.step_report.processing = StepStatus.success
                dpr.total_metrics.extend(
                    pr.metrics for pr in processing_results
                )
                dpr.timings.processing_steps = self._processing_timings
            except Exception as e:
                self.logger.error(f"Document processors failed: {e}")
                dpr.step_report.processing = StepStatus.failed
                dpr.timings.processing_steps = self._processing_timings
                dpr.timings.total_duration = time.perf_counter() - start_total
                return dpr

            # Step 3: Chunking
            self.logger.info("Starting document chunking")
            start_chunk = time.perf_counter()
            try:
                chunk_result = self.run_chunker(processing_results[-1])
                self.logger.info(
                    "Chunking completed with "
                    f"{len(chunk_result.chunks)} chunks"
                )
                dpr.chunking_result = chunk_result
                dpr.step_report.chunking = StepStatus.success
                dpr.total_metrics.extend(chunk_result.metrics)
                dpr.timings.chunking = time.perf_counter() - start_chunk
            except Exception as e:
                self.logger.error(f"Chunking failed: {e}")
                dpr.step_report.chunking = StepStatus.failed
                dpr.timings.chunking = time.perf_counter() - start_chunk
                dpr.timings.total_duration = time.perf_counter() - start_total
                return dpr

            # Step 4: Embedding (execute only if chunks are available)
            self.logger.info("Starting chunk embedding")
            start_embed = time.perf_counter()
            try:
                embedding_result = self.run_embedder(chunk_result)
                self.logger.info("Embedding completed successfully")
                dpr.embedding_result = embedding_result
                dpr.step_report.embedding = StepStatus.success
                dpr.total_metrics.extend(embedding_result.metrics)
                dpr.timings.embedding = time.perf_counter() - start_embed
            except Exception as e:
                self.logger.error(f"Embedding failed: {e}")
                dpr.step_report.embedding = StepStatus.failed
                dpr.timings.embedding = time.perf_counter() - start_embed
                dpr.timings.total_duration = time.perf_counter() - start_total
                return dpr

            # Aggregate total cost across all metrics collected
            dpr.total_cost_usd = sum(
                m.total_cost_usd
                for m in dpr.total_metrics
                if m and getattr(m, "total_cost_usd", None) is not None
            )

            self.logger.info("Document processing completed for document.")
            dpr.timings.total_duration = time.perf_counter() - start_total
            return dpr

        finally:
            # Capture transcript logs regardless of success or early return
            try:
                logs_dump = self._transcript.dump()
                dpr.logs = logs_dump
            finally:
                # Always remove the handler when done
                logging.getLogger().removeHandler(self._transcript)
