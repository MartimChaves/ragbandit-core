"""
Document processing pipeline that orchestrates multiple document processors.

This module provides the main DocumentPipeline class that manages the execution
of document processors in sequence, chunking, and embedding.
"""

import logging
import traceback
from datetime import datetime, timezone
from typing import Callable
import time
from dataclasses import dataclass
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

    @dataclass
    class _PipelineStep:
        key: str  # "ocr" | "processing" | "chunking" | "embedding"
        run: Callable[[], object]
        on_success: Callable[[object], None]

    def __init__(
        self,
        ocr_processor: BaseOCR | None = None,
        processors: list[BaseProcessor] | None = None,
        chunker: BaseChunker | None = None,
        embedder: BaseEmbedder | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize a new document processing pipeline.

        All components are optional to allow running
        individual steps independently.
        For full pipeline execution via process(),
        all components must be provided.

        Args:
            ocr_processor: OCR processor to use (required for run_ocr
                           and process)
            processors: List of document processors to execute in
                        sequence
            chunker: Chunker to use for document chunking (required
                     for run_chunker and process)
            embedder: Embedder to use for chunk embedding (required
                      for run_embedder and process)
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

    def run_ocr(self, pdf_filepath: str) -> OCRResult:
        """Perform OCR on a PDF file using the configured OCR processor.

        Args:
            pdf_filepath: Path to the PDF file to process

        Returns:
            OCRResult: The OCR result from the processor

        Raises:
            ValueError: If ocr_processor is not configured
        """
        if not self.ocr_processor:
            raise ValueError("ocr_processor is required for OCR operation")
        return self.ocr_processor.process(pdf_filepath)

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

        # Start the processor chain with the raw OCRResult; each processor
        # is responsible for converting it to ProcessingResult if needed.
        prev_result = ocr_result

        # Process the document through each processor in sequence
        for processor in self.processors:
            self.logger.info(f"Running processor: {processor}")

            # Give each processor its own usage tracker
            proc_usage = TokenUsageTracker()

            start_processing = time.perf_counter()
            proc_result = processor.process(prev_result, proc_usage)
            end_processing = time.perf_counter()

            # Attach token usage summary to metrics
            proc_result.metrics = proc_usage.get_summary()
            proc_duration = end_processing - start_processing
            proc_result.processing_duration = proc_duration

            processing_results.append(proc_result)
            prev_result = proc_result
            self.logger.info(f"{processor} completed successfully")

        return processing_results

    def run_chunker(
        self,
        doc: ProcessingResult | OCRResult,
    ) -> ChunkingResult:
        """Chunk the document using the configured chunker.

        Args:
            doc: The ProcessingResult or OCRResult to chunk

        Returns:
            A ChunkingResult object

        Raises:
            ValueError: If chunker is not configured
        """
        if not self.chunker:
            raise ValueError("chunker is required for chunking operation")
        proc_result = (
            doc
            if isinstance(doc, ProcessingResult)
            else BaseProcessor.ensure_processing_result(doc)
        )
        usage_tracker = TokenUsageTracker()
        # Generate chunks via chunker -> returns ChunkingResult
        chunk_result = self.chunker.chunk(proc_result, usage_tracker)
        return chunk_result

    def run_embedder(
        self,
        chunk_result: ChunkingResult,
    ) -> EmbeddingResult:
        """Embed chunks using the configured embedder.

        Args:
            chunk_result: The ChunkingResult to embed

        Returns:
            An EmbeddingResult containing embeddings for each chunk

        Raises:
            ValueError: If embedder is not configured
        """
        if not self.embedder:
            raise ValueError("embedder is required for embedding operation")
        usage_tracker = TokenUsageTracker()
        embedding_result = self.embedder.embed_chunks(
            chunk_result, usage_tracker
        )
        return embedding_result

    def _run_step(
        self,
        step: _PipelineStep,
        dpr: DocumentPipelineResult,
        start_total: float,
    ) -> tuple[bool, object | None]:
        key = step.key  # e.g. "ocr"
        self.logger.info(f"Starting {key} step…")
        start = time.perf_counter()
        try:
            result = step.run()
            setattr(dpr.step_report, key, StepStatus.success)
            setattr(dpr.timings, key, time.perf_counter() - start)
            step.on_success(result)
            self.logger.info(f"Step {key} completed")
            return True, result
        except Exception as exc:
            tb = traceback.format_exc()
            self.logger.error(f"Step {key} failed: {exc}\n{tb}")
            setattr(dpr.step_report, key, StepStatus.failed)
            setattr(dpr.timings, key, time.perf_counter() - start)
            dpr.timings.total_duration = time.perf_counter() - start_total
            return False, None

    def process(
        self,
        pdf_filepath: str
    ) -> DocumentPipelineResult:
        """Run the configured pipeline steps in order.

        Raises:
            ValueError: If any required component is not configured
        """
        # Validate all components are present for full pipeline execution
        if not self.ocr_processor:
            raise ValueError(
                "ocr_processor is required for full pipeline execution"
            )
        if not self.chunker:
            raise ValueError(
                "chunker is required for full pipeline execution"
            )
        if not self.embedder:
            raise ValueError(
                "embedder is required for full pipeline execution"
            )

        start_total = time.perf_counter()
        dpr = DocumentPipelineResult(
            source_file_path=pdf_filepath,
            processed_at=datetime.now(timezone.utc),
            pipeline_config={
                "ocr": str(self.ocr_processor),
                "processors": [str(p) for p in self.processors],
                "chunker": str(self.chunker),
                "embedder": str(self.embedder),
            },
            timings=TimingMetrics(),
            total_metrics=[],
            step_report=StepReport(),
        )

        # ---------------- helpers ----------------
        def _on_success(attr):
            def handler(res):
                # 1. Set the result (save the step result to DPR)
                setattr(dpr, attr, res)
                # 2. Set the metrics of the result to total metrics
                # - res may be a single result or a list of results
                if isinstance(res, list):
                    dpr.total_metrics.extend(
                        [r.metrics for r in res if r.metrics]
                    )
                else:
                    if isinstance(res.metrics, list):
                        dpr.total_metrics.extend(res.metrics or [])
                    else:
                        dpr.total_metrics.append(res.metrics)
            return handler

        # placeholders for passing results between steps
        ocr_res: OCRResult | None = None
        proc_results: list[ProcessingResult] | None = None
        chunk_res: ChunkingResult | None = None

        # ---------------- step table ----------------
        steps = [
            self._PipelineStep(
                "ocr",
                lambda: self.run_ocr(pdf_filepath),
                _on_success("ocr_result"),
            ),
            self._PipelineStep(
                "processing",
                lambda: self.run_processors(ocr_res),
                _on_success("processing_results"),
            ),
            self._PipelineStep(
                "chunking",
                lambda: self.run_chunker(
                    proc_results[-1] if proc_results else ocr_res
                ),
                _on_success("chunking_result"),
            ),
            self._PipelineStep(
                "embedding",
                lambda: self.run_embedder(chunk_res),
                _on_success("embedding_result"),
            ),
        ]

        try:
            for st in steps:
                ok, res = self._run_step(st, dpr, start_total)
                if not ok:
                    return dpr

                # propagate outputs for later steps
                if st.key == "ocr":
                    ocr_res = res  # type: ignore
                elif st.key == "processing":
                    proc_results = res  # type: ignore
                elif st.key == "chunking":
                    chunk_res = res  # type: ignore

            # aggregate total cost once
            dpr.total_cost_usd = sum(
                m.total_cost_usd  # type: ignore[attr-defined]
                for m in dpr.total_metrics
                if m and getattr(m, "total_cost_usd", None) is not None
            )

            dpr.timings.total_duration = time.perf_counter() - start_total
            self.logger.info("Document processing completed.")
            return dpr
        finally:
            dpr.logs = self._transcript.dump()
            logging.getLogger().removeHandler(self._transcript)
