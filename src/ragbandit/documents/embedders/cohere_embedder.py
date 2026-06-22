"""
Cohere embedder for generating document embeddings.

Uses the cohere SDK to embed document chunks in batches.
Supports the embed-v4.0 model.
"""

from datetime import datetime, timezone

import cohere

from ragbandit.utils.token_usage_tracker import TokenUsageTracker
from ragbandit.documents.embedders.base_embedder import BaseEmbedder
from ragbandit.schema import (
    ChunkingResult,
    EmbeddingResult,
    ChunkWithEmbedding,
)


class CohereEmbedder(BaseEmbedder):
    """Document embedder that uses Cohere embedding models.

    Sends chunks to the Cohere embeddings API in batches of up to
    ``BATCH_SIZE`` texts. Token usage is reported to the optional
    ``TokenUsageTracker`` after each batch via the response's billed
    units metadata.
    """

    VALID_MODELS = [
        "embed-v4.0",
    ]
    BATCH_SIZE = 96  # Cohere maximum texts per request

    def __init__(
        self,
        api_key: str,
        model: str = "embed-v4.0",
        input_type: str = "search_document",
    ):
        """
        Initialize the Cohere embedder.

        Args:
            api_key: Cohere API key.
            model: Embedding model name (must be in VALID_MODELS).
            input_type: Cohere input_type parameter. Use
                ``"search_document"`` when indexing chunks and
                ``"search_query"`` when embedding queries at retrieval time.

        Raises:
            ValueError: If model is not in VALID_MODELS.
        """
        if model not in self.VALID_MODELS:
            raise ValueError(
                f"Invalid model '{model}'. "
                f"Must be one of: {', '.join(self.VALID_MODELS)}"
            )
        super().__init__(api_key)
        self.model = model
        self.input_type = input_type

        self.client = cohere.Client(api_key=api_key)
        self.logger.info(
            f"Initialized CohereEmbedder with model={self.model}, "
            f"input_type={self.input_type}"
        )

    def get_config(self) -> dict:
        """Return the configuration for this embedder.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            "model": self.model,
            "input_type": self.input_type,
        }

    # ------------------------------------------------------------------
    # Public API

    def embed_chunks(
        self,
        chunk_result: ChunkingResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> EmbeddingResult:
        """Generate embeddings for all chunks in a ChunkingResult.

        Args:
            chunk_result: The ChunkingResult whose chunks will be embedded.
            usage_tracker: Optional tracker for token usage.

        Returns:
            An EmbeddingResult containing embedded chunks.
        """
        chunks = chunk_result.chunks
        if not chunks:
            self.logger.warning("No chunks to embed")
            return self._empty_embedding_result()

        try:
            texts = [c.text for c in chunks]
            embedded_chunks = self._embed_in_batches(
                texts, chunks, usage_tracker
            )
            return EmbeddingResult(
                component_name=self.get_name(),
                component_config=self.get_config(),
                processed_at=datetime.now(timezone.utc),
                chunks_with_embeddings=embedded_chunks,
                model_name=self.model,
                metrics=usage_tracker.get_summary() if usage_tracker else None,
            )
        except Exception as e:
            self.logger.error(f"Error generating Cohere embeddings: {e}")
            return self._empty_embedding_result(chunks)

    # ------------------------------------------------------------------
    # Helpers

    def _embed_in_batches(
        self,
        texts: list[str],
        chunks,
        usage_tracker: TokenUsageTracker | None,
    ) -> list[ChunkWithEmbedding]:
        """Call the Cohere API in batches and
        collect ChunkWithEmbedding objects.

        Token usage is read from ``response.meta.billed_units``
        when available.

        Args:
            texts: Plain-text strings extracted from the chunks.
            chunks: Original Chunk objects (used to preserve metadata).
            usage_tracker: Optional tracker updated with each batch's tokens.

        Returns:
            List of ChunkWithEmbedding objects in the same order as chunks.
        """
        embedded_chunks: list[ChunkWithEmbedding] = []

        for batch_start in range(0, len(texts), self.BATCH_SIZE):
            batch_texts = texts[batch_start: batch_start + self.BATCH_SIZE]
            batch_chunks = chunks[batch_start: batch_start + self.BATCH_SIZE]

            self.logger.info(
                f"Embedding batch {batch_start // self.BATCH_SIZE + 1}: "
                f"{len(batch_texts)} texts"
            )

            response = self.client.embed(
                texts=batch_texts,
                model=self.model,
                input_type=self.input_type,
                embedding_types=["float"],
            )
            raw_embeddings = response.embeddings.float

            # Track billed tokens when reported by the API
            billed = getattr(
                getattr(response, "meta", None), "billed_units", None
            )
            if billed is not None:
                token_count = getattr(billed, "input_tokens", 0) or 0
                if usage_tracker and token_count:
                    usage_tracker.add_embedding_tokens(token_count, self.model)

            for i, embedding in enumerate(raw_embeddings):
                embedded_chunks.append(
                    ChunkWithEmbedding(
                        text=batch_chunks[i].text,
                        metadata=batch_chunks[i].metadata,
                        embedding=list(embedding),
                        embedding_model=self.model,
                    )
                )

        return embedded_chunks

    def _empty_embedding_result(
        self, chunks: list | None = None
    ) -> EmbeddingResult:
        """Return an EmbeddingResult with empty embeddings (error fallback).

        Args:
            chunks: Optional list of chunks to preserve text and metadata
                in the result even when embeddings could not be generated.

        Returns:
            EmbeddingResult with empty embedding vectors.
        """
        empty_embeds = [
            ChunkWithEmbedding(
                text=c.text if hasattr(c, "text") else "",
                metadata=c.metadata if hasattr(c, "metadata") else None,
                embedding=[],
                embedding_model=self.model,
            )
            for c in (chunks or [])
        ]
        return EmbeddingResult(
            component_name=self.get_name(),
            component_config=self.get_config(),
            processed_at=None,
            chunks_with_embeddings=empty_embeds,
            model_name=self.model,
            metrics=None,
        )
