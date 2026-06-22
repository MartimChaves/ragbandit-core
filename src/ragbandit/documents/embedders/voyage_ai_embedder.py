"""
Voyage AI embedder for generating document embeddings.

Uses the voyageai SDK to embed document chunks in batches.
Supports voyage-3-large, voyage-3.5, voyage-3.5-lite, voyage-3, and
voyage-3-lite models.
"""

from datetime import datetime, timezone

import voyageai

from ragbandit.utils.token_usage_tracker import TokenUsageTracker
from ragbandit.documents.embedders.base_embedder import BaseEmbedder
from ragbandit.schema import (
    ChunkingResult,
    EmbeddingResult,
    ChunkWithEmbedding,
)


class VoyageAIEmbedder(BaseEmbedder):
    """Document embedder that uses Voyage AI embedding models.

    Sends chunks to the Voyage AI embeddings API in batches of up to
    ``BATCH_SIZE`` texts. Token usage is reported to the optional
    ``TokenUsageTracker`` after each batch.
    """

    VALID_MODELS = [
        "voyage-3-large",
        "voyage-3.5",
        "voyage-3.5-lite",
        "voyage-3",
        "voyage-3-lite",
    ]
    BATCH_SIZE = 128

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3.5",
    ):
        """
        Initialize the Voyage AI embedder.

        Args:
            api_key: Voyage AI API key.
            model: Embedding model name (must be in VALID_MODELS).

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
        self.client = voyageai.Client(api_key=api_key)
        self.logger.info(
            f"Initialized VoyageAIEmbedder with model {self.model}"
        )

    def get_config(self) -> dict:
        """Return the configuration for this embedder.

        Returns:
            dict: Configuration dictionary.
        """
        return {"model": self.model}

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
            self.logger.error(f"Error generating Voyage AI embeddings: {e}")
            return self._empty_embedding_result(chunks)

    # ------------------------------------------------------------------
    # Helpers

    def _embed_in_batches(
        self,
        texts: list[str],
        chunks,
        usage_tracker: TokenUsageTracker | None,
    ) -> list[ChunkWithEmbedding]:
        """Call the Voyage AI API in batches and
        collect ChunkWithEmbedding objects.

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
            result = self.client.embed(
                batch_texts,
                model=self.model,
                input_type="document",
            )

            batch_tokens = getattr(result, "total_tokens", 0)
            if usage_tracker and batch_tokens:
                usage_tracker.add_embedding_tokens(batch_tokens, self.model)

            for i, embedding in enumerate(result.embeddings):
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
