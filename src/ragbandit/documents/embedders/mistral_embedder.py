from dataclasses import dataclass
from datetime import datetime, timezone

from ragbandit.utils import mistral_client_manager
from ragbandit.utils.token_usage_tracker import TokenUsageTracker
from ragbandit.documents.embedders.base_embedder import BaseEmbedder
from ragbandit.schema import (
    ChunkingResult,
    EmbeddingResult,
    ChunkWithEmbedding,
)


@dataclass
class BatchedEmbeddingResponse:
    """Combined response from multiple embedding API calls."""
    data: list
    prompt_tokens: int


class MistralEmbedder(BaseEmbedder):
    """Document embedder that uses Mistral AI's embedding models."""

    VALID_MODELS = ["mistral-embed"]

    # Mistral embedding API limits
    MAX_BATCH_TOKENS = 16000  # Conservative limit (actual is ~16k)
    CHARS_PER_TOKEN = 4  # Rough estimate for token counting

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-embed",
    ):
        """
        Initialize the Mistral embedder.

        Args:
            api_key: Mistral API key
            model: Embedding model to use (must be in VALID_MODELS)

        Raises:
            ValueError: If model is not in VALID_MODELS
        """
        if model not in self.VALID_MODELS:
            raise ValueError(
                f"Invalid model '{model}'. "
                f"Must be one of: {', '.join(self.VALID_MODELS)}"
            )

        super().__init__(api_key)
        self.model = model

        # Initialize the Mistral client
        self.client = mistral_client_manager.get_client(api_key)

        self.logger.info(
            f"Initialized MistralEmbedder with model {self.model}"
        )

    def get_config(self) -> dict:
        """Return the configuration for this embedder.

        Returns:
            dict: Configuration dictionary
        """
        return {"model": self.model}

    # ------------------------------------------------------------------
    # Public API
    def embed_chunks(
        self,
        chunk_result: ChunkingResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> EmbeddingResult:
        """Orchestrate the Mistral embedding flow."""

        chunks = chunk_result.chunks
        if not chunks:
            self.logger.warning("No chunks to embed")
            return self._empty_embedding_result()

        try:
            texts = self._extract_texts(chunks)
            response = self._call_mistral_embeddings(texts)
            return self._build_embedding_result(
                chunks, response, usage_tracker
            )
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return self._empty_embedding_result(chunks)

    # ------------------------------------------------------------------
    # Helpers
    def _extract_texts(
        self,
        chunks: list[ChunkWithEmbedding | object],
    ) -> list[str]:
        """Extract raw text from chunks."""
        return [c.text for c in chunks]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def _batch_by_tokens(self, texts: list[str]) -> list[list[str]]:
        """
        Group texts into batches that stay under the token limit.

        Args:
            texts: List of texts to batch

        Returns:
            List of batches, where each batch is a list of texts
        """
        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            text_tokens = self._estimate_tokens(text)

            if current_tokens + text_tokens > self.MAX_BATCH_TOKENS:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def _call_mistral_embeddings(
        self, texts: list[str]
    ) -> BatchedEmbeddingResponse:
        """
        Call the Mistral embeddings API with token-aware batching.

        Args:
            texts: List of texts to embed

        Returns:
            BatchedEmbeddingResponse with all embeddings combined
        """
        batches = self._batch_by_tokens(texts)
        num_batches = len(batches)

        self.logger.info(
            f"Embedding {len(texts)} texts in {num_batches} batch(es)"
        )

        all_data = []
        total_prompt_tokens = 0

        for i, batch in enumerate(batches, 1):
            if num_batches > 1:
                self.logger.info(
                    f"  Batch {i}/{num_batches}: {len(batch)} texts"
                )
            response = self.client.embeddings.create(
                model=self.model, inputs=batch
            )
            all_data.extend(response.data)
            if hasattr(response, "usage") and response.usage:
                total_prompt_tokens += response.usage.prompt_tokens

        return BatchedEmbeddingResponse(all_data, total_prompt_tokens)

    def _build_embedding_result(
        self,
        chunks,
        response,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> EmbeddingResult:
        """Convert API response into EmbeddingResult."""

        embedded_chunks: list[ChunkWithEmbedding] = []
        for i, data in enumerate(response.data):
            embedded_chunks.append(
                ChunkWithEmbedding(
                    text=chunks[i].text,
                    metadata=chunks[i].metadata,
                    embedding=list(data.embedding),
                    embedding_model=self.model,
                )
            )

        if usage_tracker:
            usage_tracker.add_embedding_tokens(
                response.prompt_tokens, self.model
            )

        return EmbeddingResult(
            component_name=self.get_name(),
            component_config=self.get_config(),
            processed_at=datetime.now(timezone.utc),
            chunks_with_embeddings=embedded_chunks,
            model_name=self.model,
            metrics=usage_tracker.get_summary() if usage_tracker else None,
        )

    def _empty_embedding_result(
        self, chunks: list | None = None
    ) -> EmbeddingResult:
        """Return an EmbeddingResult with no embeddings (error fallback)."""

        empty_embeds: list[ChunkWithEmbedding] = []
        for c in (chunks or []):
            empty_embeds.append(
                ChunkWithEmbedding(
                    text=c.text if hasattr(c, "text") else "",
                    metadata=c.metadata if hasattr(c, "metadata") else None,
                    embedding=[],
                    embedding_model=self.model,
                )
            )

        return EmbeddingResult(
            component_name=self.get_name(),
            component_config=self.get_config(),
            processed_at=None,
            chunks_with_embeddings=empty_embeds,
            model_name=self.model,
            metrics=None,
        )
