from ragbandit.utils import mistral_client_manager
from ragbandit.utils.token_usage_tracker import TokenUsageTracker
from ragbandit.documents.embedders.base_embedder import BaseEmbedder
from ragbandit.schema import (
    ChunkingResult,
    EmbeddingResult,
    ChunkWithEmbedding,
)
from datetime import datetime, timezone


class MistralEmbedder(BaseEmbedder):
    """Document embedder that uses Mistral AI's embedding models."""

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-embed",
        name: str = None,
    ):
        """
        Initialize the Mistral embedder.

        Args:
            api_key: Mistral API key
            model: Embedding model to use
            name: Optional name for the embedder
        """
        super().__init__(name)
        self.model = model

        # Initialize the Mistral client
        self.client = mistral_client_manager.get_client(api_key)

        self.logger.info(
            f"Initialized MistralEmbedder with model {self.model}"
        )

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

    def _call_mistral_embeddings(self, texts: list[str]):
        """Call the Mistral embeddings API and return the raw response."""
        self.logger.info("Requesting embeddings from Mistral API")
        return self.client.embeddings.create(model=self.model, inputs=texts)

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
                response.usage.prompt_tokens, self.model
            )

        return EmbeddingResult(
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
            processed_at=None,
            chunks_with_embeddings=empty_embeds,
            model_name=self.model,
            metrics=None,
        )
