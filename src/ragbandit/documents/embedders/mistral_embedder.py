import numpy as np
from mistralai import Mistral

from ragbandit.utils.token_usage_tracker import TokenUsageTracker
from ragbandit.documents.embedders.base_embedder import BaseEmbedder


class MistralEmbedder(BaseEmbedder):
    """Document embedder that uses Mistral AI's embedding models."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "mistral-embed",
        name: str = None,
    ):
        """
        Initialize the Mistral embedder.

        Args:
            api_key: Mistral API key (if None, will try to use environment
                     variable)
            model: Embedding model to use
            name: Optional name for the embedder
        """
        super().__init__(name)
        self.model = model

        # Initialize the Mistral client
        self.client = Mistral(api_key=api_key)

        self.logger.info(
            f"Initialized MistralEmbedder with model {self.model}"
        )

    def embed_chunks(
        self,
        chunks: list[dict[str, any]],
        usage_tracker: TokenUsageTracker = None
    ) -> list[dict[str, any]]:
        """
        Generate embeddings for a list of document chunks using Mistral's API.

        Args:
            chunks: List of chunk dictionaries with chunk_text field
            usage_tracker: Optional tracker for token usage

        Returns:
            The chunks with embeddings added
        """
        if not chunks:
            self.logger.warning("No chunks to embed")
            return chunks

        # Extract texts from chunks
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]

        self.logger.info(
            f"Generating embeddings for {len(chunk_texts)} chunks"
        )

        # Call Mistral API to get embeddings
        try:
            response = self.client.embeddings.create(
                model=self.model,
                inputs=chunk_texts,
            )

            # Track token usage if a tracker is provided
            if usage_tracker:
                usage_tracker.add_embedding_tokens(
                    response.usage.prompt_tokens,
                    self.model
                )

            # Add embeddings to chunks
            for i, embedding_data in enumerate(response.data):
                # Convert embedding to numpy array for easier manipulation
                embedding_array = np.array(embedding_data.embedding)
                chunks[i]["embedding"] = embedding_array
                chunks[i]["embedding_model"] = self.model

            self.logger.info(
                f"Successfully generated {len(response.data)} embeddings"
            )

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            # Add empty embeddings to avoid breaking downstream code
            for chunk in chunks:
                chunk["embedding"] = None
                chunk["embedding_model"] = self.model
                chunk["embedding_error"] = str(e)

        return chunks
