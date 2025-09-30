from abc import ABC, abstractmethod

# Third-party
import numpy as np

# Project
from ragbandit.schema import (
    ChunkingResult,
    EmbeddingResult,
)
from ragbandit.utils.token_usage_tracker import TokenUsageTracker


class BaseEmbedder(ABC):
    """
    Abstract base class for document embedders.

    This class defines the interface for embedding document chunks.
    Concrete implementations should handle the specifics of
    generating embeddings using different models or providers.
    """

    def __init__(self, name: str = None):
        """
        Initialize the document embedder.

        Args:
            name: Optional name for the embedder
        """
        self.name = name or self.__class__.__name__

        # Set up logging
        import logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def embed_chunks(
        self,
        chunk_result: ChunkingResult,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings for a ChunkingResult.

        Args:
            chunk_result: The ChunkingResult whose chunks will be embedded
            usage_tracker: Optional tracker for token usage

        Returns:
            An EmbeddingResult containing embedded chunks
        """
        raise NotImplementedError

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embedding vectors.

        Args:
            a: First embedding vector
            b: Second embedding vector

        Returns:
            Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
        """
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine distance between two embedding vectors.

        Args:
            a: First embedding vector
            b: Second embedding vector

        Returns:
            Cosine distance (0 = identical, 2 = opposite)
        """
        return 1 - self.cosine_similarity(a, b)

    def __repr__(self) -> str:
        """Return string representation of the embedder."""
        return f"{self.__class__.__name__}"
