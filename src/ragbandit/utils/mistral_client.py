"""
Utility functions for working with the Mistral API.

This module provides helper functions for creating and managing
Mistral API client instances.

The module exports a singleton instance of MistralClientManager as
'mistral_client_manager' that should be used throughout the application
to ensure consistent client caching and management.
"""

from mistralai import Mistral
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Timeout settings (in seconds)
DEFAULT_TIMEOUT = 120.0  # 2 minutes max per request


class MistralClientManager:
    """
    Manager class for Mistral API clients.

    This class provides a way to cache and
    reuse Mistral client instances
    based on API keys, avoiding the need to
    create a new client for each request.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        """Initialize an empty client cache."""
        self._clients: dict[str, Mistral] = {}
        self._timeout = timeout

    def get_mistral_client(self, api_key: str):
        """
        Get a configured Mistral client instance.

        Returns:
            Mistral: Configured client instance

        Raises:
            ValueError: If api_key is None
        """
        if not api_key or not api_key.strip():
            raise ValueError("Mistral API key cannot be empty or None")
        return Mistral(
            api_key=api_key,
            timeout_ms=int(self._timeout * 1000),
        )

    def get_client(self, api_key: str) -> Mistral:
        """
        Get a Mistral client for the given API key.

        If a client with this API key already exists in the cache,
        it will be reused.
        Otherwise, a new client will be created and cached.

        Args:
            api_key: Mistral API key to use for authentication

        Returns:
            Mistral: A configured Mistral client instance

        Raises:
            ValueError: If api_key is empty or None
        """
        # Hash the API key to use as a dictionary key
        # This avoids storing the actual API key in memory as a dictionary key
        key_hash = hash(api_key)

        if key_hash not in self._clients:
            # Create a new client and cache it
            self._clients[key_hash] = self.get_mistral_client(api_key)

        return self._clients[key_hash]


# Global instance of the client manager
mistral_client_manager = MistralClientManager()
