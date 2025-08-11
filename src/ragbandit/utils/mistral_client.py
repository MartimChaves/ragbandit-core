"""
Centralized Mistral API client module.

This module provides a singleton Mistral client instance
to be used across the application,
ensuring consistent configuration and
preventing duplicate initializations.
"""

from mistralai import Mistral
import os
import logging

# Configure logger
logger = logging.getLogger(__name__)


def get_mistral_client():
    """
    Get a configured Mistral client instance.

    Returns:
        Mistral: Configured client instance

    Raises:
        EnvironmentError: If MISTRAL_API_KEY is not set
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.error("MISTRAL_API_KEY environment variable not set")
        raise EnvironmentError("MISTRAL_API_KEY environment variable not set")

    return Mistral(api_key=api_key)


# Singleton instance to be imported by other modules
mistral_client = get_mistral_client()
