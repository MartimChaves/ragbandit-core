"""
LLM configuration settings for ragbandit.

This module defines default settings and constants for LLM interactions.
"""

# Default model settings
DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_TEMPERATURE = 0.0

# Retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0  # exponential backoff factor
DEFAULT_TIMEOUT = 30.0  # seconds

# Token limits
MAX_PROMPT_TOKENS = {
    "mistral-small-latest": 8000,
    "mistral-medium-latest": 32000,
    "mistral-large-latest": 32000,
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
}

# System prompts
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant."""

# Response formats
JSON_FORMAT_INSTRUCTION = """
Your response must be valid JSON that matches the following schema:
{schema}
"""
