"""
LLM configuration settings for ragbandit.

This module defines default settings and constants for LLM interactions.
"""

# Default model settings
DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_TEMPERATURE = 0.0

# Model escalation chain for fallback on persistent failures
MODEL_ESCALATION_CHAIN = [
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-large-latest"
]

# Retry settings
DEFAULT_MAX_RETRIES = 5  # 5 retries with backoff = ~62s max wait
DEFAULT_RETRY_DELAY = 2.0  # seconds (initial delay)
DEFAULT_BACKOFF_FACTOR = 2.0  # exponential backoff factor
DEFAULT_TIMEOUT = 30.0  # seconds

# Token limits (context windows, as of June 2026)
MAX_PROMPT_TOKENS = {
    "mistral-small-latest": 256000,   # Mistral Small 4
    "mistral-medium-latest": 128000,  # Mistral Medium 3.5
    "mistral-large-latest": 128000,   # Mistral Large 3
}

# System prompts
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant."""

# Response formats
JSON_FORMAT_INSTRUCTION = """
Your response must be valid JSON that matches the following schema:
{schema}
"""
