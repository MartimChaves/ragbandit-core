"""
Pricing configuration for LLM API calls.

This module contains pricing constants for various
LLM models and embedding models.
"""

# Token cost rates per 1M tokens (in USD)
# Based on Mistral AI pricing as of June 2026
# (mistral-*-latest aliases: Small 4, Medium 3.5, Large 3)
MODEL_COSTS = {
    # Format: "model_name": (input_cost_per_1M, output_cost_per_1M)
    "mistral-small-latest": (0.1, 0.3),
    "mistral-medium-latest": (1.5, 7.5),
    "mistral-large-latest": (0.5, 1.5),
    # Add other models as needed
}

# Embedding model costs per 1M tokens
EMBEDDING_COSTS = {
    # Format: "model_name": cost_per_1M_tokens
    "mistral-embed": 0.10,
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    # Voyage AI models (pricing as of June 2026)
    "voyage-3-large": 0.18,
    "voyage-3.5": 0.06,
    "voyage-3.5-lite": 0.02,
    "voyage-3": 0.06,
    "voyage-3-lite": 0.02,
    # Cohere models (pricing as of June 2026)
    "embed-v4.0": 0.12,
}

# OCR model costs per page (in USD)
OCR_MODEL_COSTS = {
    # Format: "model_name": cost_per_page
    "mistral-ocr-2512": 0.002,  # $2 per 1000 pages (Mistral OCR 3)
}

# Default OCR model to use if the specified model is not in OCR_MODEL_COSTS
DEFAULT_OCR_MODEL = "mistral-ocr-2512"

# Default model to use if the specified model is not in MODEL_COSTS
DEFAULT_MODEL = "mistral-small-latest"
