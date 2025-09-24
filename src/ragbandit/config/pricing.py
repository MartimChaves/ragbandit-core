"""
Pricing configuration for LLM API calls.

This module contains pricing constants for various
LLM models and embedding models.
"""

# Token cost rates per 1M tokens (in USD)
# Based on Mistral AI pricing as of July 2025
MODEL_COSTS = {
    # Format: "model_name": (input_cost_per_1M, output_cost_per_1M)
    "mistral-small-latest": (2.00, 6.00),
    "mistral-medium-latest": (6.00, 18.00),
    "mistral-large-latest": (12.00, 36.00),
    # Add other models as needed
}

# Embedding model costs per 1M tokens
EMBEDDING_COSTS = {
    # Format: "model_name": cost_per_1M_tokens
    "mistral-embed": 0.10,
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    # Add other embedding models as needed
}

# OCR model costs per page (in EUR)
OCR_MODEL_COSTS = {
    # Format: "model_name": cost_per_page
    "mistral-ocr-latest": 0.001,  # 1 EUR per 1000 pages
    # Add other OCR models as needed
}

# Default OCR model to use if the specified model is not in OCR_MODEL_COSTS
DEFAULT_OCR_MODEL = "mistral-ocr-latest"

# Default model to use if the specified model is not in MODEL_COSTS
DEFAULT_MODEL = "mistral-small-latest"
