"""
Cost tracking utilities for LLM API calls.

This module provides functions to calculate token usage and costs
for different LLM models.
"""

import logging
import tiktoken
from ragbandit.config.pricing import (
    MODEL_COSTS,
    EMBEDDING_COSTS,
    DEFAULT_MODEL
)
from ragbandit.schema import TokenUsageMetrics

# Configure logger
logger = logging.getLogger(__name__)


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """
    Count the number of tokens in a text string for a specific model.

    Args:
        text: The text to count tokens for
        model: The model to use for token counting

    Returns:
        int: Number of tokens
    """
    try:
        # For Mistral models, use cl100k_base encoding (same as GPT-4)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(
            f"Error counting tokens: {e}. Using character-based estimate."
        )
        # Fallback: rough estimate based on characters (1 token ≈ 4 chars)
        return len(text) // 4


def calculate_cost(
    input_tokens: int, output_tokens: int, model: str = DEFAULT_MODEL
) -> tuple[float, dict[str, float]]:
    """
    Calculate the cost of an API call based on token usage.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name

    Returns:
        Tuple containing:
            - Total cost in USD
            - Dictionary with detailed cost breakdown
    """
    # Get cost rates, defaulting to mistral-small if model not found
    input_rate, output_rate = MODEL_COSTS.get(
        model, MODEL_COSTS[DEFAULT_MODEL]
    )

    # Calculate costs (rates are per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    total_cost = input_cost + output_cost

    cost_details = {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }

    return total_cost, cost_details


class TokenUsageTracker:
    """Track token usage and costs across multiple API calls."""
    total_input_tokens: int
    total_output_tokens: int
    total_embedding_tokens: int
    total_cost: float
    calls_by_model: dict[str, dict[str, int | float]]

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_embedding_tokens = 0
        self.total_cost = 0.0
        self.calls_by_model = {}

    def add_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """
        Add usage statistics from an API call.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
        """
        cost, details = calculate_cost(input_tokens, output_tokens, model)

        # Update totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        # Update per-model tracking
        if model not in self.calls_by_model:
            self.calls_by_model[model] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
            }

        self.calls_by_model[model]["calls"] += 1
        self.calls_by_model[model]["input_tokens"] += input_tokens
        self.calls_by_model[model]["output_tokens"] += output_tokens
        self.calls_by_model[model]["cost"] += cost

    def add_embedding_tokens(self, tokens: int, model: str) -> None:
        """
        Add embedding token usage statistics.

        Args:
            tokens: Number of tokens processed for embedding
            model: Embedding model name
        """
        # Calculate cost based on embedding model rates
        # Default to 0.10 if model not found
        cost_per_million = EMBEDDING_COSTS.get(model, 0.10)
        cost = (tokens / 1_000_000) * cost_per_million

        # Update totals
        self.total_embedding_tokens += tokens
        self.total_cost += cost

        # Update per-model tracking
        if model not in self.calls_by_model:
            self.calls_by_model[model] = {
                "calls": 0,
                "embedding_tokens": 0,
                "cost": 0.0,
            }
        else:
            # Add embedding_tokens field if it doesn't exist
            if "embedding_tokens" not in self.calls_by_model[model]:
                self.calls_by_model[model]["embedding_tokens"] = 0

        self.calls_by_model[model]["calls"] += 1
        self.calls_by_model[model]["embedding_tokens"] = (
            self.calls_by_model[model].get("embedding_tokens", 0) + tokens
        )
        self.calls_by_model[model]["cost"] += cost

    def get_summary(self) -> TokenUsageMetrics:
        """
        Get a summary of token usage and costs.

        Returns:
            TokenUsageMetrics object with usage summary
        """
        models_converted: dict[str, TokenUsageMetrics.ModelUsage] = {}
        for model_name, stats in self.calls_by_model.items():
            models_converted[model_name] = TokenUsageMetrics.ModelUsage(
                calls=int(stats.get("calls", 0)),
                input_tokens=int(stats.get("input_tokens", 0)),
                output_tokens=int(stats.get("output_tokens", 0)),
                embedding_tokens=int(stats.get("embedding_tokens", 0)),
                cost=float(stats.get("cost", 0.0)),
            )

        return TokenUsageMetrics(
            total_calls=sum(m.calls for m in models_converted.values()),
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_embedding_tokens=self.total_embedding_tokens,
            total_tokens=(
                self.total_input_tokens +
                self.total_output_tokens +
                self.total_embedding_tokens
            ),
            total_cost_usd=self.total_cost,
            models=models_converted,
        )

    def log_summary(self, level: int = logging.INFO) -> None:
        """Log a summary of token usage and costs."""
        summary = self.get_summary()

        # Build log message
        message = f"API Usage: {summary.total_calls} calls, "

        # Add LLM token counts if any
        if summary.total_input_tokens > 0 or summary.total_output_tokens > 0:  # noqa
            message += (
                f"LLM: {summary.total_input_tokens:,} input + "
                f"{summary.total_output_tokens:,} output tokens, "
            )

        # Add embedding token counts if any
        if summary.total_embedding_tokens > 0:
            message += f"Embeddings: {summary.total_embedding_tokens:,} tokens, "  # noqa

        # Add total cost
        message += f"Total: ${summary.total_cost_usd:.4f} USD"

        logger.log(level, message)
