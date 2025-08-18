"""
Utility functions for interacting with LLM services.

This module provides standardized ways to make LLM requests with
consistent error handling, retries, and response parsing.
"""

import json
import time
import logging
import requests
from typing import Type, TypeVar
from pydantic import BaseModel
from ragbandit.utils.mistral_client import mistral_client_manager
from ragbandit.config.llms import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_BACKOFF_FACTOR,
)
from ragbandit.utils.token_usage_tracker import TokenUsageTracker, count_tokens

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for Pydantic model return types
T = TypeVar("T", bound=BaseModel)


def query_llm(
    prompt: str,
    output_schema: Type[T],
    api_key: str,
    usage_tracker: TokenUsageTracker | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    track_usage: bool = True,
) -> T:
    """
    Send a query to the LLM with standardized formatting and retry logic.

    Args:
        prompt: The prompt to send to the LLM
        output_schema: Pydantic model class for response validation and parsing
        api_key: API key to use for the request
        usage_tracker: Optional custom token usage tracker for
                       document-specific tracking.
                       If None, no tracking will be performed even
                       if track_usage is True.
        model: Model name to use for the request
        temperature: Sampling temperature (0 = deterministic)
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay on each retry attempt
        track_usage: Whether to track token usage and costs

    Returns:
        Validated instance of the output_schema model

    Raises:
        ValueError: If response cannot be parsed according to schema
        RuntimeError: If all retry attempts fail
    """
    retry_count = 0
    current_delay = retry_delay

    # Only track usage if both conditions are met:
    # 1. User wants to track usage (track_usage=True)
    # 2. We have a tracker to use (usage_tracker is not None)
    should_track = track_usage and usage_tracker is not None

    # Count input tokens if tracking is enabled
    input_tokens = 0
    if should_track:
        # Count tokens in the prompt
        input_tokens = count_tokens(prompt, model)
        logger.debug(f"Input tokens: {input_tokens} for model {model}")

    while retry_count <= max_retries:
        try:
            # Make the API request
            client = mistral_client_manager.get_client(api_key)
            chat_response = client.chat.complete(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={
                    "type": "json_object",
                    "schema": output_schema.model_json_schema(),
                },
                temperature=temperature,
            )

            # Parse and validate the response
            response_content = chat_response.choices[0].message.content
            response_dict = json.loads(response_content)

            # Track token usage if enabled
            if should_track and hasattr(chat_response, 'usage'):
                # Get token counts from the API response
                output_tokens = chat_response.usage.completion_tokens
                actual_input_tokens = chat_response.usage.prompt_tokens

                # Use the actual input tokens from the API if available
                if actual_input_tokens > 0:
                    input_tokens = actual_input_tokens

                # Log token usage
                logger.debug(
                    f"Token usage - Input: {input_tokens}, "
                    f"Output: {output_tokens}, "
                    f"Model: {model}"
                )

                # Track in the global usage tracker
                usage_tracker.add_usage(input_tokens, output_tokens, model)
            elif should_track:
                # If API doesn't return usage stats, estimate output tokens
                output_tokens = count_tokens(response_content, model)
                usage_tracker.add_usage(input_tokens, output_tokens, model)
                logger.debug(
                    f"Estimated token usage - Input: {input_tokens}, "
                    f"Output: {output_tokens}, "
                    f"Model: {model}"
                )

            return output_schema.model_validate(response_dict)

        except (requests.RequestException, TimeoutError, ConnectionError) as e:
            # Handle network-related errors (timeouts, connection issues)
            retry_count += 1

            # If we've exhausted retries, raise the error
            if retry_count > max_retries:
                logger.error(f"Failed after {max_retries} retries: {str(e)}")
                raise RuntimeError(
                    f"LLM request failed after {max_retries} retries: {str(e)}"
                )

            # Log the error and retry
            logger.warning(
                "LLM request failed "
                f"(attempt {retry_count}/{max_retries}): {str(e)}. "
                f"Retrying in {current_delay} seconds..."
            )
            time.sleep(current_delay)
            current_delay *= backoff_factor

        except Exception as e:
            # Handle other API errors (rate limits, server errors)
            if "429" in str(e) or "too many requests" in str(e).lower():
                # Rate limiting error - retry with backoff
                retry_count += 1

                if retry_count > max_retries:
                    logger.error(
                        (
                            f"Rate limit exceeded after {max_retries} "
                            f"retries: {str(e)}"
                        )
                    )
                    raise RuntimeError(
                        f"Rate limit exceeded after {max_retries} retries"
                    )

                logger.warning(
                    f"Rate limit hit (attempt {retry_count}/{max_retries}). "
                    f"Retrying in {current_delay} seconds..."
                )
                time.sleep(current_delay)
                current_delay *= (
                    backoff_factor * 2
                )  # More aggressive backoff for rate limits
            else:
                # Other API errors - don't retry
                logger.error(f"API error: {str(e)}")
                raise RuntimeError(f"LLM API error: {str(e)}")

    # This should never be reached due to the exception in the loop
    raise RuntimeError("Unexpected error in LLM request retry loop")
