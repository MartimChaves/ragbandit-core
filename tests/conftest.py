"""
Shared pytest fixtures for ragbandit-core tests.

This module provides common fixtures used across unit and integration tests.
"""

import pytest
from datetime import datetime, timezone


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test_api_key_12345"


@pytest.fixture
def sample_timestamp():
    """Provide a consistent timestamp for testing."""
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_markdown_text():
    """Provide sample markdown text for testing."""
    return """# Sample Document

This is a sample document with some text.

## Section 1

Some content in section 1.

## Section 2

Some content in section 2.

### Subsection 2.1

More detailed content here.
"""


@pytest.fixture
def sample_ocr_page_data():
    """Provide sample OCR page data for testing."""
    return {
        "index": 0,
        "markdown": "# Page 1\n\nThis is page 1 content.",
        "images": [],
        "dimensions": {
            "width": 612.0,
            "height": 792.0,
        }
    }


@pytest.fixture
def sample_chunk_data():
    """Provide sample chunk data for testing."""
    return {
        "text": "This is a sample chunk of text.",
        "metadata": {
            "page_index": 0,
            "chunk_index": 0,
        }
    }
