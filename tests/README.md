# Ragbandit Core Tests

This directory contains unit and integration tests for the ragbandit-core package.

## Setup

1. Install the package in editable mode:
```bash
pip install -e .
```

2. For integration tests that require API calls, set your API key:
```bash
export MISTRAL_API_KEY="your_api_key_here"
```

3. For OCR tests, place a sample PDF in `tests/fixtures/sample.pdf`

## Running Tests

### Run all tests
```bash
python -m pytest
```

### Run only unit tests (no API calls)
```bash
python -m pytest -m unit
```

### Run only integration tests (requires API key)
```bash
python -m pytest -m integration
```

### Run with verbose output
```bash
python -m pytest -v
```

### Run specific test file
```bash
python -m pytest tests/unit/test_schema.py
python -m pytest tests/integration/test_ocr.py
```

### Run with coverage report
```bash
# Terminal report with missing lines
python -m pytest --cov=ragbandit --cov-report=term-missing

# HTML report (opens in browser)
python -m pytest --cov=ragbandit --cov-report=html
# Then open htmlcov/index.html

# Both terminal and HTML
python -m pytest tests/ --cov=ragbandit --cov-report=term-missing --cov-report=html
```

## Current Coverage: 87%

The test suite maintains **87% code coverage** across all modules:

- **DocumentPipeline**: 91% (30 tests)
- **Refiners**: 85-92% (FootnoteRefiner, ReferencesRefiner)
- **Chunkers**: 85-94% (FixedSizeChunker, SemanticChunker)
- **Embedders**: 88-91% (MistralEmbedder)
- **OCR**: 97% (MistralOCR)
- **Schema**: 100% (all Pydantic models)
- **Prompt Tools**: 100% (all tools)

Coverage reports are generated locally and **not committed to git** (see `.gitignore`).

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests (no external dependencies)
│   └── test_schema.py       # Schema validation tests
├── integration/             # Integration tests (real API calls)
│   ├── test_ocr.py          # OCR processor tests
│   ├── test_refiners.py     # Refiner tests
│   ├── test_chunkers.py     # Chunker tests
│   ├── test_embedders.py    # Embedder tests
│   └── test_pipeline.py     # Full pipeline integration tests
└── fixtures/                # Test data files
    └── sample.pdf           # Sample PDF for OCR tests
```

## Test Coverage

### Unit Tests
- ✅ Schema models (OCRResult, RefiningResult, ChunkingResult, EmbeddingResult)
- ✅ Schema validation and serialization

### Integration Tests

#### OCR Processors
- ✅ MistralOCR with default model
- ✅ MistralOCR with specific model
- ✅ Model validation
- ✅ Configuration methods
- ✅ Token usage metrics
- ✅ Multi-page processing

#### Refiners
- ✅ FootnoteRefiner processing
- ✅ FootnoteRefiner extraction
- ✅ ReferencesRefiner processing
- ✅ ReferencesRefiner extraction
- ✅ Refiner chaining
- ✅ Configuration methods

#### Chunkers
- ✅ FixedSizeChunker with various sizes
- ✅ FixedSizeChunker with/without overlap
- ✅ SemanticChunker with different thresholds
- ✅ Chunker comparison
- ✅ Configuration methods

#### Embedders
- ✅ MistralEmbedder with default model
- ✅ MistralEmbedder with specific model
- ✅ Model validation
- ✅ Cosine similarity/distance calculations
- ✅ Embedding dimensions
- ✅ Metadata preservation
- ✅ Edge cases (empty chunks, single chunk)
- ✅ Configuration methods

#### DocumentPipeline
- ✅ Pipeline initialization with all components
- ✅ Individual step execution (run_ocr, run_refiners, run_chunker, run_embedder)
- ✅ Full end-to-end pipeline.process() with real PDF
- ✅ Configuration tracking and serialization
- ✅ Metrics aggregation across steps
- ✅ Timing tracking for all steps
- ✅ Step status tracking (success/failed)
- ✅ Log capture during execution
- ✅ Error handling (missing components)
- ✅ Different chunker types (FixedSize, Semantic)
- ✅ Pipeline with/without refiners

## Cost Considerations

Integration tests make real API calls and will incur costs:

- **OCR tests**: ~$0.01-0.05 per test (depending on PDF size)
- **Refiner tests**: ~$0.001-0.01 per test
- **Embedder tests**: ~$0.001-0.005 per test

Total cost for full integration test suite: **~$0.10-0.50 per run**

To minimize costs during development:
- Run unit tests frequently: `pytest -m unit`
- Run integration tests selectively: `pytest tests/integration/test_embedders.py`
- Use the `-k` flag to run specific tests: `pytest -k "test_embedder_get_config"`

## Writing New Tests

### Unit Tests
- Place in `tests/unit/`
- Mark with `@pytest.mark.unit`
- Should not require external dependencies or API calls
- Use fixtures from `conftest.py`

### Integration Tests
- Place in `tests/integration/`
- Mark with `@pytest.mark.integration`
- Can make real API calls
- Should check for API keys and skip if not available
- Test real component behavior end-to-end

### Example Test
```python
import pytest
from ragbandit.documents.embedders.mistral_embedder import MistralEmbedder

@pytest.mark.integration
def test_embedder_basic(mistral_api_key):
    """Test basic embedding functionality."""
    embedder = MistralEmbedder(api_key=mistral_api_key)
    # ... test implementation
```

## CI/CD Considerations

For continuous integration:
- Run unit tests on every commit (fast, free)
- Run integration tests on pull requests or scheduled (slower, costs money)
- Use separate API keys for CI/CD with rate limits
- Consider mocking API calls for CI if costs are a concern
