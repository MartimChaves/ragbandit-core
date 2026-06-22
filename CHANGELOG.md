# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-06-22

### Changed
- **DatalabOCR**: migrated from the deprecated `/marker` endpoint to the new
  `/convert` endpoint (`POST https://www.datalab.to/api/v1/convert`). The submit
  → poll → markdown flow is unchanged.

### Removed
- **DatalabOCR**: removed the `model` parameter (and `VALID_MODELS`). The convert
  API no longer accepts a model; Datalab selects the underlying model from the
  `mode` (`fast` / `balanced` / `accurate`). `OCRResult.model` is now reported as
  `"datalab"`.

## [0.4.0] - 2026-06-15

### Added
- **VoyageAIEmbedder**: Added `voyage-3.5` (now the default) and `voyage-3.5-lite`
  models. Both output 1024 dimensions and match the price of the `voyage-3`
  family ($0.06 / $0.02 per 1M tokens).

### Changed
- **Mistral pricing** (`config/pricing.py`): updated `mistral-medium-latest` to
  Mistral Medium 3.5 pricing ($1.5 / $7.5 per 1M tokens).
- **MAX_PROMPT_TOKENS** (`config/llms.py`): refreshed context windows for the
  current Mistral `-latest` aliases (Small 4: 256k, Medium 3.5 / Large 3: 128k).

### Removed
- **MistralOCR**: removed `mistral-ocr-2505` (deprecated by Mistral, retired
  2026-02-27). `mistral-ocr-2512` (OCR 3) is the only supported model.
- **CohereEmbedder**: removed deprecated `embed-english-v3.0` and
  `embed-multilingual-v3.0`. `embed-v4.0` is the only supported model.
- **MAX_PROMPT_TOKENS**: removed dead OpenAI entries (`gpt-3.5-turbo`, `gpt-4`,
  `gpt-4-turbo`); `gpt-4` was retired from the OpenAI API and none were used.

## [0.3.0] - 2026-03-09

### Added
- **SentenceChunker**: Sentence-aware sliding-window chunker with configurable `sentences_per_chunk`, `sentence_overlap`, and `min_chunk_size`. Pure-Python regex splitting, no external dependencies.
- **RecursiveMarkdownChunker**: Hierarchical chunker that splits by heading level (H1→H2→H3→H4), then paragraph, then sentence, with overlap merging.
- **VoyageAIEmbedder**: Embedder using Voyage AI models — `voyage-3`, `voyage-3-large`, `voyage-3-lite`.
- **CohereEmbedder**: Embedder using Cohere models — `embed-v4.0`, `embed-english-v3.0`, `embed-multilingual-v3.0`. Supports configurable `input_type` for search documents vs queries.
- **TableOfContentsRefiner**: Refiner that detects and removes the table of contents section, storing it in `extracted_data["toc_markdown"]`.
- **Example scripts** (`examples/`): 4 runnable Python scripts demonstrating basic pipeline usage, component comparison, step-by-step execution, and cost tracking.
- **Jupyter notebooks** (`notebooks/`): 3 interactive notebooks — getting started walkthrough, chunker comparison, and a component explorer exercising every component with all valid configurations.
- **README component reference tables**: OCR, Refiners, Chunkers, and Embedders with models, parameters, defaults, and costs per 1M tokens.
- **README examples & notebooks section** linking to all new files with descriptions.

### Changed
- Updated README package layout tree to include `examples/`, `notebooks/`, and expanded `src/ragbandit/` subdirectories

### Dependencies
- Added `voyageai` for Voyage AI embeddings support
- Added `cohere` for Cohere embeddings support

## [0.2.6] - 2026-02-27

### Added
- Added `tiktoken>=0.12.0` runtime dependency.

## [0.2.5] - 2026-02-26

### Removed
- Removed encryption/decryption support (`SecureFileHandler`, `encrypted` parameter in OCR processors). This was unnecessary bloat for a RAG package.
- Removed `cryptography` runtime dependency.

## [0.2.4] - 2026-02-25

### Fixed
- Updated LLM token pricing in `pricing.py` to reflect current Mistral AI rates (February 2026). Previous values were significantly outdated.

## [0.2.3] - 2026-02-25

### Changed
- Renamed `MistralOCRDocument` to `MistralOCR` across the codebase

## [0.2.2] - 2026-02-23

### Changed
- JSON/Validation errors in `query_llm` now skip retries on the same model and escalate directly to the next model in the chain. Retrying the same model with the same prompt is unlikely to fix structural response issues.

## [0.2.1] - 2026-02-22

### Fixed
- Fixed `ValidationError` not being retried in `query_llm` when LLM returns valid JSON but missing required schema fields (e.g. `reason`). Now triggers retry and model escalation like `JSONDecodeError`.

## [0.2.0] - 2026-02-03

### Added
- **DatalabOCR**: New OCR processor using Datalab API with marker model
  - Supports three processing modes: fast, balanced, and accurate
  - Page range filtering and max pages limiting
  - Image extraction with optional captions
  - Cost tracking per page processed
- **OpenAIEmbedder**: New embedder using OpenAI's embedding models
  - Support for `text-embedding-3-small` (1536 dimensions)
  - Support for `text-embedding-3-large` (3072 dimensions)
  - Normalized vector embeddings with L2 norm ≈ 1
  - Token usage tracking for embedding operations
- Comprehensive integration tests for both new components
  - Regular functionality tests
  - Behavior tests ensuring embedding quality and OCR accuracy
  - Validation of embedding dimensions, normalization, and similarity properties
- Updated README with examples for DatalabOCR and OpenAIEmbedder
- Added section on using alternative OCR and embedding providers

### Changed
- Updated test coverage to 87% (from previous coverage)
- Enhanced test suite with behavior-driven tests

### Dependencies
- Added `openai>=1.0.0` for OpenAI embeddings support
- Added `datalab-python-sdk>=0.2.1` for Datalab OCR support

## [0.1.2] - 2026-02-03

### Fixed
- Fixed endless loop in `SemanticChunker` when break point index was 0

## [0.1.1]

### Changed
- Updated README documentation
- Package setup and configuration improvements

## [0.1.0]

### Added
- Initial release
- MistralOCRDocument for OCR processing
- MistralEmbedder for document embeddings
- DocumentPipeline for orchestrating document processing
- SemanticChunker and FixedSizeChunker for document chunking
- ReferencesRefiner and FootnoteRefiner for document refinement
- Token usage tracking utilities
- Basic schema definitions with Pydantic models

**Note**: Versions 0.1.0 and 0.1.1 were initial setup releases. Version 0.1.2 is the first production-ready release with comprehensive test coverage.

[Unreleased]: https://github.com/MartimChaves/ragbandit-core/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/MartimChaves/ragbandit-core/compare/v0.2.6...v0.3.0
[0.2.6]: https://github.com/MartimChaves/ragbandit-core/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/MartimChaves/ragbandit-core/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/MartimChaves/ragbandit-core/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/MartimChaves/ragbandit-core/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/MartimChaves/ragbandit-core/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/MartimChaves/ragbandit-core/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/MartimChaves/ragbandit-core/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/MartimChaves/ragbandit-core/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MartimChaves/ragbandit-core/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MartimChaves/ragbandit-core/releases/tag/v0.1.0
