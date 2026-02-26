# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/MartimChaves/ragbandit-core/compare/v0.2.5...HEAD
[0.2.5]: https://github.com/MartimChaves/ragbandit-core/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/MartimChaves/ragbandit-core/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/MartimChaves/ragbandit-core/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/MartimChaves/ragbandit-core/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/MartimChaves/ragbandit-core/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/MartimChaves/ragbandit-core/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/MartimChaves/ragbandit-core/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MartimChaves/ragbandit-core/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MartimChaves/ragbandit-core/releases/tag/v0.1.0
