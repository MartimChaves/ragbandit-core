# Ragbandit Core - Implementation Roadmap

This document outlines the tasks needed to standardize and improve the ragbandit-core package based on the December 2025 analysis.

## Phase 1: Result Schema Standardization (Critical)

### Task 1.0: Terminology Refactor (Foundation) ✅ COMPLETE
- [x] Rename `ProcessingResult` to `RefiningResult` in schema
- [x] Rename `ProcessedPage` to `RefinedPage` in schema
- [x] Rename `ProcessingTraceItem` to `RefiningTraceItem` in schema
- [x] Rename `BaseProcessor` to `BaseRefiner` in base class
- [x] Update all processor implementations to inherit from `BaseRefiner`
- [x] Rename `ReferencesProcessor` to `ReferencesRefiner`
- [x] Rename `FootnoteProcessor` to `FootnoteRefiner`
- [x] Update `DocumentPipeline` to use "refining" terminology
- [x] Rename `run_processors()` to `run_refiners()` in pipeline
- [x] Update pipeline step key from `"processing"` to `"refining"`
- [x] Update `StepReport.processing` to `StepReport.refining`
- [x] Update `TimingMetrics.processing` to `TimingMetrics.refining`
- [x] Update all docstrings and comments to use "refining" terminology
- [x] Update README.md examples to use new terminology
- [x] Update imports in `__init__.py` files
- [x] Rename `processors/` directory to `refiners/`
- [x] Rename `*_processor.py` files to `*_refiner.py`
- [x] Rename `*_processor_tools.py` files to `*_refiner_tools.py`
- [x] Update variable names: `proc_*` to `ref_*`
- [x] Update chunkers to use `RefiningResult`

**Rationale**: Eliminates terminology conflict between "Document Processing" (overall pipeline) and "Processing" (specific step). New terminology: OCR → Refining → Chunking → Embedding.

### Task 1.1: Update Schema Definitions
- [ ] Add `component_name: str` field to `OCRResult`
- [ ] Add `component_config: dict` field to `OCRResult`
- [ ] Rename `refiner_name` to `component_name` in `RefiningResult` (for consistency)
- [ ] Add `component_config: dict` field to `RefiningResult`
- [ ] Add `component_name: str` field to `ChunkingResult`
- [ ] Add `component_config: dict` field to `ChunkingResult`
- [ ] Add `component_name: str` field to `EmbeddingResult`
- [ ] Add `component_config: dict` field to `EmbeddingResult` (note: already has `model_name`, keep both)

### Task 1.2: Add Configuration Methods to Base Classes
- [ ] Add `get_config() -> dict` abstract method to `BaseOCR`
- [ ] Add `get_name() -> str` method to `BaseOCR` (default: `self.__class__.__name__`)
- [ ] Add `get_config() -> dict` abstract method to `BaseRefiner`
- [ ] Add `get_name() -> str` method to `BaseRefiner`
- [ ] Add `get_config() -> dict` abstract method to `BaseChunker`
- [ ] Add `get_name() -> str` method to `BaseChunker`
- [ ] Add `get_config() -> dict` abstract method to `BaseEmbedder`
- [ ] Add `get_name() -> str` method to `BaseEmbedder`

### Task 1.3: Implement Configuration Methods in OCR Components
- [ ] Implement `get_config()` in `MistralOCRDocument`
- [ ] Update `MistralOCRDocument.process()` to populate `component_name` and `component_config` in `OCRResult`

### Task 1.4: Implement Configuration Methods in Refiners
- [ ] Implement `get_config()` in `ReferencesRefiner`
- [ ] Implement `get_config()` in `FootnoteRefiner`
- [ ] Update refiner implementations to populate `component_name` and `component_config` in `RefiningResult`

### Task 1.5: Implement Configuration Methods in Chunkers
- [ ] Implement `get_config()` in `FixedSizeChunker`
- [ ] Implement `get_config()` in `SemanticChunker`
- [ ] Update chunker implementations to populate `component_name` and `component_config` in `ChunkingResult`

### Task 1.6: Implement Configuration Methods in Embedders
- [ ] Implement `get_config()` in `MistralEmbedder`
- [ ] Update embedder implementations to populate `component_name` and `component_config` in `EmbeddingResult`

### Task 1.7: Update DocumentPipeline
- [ ] Replace `str()` calls with `.get_config()` calls in `DocumentPipeline.process()`
- [ ] Update `pipeline_config` dict to use structured configuration data
- [ ] Ensure all result objects have proper `component_name` and `component_config` populated

## Phase 2: Consistency and Quality Improvements

### Task 2.1: Standardize String Representations
- [ ] Add `__str__()` method to `BaseEmbedder` (currently missing)
- [ ] Verify all base classes have both `__str__()` and `__repr__()` methods
- [ ] Ensure consistent formatting across all implementations

### Task 2.2: Improve Error Handling
- [ ] Create `ragbandit/exceptions.py` module
- [ ] Define `RagbanditError` base exception class
- [ ] Define `ConfigurationError` exception class
- [ ] Define `APIKeyError` exception class
- [ ] Define `ProcessingError` exception class
- [ ] Update components to use custom exceptions instead of generic `ValueError`

### Task 2.3: Add Configuration Validation
- [ ] Create `ragbandit/validation.py` module
- [ ] Add `validate_api_key()` function
- [ ] Add `validate_config()` function for each component type
- [ ] Add validation calls in component constructors
- [ ] Add helpful error messages for common configuration issues

### Task 2.4: API Key Handling Improvements
- [ ] Document that base classes don't require `api_key` parameter (design decision)
- [ ] Ensure concrete implementations that need API keys validate them properly
- [ ] Add clear error messages when API key is required but not provided
- [ ] Update docstrings to clarify when API keys are required

## Phase 3: Testing and Documentation

### Task 3.1: Create Test Infrastructure
- [ ] Create `tests/` directory structure
- [ ] Set up pytest configuration
- [ ] Create test fixtures for common test data
- [ ] Add mock API key utilities for testing

### Task 3.2: Add Unit Tests for Schema
- [ ] Test `OCRResult` with `component_name` and `component_config`
- [ ] Test `ProcessingResult` with `component_name` and `component_config`
- [ ] Test `ChunkingResult` with `component_name` and `component_config`
- [ ] Test `EmbeddingResult` with `component_name` and `component_config`
- [ ] Test schema serialization/deserialization

### Task 3.3: Add Unit Tests for Components
- [ ] Test `get_config()` methods for all OCR implementations
- [ ] Test `get_config()` methods for all processor implementations
- [ ] Test `get_config()` methods for all chunker implementations
- [ ] Test `get_config()` methods for all embedder implementations
- [ ] Test configuration validation logic

### Task 3.4: Add Integration Tests
- [ ] Test full pipeline with configuration tracking
- [ ] Test pipeline result serialization
- [ ] Test pipeline reconstruction from configuration
- [ ] Test error handling across pipeline stages

### Task 3.5: Update Documentation
- [ ] Update README.md with new configuration examples
- [ ] Document `component_name` and `component_config` fields
- [ ] Add configuration schema documentation
- [ ] Add examples of accessing configuration from results
- [ ] Document custom exceptions and when they're raised
- [ ] Add troubleshooting guide for common configuration issues

### Task 3.6: Add Type Hints and Validation
- [ ] Run mypy on codebase and fix type issues
- [ ] Add missing type hints to all public methods
- [ ] Ensure Pydantic models have proper validation
- [ ] Add runtime type checking for critical paths

## Phase 4: Web App Integration Support

### Task 4.1: Configuration Serialization
- [ ] Add `to_json()` method to `DocumentPipelineResult`
- [ ] Add `from_json()` class method to `DocumentPipelineResult`
- [ ] Ensure all configuration dicts are JSON-serializable
- [ ] Test round-trip serialization/deserialization

### Task 4.2: Pipeline Reconstruction
- [ ] Create `PipelineFactory` class for building pipelines from config
- [ ] Add `from_config()` class methods to all component classes
- [ ] Add validation for reconstructed pipelines
- [ ] Document pipeline reconstruction workflow

### Task 4.3: Configuration Display Utilities
- [ ] Create `format_config()` utility for human-readable config display
- [ ] Add configuration comparison utilities
- [ ] Create configuration diff utilities for comparing pipeline runs
- [ ] Add configuration export utilities (JSON, YAML)

## Phase 5: Performance and Optimization

### Task 5.1: Performance Profiling
- [ ] Add timing metrics for `get_config()` calls
- [ ] Profile configuration serialization overhead
- [ ] Identify bottlenecks in pipeline execution
- [ ] Add performance benchmarks

### Task 5.2: Caching Improvements
- [ ] Review `MistralClientManager` caching strategy
- [ ] Add configuration caching where appropriate
- [ ] Optimize repeated configuration access
- [ ] Add cache invalidation logic

### Task 5.3: Memory Optimization
- [ ] Review memory usage in pipeline execution
- [ ] Optimize large result object handling
- [ ] Add streaming support for large documents
- [ ] Implement result object cleanup strategies

## Phase 6: Advanced Features

### Task 6.1: Configuration Versioning
- [ ] Add `config_version` field to all result schemas
- [ ] Implement configuration migration utilities
- [ ] Add backward compatibility support
- [ ] Document configuration version history

### Task 6.2: Configuration Presets
- [ ] Create common configuration presets (e.g., "fast", "accurate", "balanced")
- [ ] Add preset loading utilities
- [ ] Document available presets
- [ ] Allow custom preset registration

### Task 6.3: Pipeline Monitoring
- [ ] Add configuration change tracking
- [ ] Add pipeline execution history
- [ ] Create configuration audit logs
- [ ] Add configuration validation reports

## Notes

- **Phase 1** is critical for web app integration and should be completed first
- **Phase 2** improves code quality and developer experience
- **Phase 3** ensures reliability and maintainability
- **Phase 4** enables web app features
- **Phase 5** optimizes performance
- **Phase 6** adds advanced capabilities

## Dependencies

- Phase 2 can start after Phase 1 Task 1.1 is complete
- Phase 3 should start after Phase 1 is mostly complete
- Phase 4 requires Phase 1 to be complete
- Phase 5 can be done in parallel with Phase 3-4
- Phase 6 should wait until Phase 1-4 are stable
