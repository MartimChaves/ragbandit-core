"""ragbandit core package.

This package contains sub-modules for document processing,
RAG pipeline configuration/execution, and evaluation utilities.
Only lightweight interfaces and shared utilities are defined here;
heavy logic resides in sub-packages.
"""

from importlib import metadata as _metadata

__version__: str
try:
    __version__ = _metadata.version("ragbandit-core")
except _metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+dev"

# Re-export public interfaces so that users can simply:
#   from ragbandit import DocumentProcessor, RAGConfig, RAGPipeline, evaluate

# from ragbandit.documents import DocumentPipeline


__all__ = [
    "__version__",
    # "DocumentPipeline",
]
