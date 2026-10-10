from __future__ import annotations


class UnsupportedTextFileError(Exception):
    """Raised when a file does not appear to contain supported text content."""


class ChromaPathError(Exception):
    """Raised when the configured Chroma persistence path is invalid or unusable."""


class EmbeddingFunctionError(Exception):
    """
    Raised when an embedding function cannot be built or its token budget cannot
    be resolved.
    """
