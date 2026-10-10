from __future__ import annotations

from typing import Any, TypeAlias

from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)

from chromy.errors import EmbeddingFunctionError

Embedder: TypeAlias = EmbeddingFunction[Any]

DEFAULT_SPEC = "default"
SENTENCE_TRANSFORMER_PREFIX = "sentence-transformers"
# Single source of truth for the user-facing model-spec list; reused in the
# CLI help text, build_embedding_function's docstring, and error messages.
MODEL_SPEC_EXAMPLES = f"'{DEFAULT_SPEC}' or '{SENTENCE_TRANSFORMER_PREFIX}:<model>'"

_CHROMA_NAME_TO_SPEC_PREFIX = {
    "sentence_transformer": SENTENCE_TRANSFORMER_PREFIX,
}


def build_embedding_function(spec: str) -> Embedder:
    """
    Build the Chroma embedding function named by ``spec``.

    Supported specs: ``default`` and ``sentence-transformers:<model>``.

    Args:
        spec (str): The model specification.

    Raises:
        EmbeddingFunctionError: If ``spec`` is not a known model specification,
            or the model's backend dependency is missing or cannot be
            initialized.

    Returns:
        Embedder: The built embedding function.

    Supported specs: {MODEL_SPEC_EXAMPLES}.
    """
    if spec == DEFAULT_SPEC:
        return DefaultEmbeddingFunction()

    prefix, separator, value = spec.partition(":")
    if not separator or not value:
        raise EmbeddingFunctionError(_unknown_model_message(spec))

    if prefix == SENTENCE_TRANSFORMER_PREFIX:
        try:
            return SentenceTransformerEmbeddingFunction(model_name=value)
        except (ImportError, ValueError) as exc:
            raise EmbeddingFunctionError(
                f"Could not initialize sentence-transformers embedding model "
                f"'{value}': {exc}"
            ) from exc

    raise EmbeddingFunctionError(_unknown_model_message(spec))


def model_max_tokens(embedding_function: Embedder) -> int | None:
    """
    Return the embedding function's maximum input length, if it reports one.

    Args:
        embedding_function (Embedder): The embedding function to query.

    Returns:
        int | None: The token limit, or ``None`` if unknown.
    """
    getter = getattr(embedding_function, "max_tokens", None)
    if getter is None:
        return None

    try:
        return int(getter())
    except (TypeError, ValueError):
        return None


def spec_from_embedding_function(embedding_function: Embedder) -> str:
    """
    Return the CLI spec that recreates ``embedding_function``, for display.

    Args:
        embedding_function (Embedder): The embedding function to name.

    Returns:
        str: The model spec, e.g. ``"default"`` or
        ``"sentence-transformers:all-MiniLM-L6-v2"``.
    """
    registered_name = _registered_name(embedding_function)
    if registered_name in ("default", "onnx_mini_lm_l6_v2"):
        return DEFAULT_SPEC

    prefix = _CHROMA_NAME_TO_SPEC_PREFIX.get(
        registered_name or "", registered_name or "unknown"
    )

    model_name = getattr(embedding_function, "model_name", None) or getattr(
        embedding_function, "model", None
    )
    if model_name:
        return f"{prefix}:{model_name}"

    return prefix


def _registered_name(embedding_function: Embedder) -> str | None:
    name_func = getattr(type(embedding_function), "name", None)
    if name_func is None:
        return None

    try:
        name = name_func()
    except TypeError:
        return None

    return str(name) if name else None


def _unknown_model_message(spec: str) -> str:
    return f"Unknown embedding model '{spec}'. Expected {MODEL_SPEC_EXAMPLES}."
