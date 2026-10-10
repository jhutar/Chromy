from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypedDict

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from chromy.embedding.registry import (
    Embedder,
    model_max_tokens,
    spec_from_embedding_function,
)
from chromy.errors import EmbeddingFunctionError

# semchunk counts tokens with tiktoken, which differs from the embedder's own
# (wordpiece) tokenizer. The headroom keeps real token counts under the
# embedder's input limit despite that counting skew.
MODEL_HEADROOM: Final = 0.8


class EmbeddingRecord(TypedDict):
    text: str
    embedding: list[float]


def embedding_budget_tokens(
    embedding_function: Embedder,
    max_tokens: int | None = None,
) -> int:
    """
    Return the per-chunk token budget for the given embedder.

    Scales the embedder's reported maximum input length by ``MODEL_HEADROOM``
    so that tiktoken-measured chunks stay within the embedder's actual token
    limit. When the embedder does not report a limit, ``max_tokens`` (typically
    the stored ``chromy_max_tokens`` collection metadata) is used instead.

    Args:
        embedding_function (EmbeddingFunction): The active embedder.
        max_tokens (int | None): An explicit token limit, used when the
            embedder does not report one.

    Raises:
        EmbeddingFunctionError: If no token limit can be determined.

    Returns:
        int: Maximum number of (tiktoken-counted) tokens per chunk.
    """

    model_limit = model_max_tokens(embedding_function)
    if model_limit is None:
        model_limit = max_tokens
    if model_limit is None:
        model = spec_from_embedding_function(embedding_function)
        raise EmbeddingFunctionError(
            f"Embedding model '{model}' does not report a maximum input length "
            "and no stored --max-tokens value is available. Re-create the "
            "collection and pass --max-tokens N to `create-collection`."
        )

    return max(1, int(model_limit * MODEL_HEADROOM))


def embed(
    chunks: Sequence[str],
    embedding_function: Embedder | None = None,
) -> list[EmbeddingRecord]:
    """
    Embed ``chunks`` into vector records.

    Args:
        chunks (Sequence[str]): The texts to embed.
        embedding_function (EmbeddingFunction | None): The embedder to use;
            defaults to the default ONNX MiniLM embedder.

    Returns:
        list[EmbeddingRecord]: One record per chunk.
    """

    if not chunks:
        return []

    embedding_function = embedding_function or DefaultEmbeddingFunction()
    embeddings = embedding_function(list(chunks))

    return [
        {
            "text": text,
            "embedding": (
                embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            ),
        }
        for text, embedding in zip(chunks, embeddings, strict=False)
    ]
