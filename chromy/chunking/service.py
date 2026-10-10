from __future__ import annotations

from pathlib import Path
from typing import cast

from semchunk import semchunk


def chunk_text(text: str, chunk_size: int) -> list[str]:
    """
    Split ``text`` into semantically meaningful chunks.

    Args:
        text (str): The text to split.
        chunk_size (int): Maximum tokens per chunk, as measured by tiktoken.
            Callers are responsible for sizing it to their embedder's budget.

    Returns:
        list[str]: The resulting chunks.
    """

    chunker = semchunk.chunkerify("gpt-4", chunk_size)
    chunks = chunker(text)

    return cast("list[str]", chunks)


def chunk_file(filename: str, chunk_size: int) -> list[str]:
    """
    Read and chunk the file at ``filename``.

    Args:
        filename (str): Path to the file to chunk.
        chunk_size (int): Maximum tokens per chunk, as measured by tiktoken.
            Callers are responsible for sizing it to their embedder's budget.

    Returns:
        list[str]: The resulting chunks.
    """

    contents = Path(filename).read_text()

    return chunk_text(contents, chunk_size)
