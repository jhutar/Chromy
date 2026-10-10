from __future__ import annotations

from pathlib import Path

from chromadb import QueryResult

from chromy.chroma_functions import (
    add_data,
    delete_data,
    has_data_for_file,
    query_data,
)
from chromy.chunking import chunk_file
from chromy.embedding import embed


def parse_key_value_pairs(s: str, option_name: str) -> dict[str, str]:
    """
    Parse a comma-separated ``key=value`` string into a dict.

    Splits on the first ``=`` of each item so values may themselves contain
    ``=``. Keys and values are stripped; an empty or whitespace-only value, an
    item without ``=``, an empty key, or an empty value raises ``ValueError``.
    Duplicate keys are resolved by the last occurrence.

    Args:
        s (str): The comma-separated ``key=value`` string to parse.
        option_name (str): The CLI option name, used in the error message.

    Raises:
        ValueError: If any item is malformed.

    Returns:
        dict[str, str]: The parsed key-value pairs.
    """

    message = (
        f"Invalid {option_name} value. Expected comma-separated <key>=<value> pairs."
    )

    if not s.strip():
        raise ValueError(message)

    pairs: dict[str, str] = {}

    for item in s.split(","):
        key, separator, value = item.partition("=")
        key = key.strip()
        value = value.strip()

        if separator == "" or not key or not value:
            raise ValueError(message)

        pairs[key] = value

    return pairs


def ingest_file(
    collection_name: str,
    file_path: str,
    extra_metadata: dict[str, str] | None = None,
) -> int:
    metadata = extra_metadata or {}
    effective_file_name = metadata.get("file_name", file_path)

    if has_data_for_file(collection_name, effective_file_name):
        delete_data(collection_name, {"file_name": effective_file_name})

    chunks = chunk_file(file_path)
    embeddings = embed(chunks)
    add_data(collection_name, embeddings, effective_file_name, metadata)
    return len(embeddings)


def run_query(
    collection_name: str,
    query_text: str,
    where: dict[str, str] | None = None,
) -> QueryResult:
    return query_data(collection_name, [query_text], where)


def is_probably_text_file(path: str | Path, sample_size: int = 8192) -> bool:
    """
    Return whether a file appears to contain text.

    Args:
        path (str | Path): The path to the file to inspect.
        sample_size (int): The maximum number of bytes to read from the file.

    Returns:
        bool: ``True`` if the sampled bytes decode as UTF-8, UTF-8 with BOM,
        UTF-16, or UTF-32, or if the file is empty. Otherwise, ``False``.
    """

    path = Path(path)

    with path.open("rb") as f:
        sample = f.read(sample_size)

    if not sample:
        return True

    encodings = (
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "utf-32",
    )

    for encoding in encodings:
        try:
            sample.decode(encoding)
            return True
        except UnicodeDecodeError:
            pass

    return False
