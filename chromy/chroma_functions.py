from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast
from uuid import uuid4

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import QueryResult, Where
from chromadb.errors import NotFoundError

from chromy.embedding import EmbeddingRecord
from chromy.errors import ChromaPathError

CHROMA_FOLDER_ENV_VAR = "CHROMA_FOLDER"
CHROMA_SUBDIRECTORY = "chroma"


def _resolve_persistence_path() -> Path | None:
    configured_parent = os.getenv(CHROMA_FOLDER_ENV_VAR)

    if configured_parent is None:
        return None

    trimmed_parent = configured_parent.strip()
    if not trimmed_parent:
        raise ChromaPathError(
            f"{CHROMA_FOLDER_ENV_VAR} is set but empty. Please set a valid parent "
            "directory path."
        )

    parent_path = Path(trimmed_parent).expanduser().resolve()
    return parent_path / CHROMA_SUBDIRECTORY


def _ensure_persistence_path_is_usable(path: Path, configured_parent: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)

        if not path.is_dir():
            raise ChromaPathError(
                f"Configured Chroma directory '{path}' is not a directory."
            )

        with NamedTemporaryFile(dir=path, prefix=".chromy-write-test-", delete=True):
            pass
    except ChromaPathError:
        raise
    except OSError as exc:
        raise ChromaPathError(
            f"Could not create or access Chroma directory '{path}' from "
            f"{CHROMA_FOLDER_ENV_VAR}='{configured_parent}': {exc}"
        ) from exc


def get_client() -> ClientAPI:
    persistence_path = _resolve_persistence_path()

    if persistence_path is None:
        return chromadb.PersistentClient()

    configured_parent = os.getenv(CHROMA_FOLDER_ENV_VAR, "")
    _ensure_persistence_path_is_usable(persistence_path, configured_parent)

    try:
        return chromadb.PersistentClient(path=str(persistence_path))
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise ChromaPathError(
            f"Could not initialize Chroma client at '{persistence_path}' from "
            f"{CHROMA_FOLDER_ENV_VAR}='{configured_parent}': {exc}"
        ) from exc


def _get_client_and_collection(
    collection_name: str,
) -> tuple[ClientAPI, chromadb.Collection]:
    client = get_client()

    try:
        collection = client.get_collection(name=collection_name)
    except NotFoundError:
        raise

    return client, collection


def list_collections() -> list[str]:
    client = get_client()
    collections = client.list_collections()

    if not collections:
        return []

    return [getattr(collection, "name", str(collection)) for collection in collections]


def create_collection(name: str) -> str:
    client = get_client()
    collection = client.create_collection(name=name)

    return getattr(collection, "name", name)


def delete_collection(name: str) -> None:
    client = get_client()
    client.delete_collection(name=name)


def build_where(flat: Mapping[str, str]) -> Where:
    """
    Convert flat key-value pairs into a Chroma ``Where`` filter.

    Chroma's ``validate_where`` requires exactly one key per dict level, so a
    single pair is returned as-is while two or more pairs are wrapped in an
    ``$and`` expression. This is shared by ``query_data`` and ``delete_data``.

    Args:
        flat (Mapping[str, str]): The parsed key-value pairs to filter on.

    Returns:
        Where: A Chroma-compatible ``Where`` expression.
    """

    if len(flat) == 1:
        return cast(Where, dict(flat))

    return cast(Where, {"$and": [{key: value} for key, value in flat.items()]})


def delete_data(collection_name: str, where: Mapping[str, str]) -> int:
    _, collection = _get_client_and_collection(collection_name)
    result = collection.delete(where=build_where(where))

    return int(result.get("deleted", 0))


def has_data_for_file(collection_name: str, file_name: str) -> bool:
    _, collection = _get_client_and_collection(collection_name)
    result = collection.get(where=cast(Where, {"file_name": file_name}))
    ids = result.get("ids", [])

    return len(ids) > 0


def count_collection(collection_name: str) -> int:
    _, collection = _get_client_and_collection(collection_name)
    return collection.count()


def add_data(
    collection_name: str,
    data: Sequence[EmbeddingRecord],
    file_name: str,
    extra_metadata: Mapping[str, str] | None = None,
) -> None:
    if not data:
        return

    _, collection = _get_client_and_collection(collection_name)

    embeddings: list[Sequence[float]] = [record["embedding"] for record in data]
    extra = dict(extra_metadata or {})

    collection.add(
        ids=[str(uuid4()) for _ in data],
        metadatas=[{**extra, "file_name": file_name} for _ in data],
        documents=[record["text"] for record in data],
        embeddings=embeddings,
    )


def query_data(
    collection_name: str,
    texts: Sequence[str],
    where: Mapping[str, str] | None = None,
) -> QueryResult:
    if not texts:
        return {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
            "embeddings": None,
            "uris": None,
            "data": None,
            "included": ["documents", "metadatas", "distances"],
        }

    _, collection = _get_client_and_collection(collection_name)

    if where is None:
        return collection.query(query_texts=list(texts))

    return collection.query(query_texts=list(texts), where=build_where(where))
