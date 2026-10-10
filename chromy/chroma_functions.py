from __future__ import annotations

import os
from collections.abc import Sequence
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

_chroma_folder_override: str | None = None


def set_chroma_folder(value: str | None) -> None:
    """Set the CLI-provided Chroma parent folder used by get_client()."""
    global _chroma_folder_override
    _chroma_folder_override = value


def _configured_parent() -> str | None:
    """Return the effective Chroma parent folder from CLI or env."""
    if _chroma_folder_override is not None:
        return _chroma_folder_override
    return os.getenv(CHROMA_FOLDER_ENV_VAR)


def _resolve_persistence_path() -> Path | None:
    configured_parent = _configured_parent()

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
    configured_parent = _configured_parent()
    persistence_path = _resolve_persistence_path()

    if persistence_path is None:
        return chromadb.PersistentClient()

    assert configured_parent is not None  # path exists only when parent set
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


def delete_data(collection_name: str, where: dict[str, str]) -> int:
    _, collection = _get_client_and_collection(collection_name)
    result = collection.delete(where=cast(Where, where))

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
) -> None:
    if not data:
        return

    _, collection = _get_client_and_collection(collection_name)

    embeddings: list[Sequence[float]] = [record["embedding"] for record in data]

    collection.add(
        ids=[str(uuid4()) for _ in data],
        metadatas=[{"file_name": file_name} for _ in data],
        documents=[record["text"] for record in data],
        embeddings=embeddings,
    )


def query_data(collection_name: str, texts: Sequence[str]) -> QueryResult:
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

    return collection.query(query_texts=list(texts))
