from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from chromy.chroma_functions import (
    add_data,
    build_where,
    create_collection,
    delete_data,
    get_client,
    get_collection_embedding_context,
    list_collections,
    query_data,
)
from chromy.embedding import EmbeddingRecord
from chromy.errors import ChromaPathError, EmbeddingFunctionError


class BuildWhereTests(unittest.TestCase):
    def test_single_pair_returned_flat(self) -> None:
        self.assertEqual(
            build_where({"ticket": "PROJ-123"}),
            {"ticket": "PROJ-123"},
        )

    def test_multiple_pairs_wrapped_in_and(self) -> None:
        self.assertEqual(
            build_where({"a": "1", "b": "2"}),
            {"$and": [{"a": "1"}, {"b": "2"}]},
        )


class QueryDataTests(unittest.TestCase):
    def test_query_without_where_omits_where_argument(self) -> None:
        collection = MagicMock()

        with patch(
            "chromy.chroma_functions._get_client_and_collection",
            return_value=(None, collection),
        ):
            query_data("notes", ["hello"])

        collection.query.assert_called_once_with(query_texts=["hello"])

    def test_query_single_pair_passes_flat_dict(self) -> None:
        collection = MagicMock()

        with patch(
            "chromy.chroma_functions._get_client_and_collection",
            return_value=(None, collection),
        ):
            query_data("notes", ["hello"], {"ticket": "PROJ-123"})

        collection.query.assert_called_once_with(
            query_texts=["hello"],
            where={"ticket": "PROJ-123"},
        )

    def test_query_multiple_pairs_wraps_in_and(self) -> None:
        collection = MagicMock()

        with patch(
            "chromy.chroma_functions._get_client_and_collection",
            return_value=(None, collection),
        ):
            query_data("notes", ["hello"], {"a": "1", "b": "2"})

        collection.query.assert_called_once_with(
            query_texts=["hello"],
            where={"$and": [{"a": "1"}, {"b": "2"}]},
        )

    def test_query_empty_texts_returns_empty_result(self) -> None:
        result = query_data("notes", [])

        self.assertEqual(result["ids"], [])


class DeleteDataTests(unittest.TestCase):
    def test_delete_single_pair_uses_flat_where(self) -> None:
        collection = MagicMock()
        collection.delete.return_value = {"deleted": 3}

        with patch(
            "chromy.chroma_functions._get_client_and_collection",
            return_value=(None, collection),
        ):
            deleted = delete_data("notes", {"file_name": "/tmp/x"})

        collection.delete.assert_called_once_with(where={"file_name": "/tmp/x"})
        self.assertEqual(deleted, 3)

    def test_delete_multiple_pairs_wraps_in_and(self) -> None:
        collection = MagicMock()
        collection.delete.return_value = {"deleted": 2}

        with patch(
            "chromy.chroma_functions._get_client_and_collection",
            return_value=(None, collection),
        ):
            deleted = delete_data("notes", {"a": "1", "b": "2"})

        collection.delete.assert_called_once_with(
            where={"$and": [{"a": "1"}, {"b": "2"}]},
        )
        self.assertEqual(deleted, 2)


class AddDataTests(unittest.TestCase):
    def _records(self) -> list[EmbeddingRecord]:
        return [
            {"text": "chunk 1", "embedding": [0.1, 0.2]},
            {"text": "chunk 2", "embedding": [0.3, 0.4]},
        ]

    def test_add_data_merges_extra_metadata(self) -> None:
        collection = MagicMock()

        with patch(
            "chromy.chroma_functions._get_client_and_collection",
            return_value=(None, collection),
        ):
            add_data("notes", self._records(), "/tmp/x", {"ticket": "PROJ-123"})

        metadatas = collection.add.call_args.kwargs["metadatas"]
        self.assertEqual(
            metadatas,
            [
                {"ticket": "PROJ-123", "file_name": "/tmp/x"},
                {"ticket": "PROJ-123", "file_name": "/tmp/x"},
            ],
        )

    def test_add_data_reserved_file_name_always_wins(self) -> None:
        collection = MagicMock()

        with patch(
            "chromy.chroma_functions._get_client_and_collection",
            return_value=(None, collection),
        ):
            add_data(
                "notes",
                self._records(),
                "/tmp/x",
                {"file_name": "evil", "ticket": "1"},
            )

        metadatas = collection.add.call_args.kwargs["metadatas"]
        self.assertEqual(metadatas[0]["file_name"], "/tmp/x")
        self.assertEqual(metadatas[0]["ticket"], "1")

    def test_add_data_empty_data_is_noop(self) -> None:
        collection = MagicMock()

        with patch(
            "chromy.chroma_functions._get_client_and_collection",
            return_value=(None, collection),
        ):
            add_data("notes", [], "/tmp/x")

        collection.add.assert_not_called()


class ChromaFunctionsTests(unittest.TestCase):
    def test_get_client_uses_default_when_env_is_unset(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("chromy.chroma_functions.chromadb.PersistentClient") as persistent,
        ):
            get_client()

        persistent.assert_called_once_with()

    def test_get_client_uses_chroma_folder_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            configured_parent = Path(temp_dir) / "data"

            with (
                patch.dict(os.environ, {"CHROMA_FOLDER": str(configured_parent)}),
                patch(
                    "chromy.chroma_functions.chromadb.PersistentClient"
                ) as persistent,
            ):
                get_client()

            expected_path = configured_parent.resolve() / "chroma"
            persistent.assert_called_once_with(path=str(expected_path))
            self.assertTrue(expected_path.is_dir())

    def test_get_client_resolves_relative_chroma_folder_from_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)
            previous_cwd = Path.cwd()

            try:
                os.chdir(working_dir)
                with (
                    patch.dict(os.environ, {"CHROMA_FOLDER": "relative-parent"}),
                    patch(
                        "chromy.chroma_functions.chromadb.PersistentClient"
                    ) as persistent,
                ):
                    get_client()
            finally:
                os.chdir(previous_cwd)

            expected_path = (working_dir / "relative-parent").resolve() / "chroma"
            persistent.assert_called_once_with(path=str(expected_path))

    def test_get_client_fails_when_configured_path_is_not_usable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_parent = Path(temp_dir) / "not-a-directory"
            invalid_parent.write_text("x", encoding="utf-8")

            with (
                patch.dict(os.environ, {"CHROMA_FOLDER": str(invalid_parent)}),
                self.assertRaisesRegex(
                    ChromaPathError,
                    "Could not create or access Chroma directory",
                ),
            ):
                get_client()

    def test_get_client_wraps_client_initialization_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            configured_parent = Path(temp_dir)

            with (
                patch.dict(os.environ, {"CHROMA_FOLDER": str(configured_parent)}),
                patch(
                    "chromy.chroma_functions.chromadb.PersistentClient",
                    side_effect=RuntimeError("boom"),
                ),
                self.assertRaisesRegex(
                    ChromaPathError,
                    "Could not initialize Chroma client",
                ),
            ):
                get_client()


class _FakeSentenceTransformer:
    model_name = "my-model"

    @staticmethod
    def name() -> str:
        return "sentence_transformer"


def _embedding_function(max_tokens: int | None = 256) -> MagicMock:
    embedding_function = MagicMock(name="embedding_function")
    if max_tokens is None:
        return MagicMock(spec=[])
    embedding_function.max_tokens.return_value = max_tokens
    return embedding_function


class ChromaModelTests(unittest.TestCase):
    def test_create_collection_passes_built_embedding_function(self) -> None:
        embedding_function = _embedding_function()
        client = MagicMock()
        client.create_collection.return_value.name = "books"

        with (
            patch("chromy.chroma_functions.get_client", return_value=client),
            patch(
                "chromy.chroma_functions.build_embedding_function",
                return_value=embedding_function,
            ) as builder,
        ):
            name = create_collection(
                "books",
                model="sentence-transformers:my-model",
                max_tokens=512,
            )

        builder.assert_called_once_with("sentence-transformers:my-model")
        client.create_collection.assert_called_once_with(
            name="books",
            embedding_function=embedding_function,
            metadata={"chromy_max_tokens": 512},
        )
        self.assertEqual(name, "books")

    def test_create_collection_with_default_model_passes_no_metadata(self) -> None:
        embedding_function = _embedding_function()
        client = MagicMock()
        client.create_collection.return_value.name = "books"

        with (
            patch("chromy.chroma_functions.get_client", return_value=client),
            patch(
                "chromy.chroma_functions.build_embedding_function",
                return_value=embedding_function,
            ),
        ):
            create_collection("books")

        client.create_collection.assert_called_once_with(
            name="books",
            embedding_function=embedding_function,
            metadata=None,
        )

    def test_create_collection_requires_max_tokens_when_model_has_no_limit(
        self,
    ) -> None:
        embedding_function = _embedding_function(max_tokens=None)
        client = MagicMock()

        with (
            patch("chromy.chroma_functions.get_client", return_value=client),
            patch(
                "chromy.chroma_functions.build_embedding_function",
                return_value=embedding_function,
            ),
            self.assertRaisesRegex(
                EmbeddingFunctionError,
                "does not report a maximum input length",
            ),
        ):
            create_collection("books", model="sentence-transformers:my-model")

        client.create_collection.assert_not_called()

    def test_get_collection_embedding_context_uses_model_limit(self) -> None:
        embedding_function = _embedding_function(max_tokens=256)
        collection = MagicMock()
        collection.configuration = {"embedding_function": embedding_function}
        collection.metadata = {}

        with patch(
            "chromy.chroma_functions._get_client_and_collection",
            return_value=(None, collection),
        ):
            resolved_ef, chunk_size = get_collection_embedding_context("notes")

        self.assertIs(resolved_ef, embedding_function)
        self.assertEqual(chunk_size, 204)

    def test_get_collection_embedding_context_uses_stored_max_tokens(
        self,
    ) -> None:
        embedding_function = _embedding_function(max_tokens=None)
        collection = MagicMock()
        collection.configuration = {"embedding_function": embedding_function}
        collection.metadata = {"chromy_max_tokens": 512}

        with patch(
            "chromy.chroma_functions._get_client_and_collection",
            return_value=(None, collection),
        ):
            _, chunk_size = get_collection_embedding_context("notes")

        self.assertEqual(chunk_size, 409)

    def test_get_collection_embedding_context_raises_without_any_limit(
        self,
    ) -> None:
        embedding_function = _embedding_function(max_tokens=None)
        collection = MagicMock()
        collection.configuration = {"embedding_function": embedding_function}
        collection.metadata = {}

        with (
            patch(
                "chromy.chroma_functions._get_client_and_collection",
                return_value=(None, collection),
            ),
            self.assertRaisesRegex(
                EmbeddingFunctionError,
                "Collection 'notes': .* --max-tokens",
            ),
        ):
            get_collection_embedding_context("notes")

    def test_list_collections_includes_embedding_model(self) -> None:
        default_collection = MagicMock()
        default_collection.name = "books"
        default_collection.configuration = {
            "embedding_function": DefaultEmbeddingFunction(),
        }

        st_collection = MagicMock()
        st_collection.name = "jira"
        st_collection.configuration = {
            "embedding_function": _FakeSentenceTransformer(),
        }

        client = MagicMock()
        client.list_collections.return_value = [default_collection, st_collection]

        with patch("chromy.chroma_functions.get_client", return_value=client):
            result = list_collections()

        self.assertEqual(
            result,
            [("books", "default"), ("jira", "sentence-transformers:my-model")],
        )


if __name__ == "__main__":
    unittest.main()
