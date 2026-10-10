from __future__ import annotations

import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from chromadb.errors import InternalError, NotFoundError
from click.testing import Result
from typer.testing import CliRunner

from chromy.cli import app
from chromy.errors import ChromaPathError, EmbeddingFunctionError


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        import chromy.chroma_functions as chroma_functions

        chroma_functions.set_chroma_folder(None)

    @staticmethod
    def _fixture_path(path: str) -> str:
        return str(Path(path).resolve())

    def test_list_empty_collections(self) -> None:
        with patch(
            "chromy.handlers.list_collections.list_collections",
            return_value=[],
        ):
            result = _invoke(["list-collections"])

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "No collections found.\n")

    def test_list_existing_collections(self) -> None:
        with patch(
            "chromy.handlers.list_collections.list_collections",
            return_value=[
                ("books", "default"),
                ("jira", "sentence-transformers:my-model"),
            ],
        ):
            result = _invoke(["list-collections"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("books", result.stdout)
        self.assertIn("jira", result.stdout)
        self.assertIn("default", result.stdout)
        self.assertIn("sentence-transformers:my-model", result.stdout)

    def test_create_collection(self) -> None:
        with patch(
            "chromy.handlers.create_collection.create_collection",
            return_value="notes",
        ) as create_collection:
            result = _invoke(["create-collection", "notes"])

        create_collection.assert_called_once_with(
            "notes",
            model="default",
            max_tokens=None,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            result.stdout,
            "Created: collection 'notes' with embedding model 'default'.\n",
        )

    def test_create_collection_with_model_and_max_tokens(self) -> None:
        with patch(
            "chromy.handlers.create_collection.create_collection",
            return_value="notes",
        ) as create_collection:
            result = _invoke(
                [
                    "create-collection",
                    "notes",
                    "--model",
                    "sentence-transformers:all-MiniLM-L6-v2",
                    "--max-tokens",
                    "512",
                ],
            )

        create_collection.assert_called_once_with(
            "notes",
            model="sentence-transformers:all-MiniLM-L6-v2",
            max_tokens=512,
        )
        self.assertEqual(result.exit_code, 0)

    def test_create_collection_surfaces_embedding_function_errors(self) -> None:
        with patch(
            "chromy.handlers.create_collection.create_collection",
            side_effect=EmbeddingFunctionError("bad model"),
        ) as create_collection:
            result = _invoke(["create-collection", "notes", "--model", "bogus"])

        create_collection.assert_called_once_with(
            "notes",
            model="bogus",
            max_tokens=None,
        )
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.stdout, "Error: bad model\n")

    def test_create_collection_with_same_name(self) -> None:
        with patch(
            "chromy.handlers.create_collection.create_collection",
            side_effect=InternalError(),
        ) as create_collection:
            result = _invoke(["create-collection", "notes"])

        create_collection.assert_called_once_with(
            "notes",
            model="default",
            max_tokens=None,
        )
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.stdout, "Error: Collection 'notes' already exists.\n")

    def test_delete_collection(self) -> None:
        with patch(
            "chromy.handlers.delete_collection.delete_collection",
        ) as delete_collection:
            result = _invoke(["delete-collection", "notes"])

        delete_collection.assert_called_once_with("notes")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "Deleted collection 'notes'.\n")

    def test_delete_non_existent_collection(self) -> None:
        with patch(
            "chromy.handlers.delete_collection.delete_collection",
            side_effect=NotFoundError(),
        ) as delete_collection:
            result = _invoke(["delete-collection", "notes"])

        delete_collection.assert_called_once_with("notes")
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.stdout, "Error: Collection 'notes' does not exist.\n")

    def test_count(self) -> None:
        with patch(
            "chromy.handlers.count_collection.count_collection",
            return_value=7,
        ) as count_collection:
            result = _invoke(["count", "notes"])

        count_collection.assert_called_once_with("notes")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            result.stdout,
            "The 'notes' collection contains 7 records.\n",
        )

    def test_command_aliases(self) -> None:
        with self.subTest(alias="lc"):
            with patch(
                "chromy.handlers.list_collections.list_collections",
                return_value=[],
            ) as mocked:
                result = _invoke(["lc"])

            mocked.assert_called_once_with()
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.stdout, "No collections found.\n")

        with self.subTest(alias="cc"):
            with patch(
                "chromy.handlers.create_collection.create_collection",
                return_value="notes",
            ) as mocked:
                result = _invoke(["cc", "notes"])

            mocked.assert_called_once_with("notes", model="default", max_tokens=None)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(
                result.stdout,
                "Created: collection 'notes' with embedding model 'default'.\n",
            )

        with self.subTest(alias="dc"):
            with patch(
                "chromy.handlers.delete_collection.delete_collection",
            ) as mocked:
                result = _invoke(["dc", "notes"])

            mocked.assert_called_once_with("notes")
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.stdout, "Deleted collection 'notes'.\n")

        with self.subTest(alias="c"):
            with patch(
                "chromy.handlers.count_collection.count_collection",
                return_value=7,
            ) as mocked:
                result = _invoke(["c", "notes"])

            mocked.assert_called_once_with("notes")
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(
                result.stdout,
                "The 'notes' collection contains 7 records.\n",
            )

        with self.subTest(alias="i"):
            with (
                patch(
                    "chromy.handlers.import_data.get_collection_embedding_context",
                    return_value=("EMBEDDER", 204),
                ),
                patch(
                    "chromy.handlers.import_data.ingest_file",
                    return_value=3,
                ) as mocked,
            ):
                result = _invoke(["i", "notes", "romeo_and_juliet.txt"])

            mocked.assert_called_once_with(
                "notes",
                self._fixture_path("romeo_and_juliet.txt"),
                None,
                embedding_function="EMBEDDER",
                chunk_size=204,
            )
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(
                result.stdout,
                "Added 3 records from 'romeo_and_juliet.txt' to collection 'notes'.\n"
                "Imported 1 file(s) successfully; 0 failed.\n",
            )

        with self.subTest(alias="q"):
            query_result = {"ids": [["1"]], "documents": [["hello"]]}
            with (
                patch(
                    "chromy.handlers.query.run_query",
                    return_value=query_result,
                ) as mocked,
                patch(
                    "chromy.handlers.query.format_query_result",
                    return_value=["1"],
                ),
            ):
                result = _invoke(["q", "notes", "Where is Romeo?"])

            mocked.assert_called_once_with("notes", "Where is Romeo?", None)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.stdout, "1\n")

        with self.subTest(alias="del"):
            with patch(
                "chromy.handlers.delete_collection.delete_data",
                return_value=2,
            ) as mocked:
                result = _invoke(
                    ["del", "notes", "--where", "file_name=play.txt"],
                )

            mocked.assert_called_once_with("notes", {"file_name": "play.txt"})
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(
                result.stdout,
                "Deleted 2 record(s) from collection 'notes' where "
                "file_name=play.txt.\n",
            )

    def test_import_data(self) -> None:
        with (
            patch(
                "chromy.handlers.import_data.get_collection_embedding_context",
                return_value=("EMBEDDER", 204),
            ),
            patch(
                "chromy.handlers.import_data.ingest_file",
                return_value=3,
            ) as ingest_file,
        ):
            result = _invoke(["import", "notes", "romeo_and_juliet.txt"])

        ingest_file.assert_called_once_with(
            "notes",
            self._fixture_path("romeo_and_juliet.txt"),
            None,
            embedding_function="EMBEDDER",
            chunk_size=204,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            result.stdout,
            "Added 3 records from 'romeo_and_juliet.txt' to collection 'notes'.\n"
            "Imported 1 file(s) successfully; 0 failed.\n",
        )

    def test_import_data_trailing_metadata_option(self) -> None:
        with (
            patch(
                "chromy.handlers.import_data.get_collection_embedding_context",
                return_value=("EMBEDDER", 204),
            ),
            patch(
                "chromy.handlers.import_data.ingest_file",
                return_value=3,
            ) as ingest_file,
        ):
            result = _invoke(
                [
                    "import",
                    "notes",
                    "romeo_and_juliet.txt",
                    "--metadata",
                    "ticket=PROJ-123,file_name=/files/PROJ-123.md",
                ],
            )

        ingest_file.assert_called_once_with(
            "notes",
            self._fixture_path("romeo_and_juliet.txt"),
            {"ticket": "PROJ-123", "file_name": "/files/PROJ-123.md"},
            embedding_function="EMBEDDER",
            chunk_size=204,
        )
        self.assertEqual(result.exit_code, 0)

    def test_import_data_metadata_option_on_alias(self) -> None:
        with (
            patch(
                "chromy.handlers.import_data.get_collection_embedding_context",
                return_value=("EMBEDDER", 204),
            ),
            patch(
                "chromy.handlers.import_data.ingest_file",
                return_value=3,
            ) as ingest_file,
        ):
            result = _invoke(
                [
                    "i",
                    "notes",
                    "romeo_and_juliet.txt",
                    "--metadata",
                    "ticket=PROJ-123",
                ],
            )

        ingest_file.assert_called_once_with(
            "notes",
            self._fixture_path("romeo_and_juliet.txt"),
            {"ticket": "PROJ-123"},
            embedding_function="EMBEDDER",
            chunk_size=204,
        )
        self.assertEqual(result.exit_code, 0)

    def test_import_data_invalid_metadata_keeps_user_facing_error(self) -> None:
        result = _invoke(
            ["import", "notes", "romeo_and_juliet.txt", "--metadata", "ticket"]
        )

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(
            result.stdout,
            "Error: Invalid --metadata value. Expected comma-separated "
            "<key>=<value> pairs.\n",
        )

    def test_import_data_accepts_multiple_files(self) -> None:
        with (
            patch(
                "chromy.handlers.import_data.get_collection_embedding_context",
                return_value=("EMBEDDER", 204),
            ),
            patch(
                "chromy.handlers.import_data.ingest_file",
                side_effect=[3, 2],
            ) as ingest_file,
        ):
            result = _invoke(
                ["import", "notes", "romeo_and_juliet.txt", "README.md"],
            )

        self.assertEqual(ingest_file.call_count, 2)
        ingest_file.assert_any_call(
            "notes",
            self._fixture_path("romeo_and_juliet.txt"),
            None,
            embedding_function="EMBEDDER",
            chunk_size=204,
        )
        ingest_file.assert_any_call(
            "notes",
            self._fixture_path("README.md"),
            None,
            embedding_function="EMBEDDER",
            chunk_size=204,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            result.stdout,
            "Added 3 records from 'romeo_and_juliet.txt' to collection 'notes'.\n"
            "Added 2 records from 'README.md' to collection 'notes'.\n"
            "Imported 2 file(s) successfully; 0 failed.\n",
        )

    def test_import_data_continues_after_missing_file(self) -> None:
        with (
            patch(
                "chromy.handlers.import_data.get_collection_embedding_context",
                return_value=("EMBEDDER", 204),
            ),
            patch(
                "chromy.handlers.import_data.ingest_file",
                return_value=3,
            ) as ingest_file,
        ):
            result = _invoke(
                ["import", "notes", "missing.txt", "romeo_and_juliet.txt"],
            )

        ingest_file.assert_called_once_with(
            "notes",
            self._fixture_path("romeo_and_juliet.txt"),
            None,
            embedding_function="EMBEDDER",
            chunk_size=204,
        )
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(
            result.stdout,
            "Error: The file 'missing.txt' was not found.\n"
            "Added 3 records from 'romeo_and_juliet.txt' to collection 'notes'.\n"
            "Imported 1 file(s) successfully; 1 failed.\n",
        )

    def test_import_data_rejects_non_text_files(self) -> None:
        with (
            patch(
                "chromy.handlers.import_data.get_collection_embedding_context",
                return_value=("EMBEDDER", 204),
            ),
            patch(
                "chromy.handlers.import_data.is_probably_text_file",
                return_value=False,
            ),
        ):
            result = _invoke(["import", "notes", "romeo_and_juliet.txt"])

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(
            result.stdout,
            "Error: The file 'romeo_and_juliet.txt' is not a text file.\n"
            "Imported 0 file(s) successfully; 1 failed.\n",
        )

    def test_import_data_treats_literal_glob_as_missing_file(self) -> None:
        with patch(
            "chromy.handlers.import_data.get_collection_embedding_context",
            return_value=("EMBEDDER", 204),
        ):
            result = _invoke(["import", "notes", "*.md"])

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(
            result.stdout,
            "Error: The file '*.md' was not found.\n"
            "Imported 0 file(s) successfully; 1 failed.\n",
        )

    def test_import_data_deduplicates_paths_within_single_invocation(self) -> None:
        with (
            patch(
                "chromy.handlers.import_data.get_collection_embedding_context",
                return_value=("EMBEDDER", 204),
            ),
            patch(
                "chromy.handlers.import_data.ingest_file",
                return_value=3,
            ) as ingest_file,
        ):
            result = _invoke(
                ["import", "notes", "README.md", "./README.md"],
            )

        ingest_file.assert_called_once_with(
            "notes",
            self._fixture_path("README.md"),
            None,
            embedding_function="EMBEDDER",
            chunk_size=204,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            result.stdout,
            "Added 3 records from 'README.md' to collection 'notes'.\n"
            "Imported 1 file(s) successfully; 0 failed.\n",
        )

    def test_query(self) -> None:
        query_result = {"ids": [["1"]], "documents": [["hello"]]}

        with (
            patch("chromy.handlers.query.run_query", return_value=query_result) as run,
            patch(
                "chromy.handlers.query.format_query_result",
                return_value=["Query results:", "1"],
            ) as format_result,
        ):
            result = _invoke(["query", "notes", "Where is Romeo?"])

        run.assert_called_once_with("notes", "Where is Romeo?", None)
        format_result.assert_called_once_with(query_result)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "Query results:\n1\n")

    def test_query_where_option(self) -> None:
        query_result = {"ids": [["1"]], "documents": [["hello"]]}

        with (
            patch("chromy.handlers.query.run_query", return_value=query_result) as run,
            patch(
                "chromy.handlers.query.format_query_result",
                return_value=["Query results:", "1"],
            ),
        ):
            result = _invoke(
                ["query", "notes", "Where is Romeo?", "--where", "ticket=PROJ-123"],
            )

        run.assert_called_once_with(
            "notes",
            "Where is Romeo?",
            {"ticket": "PROJ-123"},
        )
        self.assertEqual(result.exit_code, 0)

    def test_query_where_option_on_alias(self) -> None:
        query_result = {"ids": [["1"]], "documents": [["hello"]]}

        with (
            patch("chromy.handlers.query.run_query", return_value=query_result) as run,
            patch(
                "chromy.handlers.query.format_query_result",
                return_value=["Query results:", "1"],
            ),
        ):
            result = _invoke(
                ["q", "notes", "Where is Romeo?", "--where", "ticket=PROJ-123"],
            )

        run.assert_called_once_with(
            "notes",
            "Where is Romeo?",
            {"ticket": "PROJ-123"},
        )
        self.assertEqual(result.exit_code, 0)

    def test_query_invalid_where_keeps_user_facing_error(self) -> None:
        result = _invoke(["query", "notes", "hello", "--where", "ticket"])

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(
            result.stdout,
            "Error: Invalid --where value. Expected comma-separated "
            "<key>=<value> pairs.\n",
        )

    def test_delete_records(self) -> None:
        with patch(
            "chromy.handlers.delete_collection.delete_data",
            return_value=2,
        ) as delete_data:
            result = _invoke(
                ["delete", "notes", "--where", " file_name = play.txt "],
            )

        delete_data.assert_called_once_with("notes", {"file_name": "play.txt"})
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            result.stdout,
            "Deleted 2 record(s) from collection 'notes' where file_name=play.txt.\n",
        )

    def test_delete_records_multi_key(self) -> None:
        with patch(
            "chromy.handlers.delete_collection.delete_data",
            return_value=2,
        ) as delete_data:
            result = _invoke(
                ["delete", "notes", "--where", "a=1,b=2"],
            )

        delete_data.assert_called_once_with(
            "notes",
            {"a": "1", "b": "2"},
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            result.stdout,
            "Deleted 2 record(s) from collection 'notes' where a=1, b=2.\n",
        )

    def test_invalid_delete_filter_keeps_user_facing_error(self) -> None:
        result = _invoke(["delete", "notes", "--where", "file_name"])

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(
            result.stdout,
            "Error: Invalid --where value. Expected comma-separated "
            "<key>=<value> pairs.\n",
        )

    def test_delete_requires_where_option(self) -> None:
        result = _invoke(["delete", "notes"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Missing option", result.output)

    def test_cli_without_arguments_prints_error_and_help(self) -> None:
        result = _invoke([])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("Error: Missing command.", result.stdout)
        self.assertIn("Usage:", result.stdout)
        self.assertIn("Commands", result.stdout)

    def test_cli_help_documents_chroma_folder_env_var(self) -> None:
        result = _invoke(["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("CHROMA_FOLDER", result.stdout)
        self.assertIn("parent directory", result.stdout)
        self.assertIn("<CHROMA_FOLDER>/chroma", result.stdout)

    def test_cli_help_documents_chroma_folder_option(self) -> None:
        result = _invoke(["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("--chroma-folder", result.stdout)

    def test_chroma_folder_option_wins_over_env_var(self) -> None:
        import chromy.chroma_functions as chroma_functions

        with (
            patch(
                "chromy.chroma_functions._ensure_persistence_path_is_usable"
            ) as ensure_mock,
            patch("chromy.chroma_functions.chromadb.PersistentClient") as client_mock,
            patch.dict(
                "os.environ", {chroma_functions.CHROMA_FOLDER_ENV_VAR: "/env/parent"}
            ),
        ):
            client_mock_instance = client_mock.return_value
            client_mock_instance.list_collections.return_value = []
            result = _invoke(["--chroma-folder", "/flag/parent", "list-collections"])

        self.assertEqual(result.exit_code, 0)
        expected_path = Path("/flag/parent") / "chroma"
        ensure_mock.assert_called_once_with(expected_path, "/flag/parent")
        client_mock.assert_called_once_with(path=str(expected_path))

    def test_cli_help_documents_command_aliases(self) -> None:
        result = _invoke(["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Command aliases", result.stdout)
        self.assertIn("list-collections: lc", result.stdout)
        self.assertIn("create-collection: cc", result.stdout)
        self.assertIn("delete-collection: dc", result.stdout)
        self.assertIn("count: c", result.stdout)
        self.assertIn("import: i", result.stdout)
        self.assertIn("query: q", result.stdout)
        self.assertIn("delete: del", result.stdout)

    def test_cli_dash_h_matches_dash_dash_help(self) -> None:
        help_result = _invoke(["--help"])
        short_result = _invoke(["-h"])

        self.assertEqual(short_result.exit_code, 0)
        self.assertEqual(short_result.stdout, help_result.stdout)

    def test_command_dash_h_shows_command_help(self) -> None:
        result = _invoke(["count", "-h"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Usage:", result.stdout)

    def test_cli_surfaces_chroma_path_errors(self) -> None:
        with patch(
            "chromy.handlers.list_collections.list_collections",
            side_effect=ChromaPathError("configured path is not writable"),
        ):
            result = _invoke(["list-collections"])

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(
            result.stdout,
            "Error: configured path is not writable\n",
        )


def _invoke(arguments: Sequence[str]) -> Result:
    return CliRunner().invoke(app, list(arguments))


if __name__ == "__main__":
    unittest.main()
