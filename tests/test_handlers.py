from __future__ import annotations

import io
import unittest
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import TypeVar
from unittest.mock import MagicMock, patch

from chromy.handlers.count_collection import handle_count_collection
from chromy.handlers.create_collection import handle_create_collection
from chromy.handlers.delete_collection import (
    handle_delete_collection,
    handle_delete_records,
)
from chromy.handlers.import_data import handle_import
from chromy.handlers.list_collections import handle_list_collections
from chromy.handlers.query import handle_query

CommandT = TypeVar("CommandT")


class HandlerTests(unittest.TestCase):
    @staticmethod
    def _fixture_path(path: str) -> str:
        return str(Path(path).resolve())

    def test_list_collections_prints_empty_message(self) -> None:
        with patch(
            "chromy.handlers.list_collections.list_collections", return_value=[]
        ):
            exit_code, output = _capture_output(
                handle_list_collections,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output, "No collections found.\n")

    def test_list_collections_prints_collection_names(self) -> None:
        with patch(
            "chromy.handlers.list_collections.list_collections",
            return_value=["notes", "plays"],
        ):
            exit_code, output = _capture_output(
                handle_list_collections,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output, "· notes\n· plays\n")

    def test_create_collection_uses_typed_input(self) -> None:
        with patch(
            "chromy.handlers.create_collection.create_collection",
            return_value="notes",
        ) as create_collection:
            exit_code, output = _capture_output(
                handle_create_collection,
                "notes",
            )

        create_collection.assert_called_once_with("notes")
        self.assertEqual(exit_code, 0)
        self.assertEqual(output, "Created: collection 'notes'.\n")

    def test_delete_collection_uses_typed_input(self) -> None:
        with patch("chromy.handlers.delete_collection.delete_collection") as delete:
            exit_code, output = _capture_output(
                handle_delete_collection,
                "notes",
            )

        delete.assert_called_once_with("notes")
        self.assertEqual(exit_code, 0)
        self.assertEqual(output, "Deleted collection 'notes'.\n")

    def test_count_collection_uses_typed_input(self) -> None:
        with patch(
            "chromy.handlers.count_collection.count_collection",
            return_value=7,
        ) as count:
            exit_code, output = _capture_output(
                handle_count_collection,
                "notes",
            )

        count.assert_called_once_with("notes")
        self.assertEqual(exit_code, 0)
        self.assertEqual(output, "The 'notes' collection contains 7 records.\n")

    def test_import_data_uses_typed_input(self) -> None:
        with patch(
            "chromy.handlers.import_data.ingest_file",
            return_value=3,
        ) as ingest_file:
            exit_code, output = _capture_output(
                handle_import,
                "notes",
                ["romeo_and_juliet.txt"],
            )

        ingest_file.assert_called_once_with(
            "notes",
            self._fixture_path("romeo_and_juliet.txt"),
            None,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            output,
            "Added 3 records from 'romeo_and_juliet.txt' to collection 'notes'.\n"
            "Imported 1 file(s) successfully; 0 failed.\n",
        )

    def test_import_data_passes_metadata_to_every_file(self) -> None:
        with patch(
            "chromy.handlers.import_data.ingest_file",
            return_value=3,
        ) as ingest_file:
            exit_code, output = _capture_output(
                handle_import,
                "notes",
                ["romeo_and_juliet.txt"],
                "ticket=PROJ-123,file_name=/files/PROJ-123.md",
            )

        ingest_file.assert_called_once_with(
            "notes",
            self._fixture_path("romeo_and_juliet.txt"),
            {"ticket": "PROJ-123", "file_name": "/files/PROJ-123.md"},
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            output,
            "Added 3 records from 'romeo_and_juliet.txt' to collection 'notes'.\n"
            "Imported 1 file(s) successfully; 0 failed.\n",
        )

    def test_import_data_parse_error_aborts_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid --metadata value"):
            _capture_output(
                handle_import,
                "notes",
                ["romeo_and_juliet.txt"],
                "ticket",
            )

    def test_import_data_continues_after_missing_file(self) -> None:
        with patch(
            "chromy.handlers.import_data.ingest_file",
            return_value=3,
        ) as ingest_file:
            exit_code, output = _capture_output(
                handle_import,
                "notes",
                ["missing.txt", "romeo_and_juliet.txt"],
            )

        ingest_file.assert_called_once_with(
            "notes",
            self._fixture_path("romeo_and_juliet.txt"),
            None,
        )
        self.assertEqual(exit_code, 1)
        self.assertEqual(
            output,
            "Error: The file 'missing.txt' was not found.\n"
            "Added 3 records from 'romeo_and_juliet.txt' to collection 'notes'.\n"
            "Imported 1 file(s) successfully; 1 failed.\n",
        )

    def test_import_data_rejects_non_text_files(self) -> None:
        with patch(
            "chromy.handlers.import_data.is_probably_text_file",
            return_value=False,
        ):
            exit_code, output = _capture_output(
                handle_import,
                "notes",
                ["romeo_and_juliet.txt"],
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            output,
            "Error: The file 'romeo_and_juliet.txt' is not a text file.\n"
            "Imported 0 file(s) successfully; 1 failed.\n",
        )

    def test_import_data_deduplicates_files(self) -> None:
        with patch(
            "chromy.handlers.import_data.ingest_file",
            return_value=3,
        ) as ingest_file:
            exit_code, output = _capture_output(
                handle_import,
                "notes",
                ["README.md", "./README.md"],
            )

        ingest_file.assert_called_once_with(
            "notes",
            self._fixture_path("README.md"),
            None,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            output,
            "Added 3 records from 'README.md' to collection 'notes'.\n"
            "Imported 1 file(s) successfully; 0 failed.\n",
        )

    def test_import_data_suppresses_per_file_output_with_progress(self) -> None:
        progress = MagicMock()
        progress.__enter__.return_value = progress
        progress.__exit__.return_value = None
        progress.console.print = print
        progress.add_task.return_value = 1

        with (
            patch("chromy.handlers.import_data.ingest_file", side_effect=[3, 2]),
            patch(
                "chromy.handlers.import_data._should_show_progress",
                return_value=True,
            ),
            patch("chromy.handlers.import_data.Progress", return_value=progress),
        ):
            exit_code, output = _capture_output(
                handle_import,
                "notes",
                ["romeo_and_juliet.txt", "README.md"],
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output, "Imported 2 file(s) successfully; 0 failed.\n")

    def test_import_data_truncates_long_file_names_in_progress(self) -> None:
        progress = MagicMock()
        progress.__enter__.return_value = progress
        progress.__exit__.return_value = None
        progress.console.print = print
        progress.add_task.return_value = 1

        with (
            patch(
                "chromy.handlers.import_data._get_absolute_path",
                side_effect=[
                    "/tmp/this_is_a_very_long_file_name.txt",
                    self._fixture_path("README.md"),
                    "/tmp/this_is_a_very_long_file_name.txt",
                    self._fixture_path("README.md"),
                ],
            ),
            patch("chromy.handlers.import_data._import_one", return_value=3),
            patch(
                "chromy.handlers.import_data._should_show_progress",
                return_value=True,
            ),
            patch("chromy.handlers.import_data.Progress", return_value=progress),
        ):
            handle_import(
                "notes",
                ["this_is_a_very_long_file_name.txt", "README.md"],
            )

        progress.update.assert_any_call(
            1,
            description="Importing [bold]this_is_a_very_lo...[/]...",
        )

    def test_query_uses_typed_input(self) -> None:
        query_result = {"ids": [["1"]], "documents": [["hello"]]}
        with (
            patch("chromy.handlers.query.run_query", return_value=query_result) as run,
            patch(
                "chromy.handlers.query.format_query_result",
                return_value=["Query results:", "1"],
            ) as format_result,
        ):
            exit_code, output = _capture_output(
                handle_query,
                "notes",
                "hello",
            )

        run.assert_called_once_with("notes", "hello", None)
        format_result.assert_called_once_with(query_result)
        self.assertEqual(exit_code, 0)
        self.assertEqual(output, "Query results:\n1\n")

    def test_query_passes_where_filter(self) -> None:
        query_result = {"ids": [["1"]], "documents": [["hello"]]}
        with (
            patch("chromy.handlers.query.run_query", return_value=query_result) as run,
            patch(
                "chromy.handlers.query.format_query_result",
                return_value=["Query results:", "1"],
            ) as format_result,
        ):
            exit_code, output = _capture_output(
                handle_query,
                "notes",
                "hello",
                "ticket=PROJ-123,content_type=comment",
            )

        run.assert_called_once_with(
            "notes",
            "hello",
            {"ticket": "PROJ-123", "content_type": "comment"},
        )
        format_result.assert_called_once_with(query_result)
        self.assertEqual(exit_code, 0)
        self.assertEqual(output, "Query results:\n1\n")

    def test_query_parse_error_propagates(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid --where value"):
            _capture_output(
                handle_query,
                "notes",
                "hello",
                "ticket",
            )

    def test_delete_records_parses_where_filter(self) -> None:
        with patch(
            "chromy.handlers.delete_collection.delete_data",
            return_value=2,
        ) as delete_data:
            exit_code, output = _capture_output(
                handle_delete_records,
                "notes",
                " file_name = play.txt ",
            )

        delete_data.assert_called_once_with("notes", {"file_name": "play.txt"})
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            output,
            "Deleted 2 record(s) from collection 'notes' where file_name=play.txt.\n",
        )

    def test_delete_records_supports_multiple_pairs(self) -> None:
        with patch(
            "chromy.handlers.delete_collection.delete_data",
            return_value=2,
        ) as delete_data:
            exit_code, output = _capture_output(
                handle_delete_records,
                "notes",
                "a=1,b=sub=2",
            )

        delete_data.assert_called_once_with(
            "notes",
            {"a": "1", "b": "sub=2"},
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            output,
            "Deleted 2 record(s) from collection 'notes' where a=1, b=sub=2.\n",
        )

    def test_delete_records_rejects_invalid_where_filter(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Invalid --where value. Expected comma-separated <key>=<value> pairs.",
        ):
            handle_delete_records("notes", "file_name")


def _capture_output(
    handler: Callable[..., int],
    *arguments: object,
) -> tuple[int, str]:
    output = io.StringIO()

    with redirect_stdout(output):
        exit_code = handler(*arguments)

    return exit_code, output.getvalue()


if __name__ == "__main__":
    unittest.main()
