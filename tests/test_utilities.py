from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, patch

from chromy.utilities import ingest_file, parse_key_value_pairs


class KeyValueParserTests(unittest.TestCase):
    def test_parses_multiple_pairs(self) -> None:
        result = parse_key_value_pairs(
            "ticket=PROJ-123,content_type=comment",
            "--metadata",
        )

        self.assertEqual(
            result,
            {"ticket": "PROJ-123", "content_type": "comment"},
        )

    def test_trims_keys_and_values(self) -> None:
        result = parse_key_value_pairs(
            " ticket = PROJ-123 , file = /tmp/x ",
            "--metadata",
        )

        self.assertEqual(result, {"ticket": "PROJ-123", "file": "/tmp/x"})

    def test_allows_equals_sign_inside_value(self) -> None:
        result = parse_key_value_pairs("file_name=/path/a=b.txt", "--metadata")

        self.assertEqual(result, {"file_name": "/path/a=b.txt"})

    def test_duplicate_key_last_wins(self) -> None:
        result = parse_key_value_pairs("ticket=a,ticket=b", "--metadata")

        self.assertEqual(result, {"ticket": "b"})

    def test_rejects_empty_option(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid --metadata value"):
            parse_key_value_pairs("", "--metadata")

    def test_rejects_whitespace_only_option(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid --metadata value"):
            parse_key_value_pairs("   ", "--metadata")

    def test_rejects_item_without_equals(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid --metadata value"):
            parse_key_value_pairs("ticket", "--metadata")

    def test_rejects_empty_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid --metadata value"):
            parse_key_value_pairs("=value", "--metadata")

    def test_rejects_empty_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid --metadata value"):
            parse_key_value_pairs("ticket=", "--metadata")

    def test_error_message_uses_option_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid --where value"):
            parse_key_value_pairs("ticket", "--where")


class UtilityTests(unittest.TestCase):
    def test_ingest_file_adds_new_file_without_deleting(self) -> None:
        chunks = ["chunk 1", "chunk 2"]
        embeddings = [
            {"text": "chunk 1", "embedding": [0.1, 0.2]},
            {"text": "chunk 2", "embedding": [0.3, 0.4]},
        ]

        with (
            patch("chromy.utilities.has_data_for_file", return_value=False) as has_data,
            patch("chromy.utilities.delete_data") as delete_data,
            patch("chromy.utilities.chunk_file", return_value=chunks) as chunk_file,
            patch("chromy.utilities.embed", return_value=embeddings) as embed,
            patch("chromy.utilities.add_data") as add_data,
        ):
            records_added = ingest_file("notes", "/tmp/play.txt")

        has_data.assert_called_once_with("notes", "/tmp/play.txt")
        delete_data.assert_not_called()
        chunk_file.assert_called_once_with("/tmp/play.txt")
        embed.assert_called_once_with(chunks)
        add_data.assert_called_once_with("notes", embeddings, "/tmp/play.txt", {})
        self.assertEqual(records_added, 2)

    def test_ingest_file_replaces_existing_file_records_before_adding(self) -> None:
        chunks = ["chunk 1"]
        embeddings = [{"text": "chunk 1", "embedding": [0.1, 0.2]}]
        manager = MagicMock()

        with (
            patch("chromy.utilities.has_data_for_file", return_value=True) as has_data,
            patch("chromy.utilities.delete_data") as delete_data,
            patch("chromy.utilities.chunk_file", return_value=chunks) as chunk_file,
            patch("chromy.utilities.embed", return_value=embeddings) as embed,
            patch("chromy.utilities.add_data") as add_data,
        ):
            manager.attach_mock(has_data, "has_data")
            manager.attach_mock(delete_data, "delete_data")
            manager.attach_mock(chunk_file, "chunk_file")
            manager.attach_mock(embed, "embed")
            manager.attach_mock(add_data, "add_data")

            records_added = ingest_file("notes", "/tmp/play.txt")

        self.assertEqual(
            manager.mock_calls,
            [
                call.has_data("notes", "/tmp/play.txt"),
                call.delete_data("notes", {"file_name": "/tmp/play.txt"}),
                call.chunk_file("/tmp/play.txt"),
                call.embed(chunks),
                call.add_data("notes", embeddings, "/tmp/play.txt", {}),
            ],
        )
        self.assertEqual(records_added, 1)

    def test_ingest_file_overrides_logical_file_name(self) -> None:
        chunks = ["chunk 1"]
        embeddings = [{"text": "chunk 1", "embedding": [0.1, 0.2]}]
        metadata = {"ticket": "PROJ-123", "file_name": "/files/PROJ-123.md"}
        manager = MagicMock()

        with (
            patch("chromy.utilities.has_data_for_file", return_value=True) as has_data,
            patch("chromy.utilities.delete_data") as delete_data,
            patch("chromy.utilities.chunk_file", return_value=chunks) as chunk_file,
            patch("chromy.utilities.embed", return_value=embeddings) as embed,
            patch("chromy.utilities.add_data") as add_data,
        ):
            manager.attach_mock(has_data, "has_data")
            manager.attach_mock(delete_data, "delete_data")
            manager.attach_mock(chunk_file, "chunk_file")
            manager.attach_mock(embed, "embed")
            manager.attach_mock(add_data, "add_data")

            records_added = ingest_file("notes", "/tmp/play.txt", metadata)

        self.assertEqual(
            manager.mock_calls,
            [
                call.has_data("notes", "/files/PROJ-123.md"),
                call.delete_data("notes", {"file_name": "/files/PROJ-123.md"}),
                call.chunk_file("/tmp/play.txt"),
                call.embed(chunks),
                call.add_data(
                    "notes",
                    embeddings,
                    "/files/PROJ-123.md",
                    metadata,
                ),
            ],
        )
        self.assertEqual(records_added, 1)


if __name__ == "__main__":
    unittest.main()
