from __future__ import annotations

import unittest

from chromadb import QueryResult

from chromy.output import format_query_result


def _render(result: QueryResult) -> str:
    lines = format_query_result(result)
    return "\n".join(str(line) for line in lines)


class FormatQueryResultTests(unittest.TestCase):
    def test_renders_extra_metadata_fields(self) -> None:
        result: QueryResult = {
            "ids": [["1"]],
            "documents": [["chunk text"]],
            "distances": [[0.5]],
            "metadatas": [
                [
                    {
                        "file_name": "/tmp/x",
                        "ticket": "PROJ-123",
                        "content_type": "comment",
                    }
                ]
            ],
            "embeddings": None,
            "uris": None,
            "data": None,
            "included": ["documents", "metadatas", "distances"],
        }

        rendered = _render(result)

        self.assertIn("file_name\t/tmp/x", rendered)
        self.assertIn("content_type\tcomment", rendered)
        self.assertIn("ticket\tPROJ-123", rendered)

    def test_extra_metadata_fields_sorted_after_file_name(self) -> None:
        result: QueryResult = {
            "ids": [["1"]],
            "documents": [["chunk text"]],
            "distances": [[0.5]],
            "metadatas": [[{"zeta": "z", "alpha": "a", "file_name": "/tmp/x"}]],
            "embeddings": None,
            "uris": None,
            "data": None,
            "included": ["documents", "metadatas", "distances"],
        }

        rendered = _render(result)
        file_name_index = rendered.index("file_name\t/tmp/x")
        alpha_index = rendered.index("alpha\ta")
        zeta_index = rendered.index("zeta\tz")

        self.assertLess(alpha_index, zeta_index)
        self.assertLess(file_name_index, alpha_index)

    def test_no_metadata_fields_when_metadatas_empty(self) -> None:
        result: QueryResult = {
            "ids": [["1"]],
            "documents": [["chunk text"]],
            "distances": [[0.5]],
            "metadatas": [[{}]],
            "embeddings": None,
            "uris": None,
            "data": None,
            "included": ["documents", "metadatas", "distances"],
        }

        rendered = _render(result)

        self.assertIn("chunk text", rendered)
        self.assertNotIn("file_name", rendered)

    def test_no_results_returns_notice(self) -> None:
        result: QueryResult = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "embeddings": None,
            "uris": None,
            "data": None,
            "included": ["documents", "metadatas", "distances"],
        }

        rendered = _render(result)

        self.assertEqual(rendered, "No results found.")


if __name__ == "__main__":
    unittest.main()
