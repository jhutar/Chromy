from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from chromy.chunking import chunk_file, chunk_text


class ChunkingTests(unittest.TestCase):
    def test_chunk_text_uses_given_chunk_size(self) -> None:
        with patch("chromy.chunking.service.semchunk") as semchunk_mock:
            chunk_text("hello world", chunk_size=7)

        semchunk_mock.chunkerify.assert_called_once_with("gpt-4", 7)
        semchunk_mock.chunkerify.return_value.assert_called_once_with("hello world")

    def test_chunk_file_reads_contents_and_chunks(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("some contents")
            file_path = handle.name

        try:
            with patch(
                "chromy.chunking.service.chunk_text", return_value=["chunked"]
            ) as chunk_text_mock:
                result = chunk_file(file_path, chunk_size=7)

            chunk_text_mock.assert_called_once_with("some contents", 7)
            self.assertEqual(result, ["chunked"])
        finally:
            os.unlink(file_path)


if __name__ == "__main__":
    unittest.main()
