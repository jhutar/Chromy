from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from chromy.embedding.registry import (
    build_embedding_function,
    model_max_tokens,
)
from chromy.errors import EmbeddingFunctionError


class RegistryTests(unittest.TestCase):
    def test_build_embedding_function_default(self) -> None:
        self.assertIsInstance(
            build_embedding_function("default"),
            DefaultEmbeddingFunction,
        )

    def test_build_embedding_function_unknown_model(self) -> None:
        with self.assertRaisesRegex(EmbeddingFunctionError, "Unknown embedding model"):
            build_embedding_function("bogus")

    def test_build_embedding_function_sentence_transformers(self) -> None:
        with patch(
            "chromy.embedding.registry.SentenceTransformerEmbeddingFunction",
        ) as sentence_transformer:
            result = build_embedding_function("sentence-transformers:all-MiniLM-L6-v2")

        sentence_transformer.assert_called_once_with(
            model_name="all-MiniLM-L6-v2",
        )
        self.assertIs(result, sentence_transformer.return_value)

    def test_model_max_tokens_returns_none_without_attribute(self) -> None:
        embedding_function = MagicMock(spec=[])  # no max_tokens attribute
        self.assertIsNone(model_max_tokens(embedding_function))

    def test_model_max_tokens_coerces_reported_limit_to_int(self) -> None:
        embedding_function = MagicMock()
        embedding_function.max_tokens.return_value = "256"
        self.assertEqual(model_max_tokens(embedding_function), 256)

    def test_model_max_tokens_returns_none_when_reported_value_is_not_castable(
        self,
    ) -> None:
        embedding_function = MagicMock()
        embedding_function.max_tokens.return_value = None
        self.assertIsNone(model_max_tokens(embedding_function))

    def test_model_max_tokens_returns_none_when_getter_raises(self) -> None:
        embedding_function = MagicMock()
        embedding_function.max_tokens.side_effect = ValueError("boom")
        self.assertIsNone(model_max_tokens(embedding_function))


if __name__ == "__main__":
    unittest.main()
