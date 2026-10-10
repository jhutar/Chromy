from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from chromy.embedding import (
    MODEL_HEADROOM,
    embed,
    embedding_budget_tokens,
)
from chromy.errors import EmbeddingFunctionError


class EmbedTest(unittest.TestCase):
    def test_embed_returns_empty_list_for_empty_chunks(self) -> None:
        self.assertEqual(embed([]), [])

    def test_embed_pairs_text_with_list_embeddings(self) -> None:
        with patch(
            "chromy.embedding.service.DefaultEmbeddingFunction",
            return_value=lambda chunks: ((1.0, 2.0), (3.0, 4.0)),
        ):
            result = embed(["first", "second"])

        self.assertEqual(
            result,
            [
                {"text": "first", "embedding": [1.0, 2.0]},
                {"text": "second", "embedding": [3.0, 4.0]},
            ],
        )

    def test_embed_uses_explicit_embedding_function(self) -> None:
        embedding_function = MagicMock(return_value=((9.0,),))
        result = embed(["only"], embedding_function=embedding_function)

        embedding_function.assert_called_once_with(["only"])
        self.assertEqual(result, [{"text": "only", "embedding": [9.0]}])

    def test_embedding_budget_tokens_scales_default_model_by_headroom(self) -> None:
        self.assertEqual(
            embedding_budget_tokens(DefaultEmbeddingFunction()),
            max(1, int(DefaultEmbeddingFunction().max_tokens() * MODEL_HEADROOM)),
        )

    def test_embedding_budget_tokens_uses_explicit_model_limit(self) -> None:
        embedding_function = MagicMock()
        embedding_function.max_tokens.return_value = 512

        self.assertEqual(embedding_budget_tokens(embedding_function), 409)

    def test_embedding_budget_tokens_uses_override_for_model_without_limit(
        self,
    ) -> None:
        embedding_function = MagicMock(spec=[])  # no max_tokens attribute
        self.assertEqual(
            embedding_budget_tokens(embedding_function, max_tokens=512), 409
        )

    def test_embedding_budget_tokens_raises_without_any_limit(self) -> None:
        embedding_function = MagicMock(spec=[])
        with self.assertRaisesRegex(
            EmbeddingFunctionError,
            "does not report a maximum input length",
        ):
            embedding_budget_tokens(embedding_function)


if __name__ == "__main__":
    unittest.main()
