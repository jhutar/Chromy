from __future__ import annotations

from rich import print

from chromy.chroma_functions import create_collection


def handle_create_collection(
    collection: str,
    model: str = "default",
    max_tokens: int | None = None,
) -> int:
    collection_name = create_collection(collection, model=model, max_tokens=max_tokens)
    print(
        f"[bold green]Created[/]: collection '{collection_name}' with embedding "
        f"model '{model}'."
    )
    return 0
