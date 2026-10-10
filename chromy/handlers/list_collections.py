from __future__ import annotations

from chromy.chroma_functions import list_collections
from chromy.output import format_collection_table, print_lines


def handle_list_collections() -> int:
    collections = list_collections()

    if not collections:
        print("No collections found.")
        return 0

    print_lines([format_collection_table(collections)])

    return 0
