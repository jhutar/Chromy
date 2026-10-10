from __future__ import annotations

from rich import print

from chromy.chroma_functions import delete_collection, delete_data
from chromy.utilities import parse_key_value_pairs


def handle_delete_collection(collection: str) -> int:
    delete_collection(collection)
    print(f"[bold green]Deleted[/] collection '{collection}'.")
    return 0


def handle_delete_records(collection: str, where_clause: str) -> int:
    where = parse_key_value_pairs(where_clause, "--where")
    deleted = delete_data(collection, where)
    where_desc = ", ".join(f"{key}={value}" for key, value in where.items())
    print(
        f"[bold green]Deleted[/] {deleted} record(s) from collection '{collection}' "
        f"where {where_desc}."
    )
    return 0
