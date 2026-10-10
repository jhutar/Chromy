from __future__ import annotations

from chromy.output import format_query_result, print_lines
from chromy.utilities import parse_key_value_pairs, run_query


def handle_query(
    collection: str,
    query_text: str,
    where_str: str | None = None,
) -> int:
    where = (
        parse_key_value_pairs(where_str, "--where") if where_str is not None else None
    )
    result = run_query(collection, query_text, where)
    print_lines(format_query_result(result))
    return 0
