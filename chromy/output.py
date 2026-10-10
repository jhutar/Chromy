from __future__ import annotations

from collections.abc import Mapping, Sequence

from chromadb import QueryResult
from rich.console import Console
from rich.rule import Rule
from rich.text import Text

CONSOLE = Console()


def print_lines(lines: Sequence[Rule | Text | str]) -> None:
    for line in lines:
        CONSOLE.print(line)


def format_collection_names(collections: Sequence[str]) -> list[Text]:
    return [Text(f"· {collection}") for collection in collections]


def format_count_message(collection_name: str, count: int) -> str:
    return (
        f"The '{collection_name}' collection contains [bold green]{count}[/] records."
    )


def format_query_result(result: QueryResult) -> list[Rule | Text]:
    ids = result.get("ids", [[]])
    documents = result.get("documents", [[]])
    distances = result.get("distances", [[]])
    metadatas = result.get("metadatas", [[]])

    first_ids = ids[0] if ids else []
    first_documents = documents[0] if documents else []
    first_distances = distances[0] if distances else []
    first_metadatas = metadatas[0] if metadatas else []

    if not first_ids:
        return [Text.from_markup("[yellow]No results found.[/]")]

    lines: list[Rule | Text] = [Rule(title="Query results")]

    for index, document_id in enumerate(first_ids, start=1):
        lines.append(
            Text.from_markup(f"[bold]{index}[/].\t[green]id[/]\t\t{document_id}")
        )
        i = index - 1

        if i < len(first_distances):
            lines.append(
                Text.from_markup(f"\t[green]distance[/]\t{first_distances[i]}")
            )

        if i < len(first_metadatas):
            metadata = first_metadatas[i]

            if isinstance(metadata, Mapping):
                file_name = metadata.get("file_name")

                if file_name:
                    lines.append(
                        Text.from_markup(f"\t[green]file_name[/]\t{file_name}")
                    )

                extra_items = sorted(
                    (str(key), value)
                    for key, value in metadata.items()
                    if key != "file_name"
                )

                for key, value in extra_items:
                    lines.append(Text.from_markup(f"\t[green]{key}[/]\t{str(value)}"))

        if i < len(first_documents):
            lines.append(Text.from_markup("\n[bold green]Retrieved contents[/]\n"))
            lines.append(Text(first_documents[i]))

        lines.append(Rule())

    return lines
