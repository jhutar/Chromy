from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

from rich import print
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from chromy.errors import UnsupportedTextFileError
from chromy.utilities import ingest_file, parse_key_value_pairs

from ..utilities import is_probably_text_file

SUCCESS_EXIT_CODE: Final = 0
FAILURE_EXIT_CODE: Final = 1


def _get_absolute_path(file: str) -> str:
    """
    A helper method that, given a valid relative path to a file, returns its
    absolute path.

    Args:
        file (str): The relative path to the file.

    Raises:
        FileNotFoundError(): If the file does not exist.
    """
    if not os.path.exists(file):
        raise FileNotFoundError()

    file_path = Path(file)
    return str(file_path.resolve())


def _import_one(
    collection: str,
    file: str,
    metadata: dict[str, str] | None = None,
) -> int:
    absolute_path = _get_absolute_path(file)

    if not Path(absolute_path).is_file():
        raise FileNotFoundError()

    if not is_probably_text_file(absolute_path):
        raise UnsupportedTextFileError()

    return ingest_file(collection, absolute_path, metadata)


def _should_show_progress(file_count: int) -> bool:
    return file_count > 1 and sys.stdout.isatty()


def _truncate_file_name(file_name: str, max_length: int = 20) -> str:
    if len(file_name) <= max_length:
        return file_name

    return f"{file_name[: max_length - 3]}..."


def handle_import(
    collection: str,
    files: list[str],
    metadata_str: str | None = None,
) -> int:
    metadata = (
        parse_key_value_pairs(metadata_str, "--metadata")
        if metadata_str is not None
        else None
    )

    successful_imports = 0
    failed_imports = 0
    seen_paths: set[str] = set()
    unique_files: list[str] = []

    for file in files:
        try:
            absolute_path = _get_absolute_path(file)
        except FileNotFoundError:
            unique_files.append(file)
            continue

        if absolute_path in seen_paths:
            continue

        seen_paths.add(absolute_path)
        unique_files.append(file)

    show_progress = _should_show_progress(len(unique_files))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=True,
        disable=not show_progress,
    ) as progress:
        task_id = progress.add_task("Importing files...", total=len(unique_files))

        for file in unique_files:
            file_name = _truncate_file_name(Path(file).name)
            description = f"Importing [bold]{file_name}[/]..."
            progress.update(task_id, description=description)
            try:
                records_added = _import_one(collection, file, metadata)
                successful_imports += 1
                if not show_progress:
                    progress.console.print(
                        "[bold green]Added[/] "
                        f"{records_added} records from '{file}' to "
                        f"collection '{collection}'."
                    )
            except FileNotFoundError:
                failed_imports += 1
                progress.console.print(
                    f"[bold red]Error[/]: The file '{file}' was not found."
                )
            except UnsupportedTextFileError:
                failed_imports += 1
                progress.console.print(
                    f"[bold red]Error[/]: The file '{file}' is not a text file."
                )
            finally:
                progress.advance(task_id)

    print(
        f"Imported {successful_imports} file(s) successfully; {failed_imports} failed."
    )

    if failed_imports:
        return FAILURE_EXIT_CODE

    return SUCCESS_EXIT_CODE
