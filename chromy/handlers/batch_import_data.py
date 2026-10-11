from __future__ import annotations

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

from chromy.chroma_functions import get_collection_embedding_context
from chromy.errors import UnsupportedTextFileError
from chromy.utilities import parse_key_value_pairs

from . import import_data

SUCCESS_EXIT_CODE: Final = 0
FAILURE_EXIT_CODE: Final = 1


def _scan_directory(directory: Path, metadata_file: Path | None) -> dict[str, Path]:
    """
    Scan the top level of ``directory`` for importable files.

    Args:
        directory (Path): The directory to scan (top level only).
        metadata_file (Path | None): The resolved metadata file path; if the
            metadata file sits inside ``directory`` it is excluded from the
            result.

    Returns:
        dict[str, Path]: Mapping of file name to path for every regular file.

    Raises:
        ValueError: If the directory contains a subdirectory or any other
            non-regular-file entry.
    """

    entries: dict[str, Path] = {}

    for entry in sorted(directory.iterdir()):
        if metadata_file is not None and entry.resolve() == metadata_file:
            continue

        if not entry.is_file():
            raise ValueError(
                f"The directory '{directory}' contains a subdirectory or "
                f"non-file entry: '{entry.name}'. Only regular files are "
                "supported."
            )

        entries[entry.name] = entry

    return entries


def _parse_metadata_file(metadata_file: Path) -> dict[str, dict[str, str]]:
    """
    Parse a batch metadata file.

    Each non-empty, non-comment line must have the form
    ``<file_name> <key>=<value>[,<key>=<value>...]`` with a single whitespace
    separator. Pairs are split on the first ``=`` so values may contain ``=``.

    Args:
        metadata_file (Path): The metadata file to parse.

    Returns:
        dict[str, dict[str, str]]: Mapping of file name to metadata dict.

    Raises:
        ValueError: If a line is malformed or lists a file name more than
            once.
    """

    metadata_by_file: dict[str, dict[str, str]] = {}

    with metadata_file.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split(None, 1)
            if len(parts) != 2 or not parts[1].strip():
                raise ValueError(
                    f"Invalid line {line_number} in metadata file "
                    f"'{metadata_file}'. Expected: "
                    "<file_name> <key>=<value>[,<key>=<value>...]"
                )

            file_name, pairs_str = parts

            try:
                pairs = parse_key_value_pairs(pairs_str, "--metadata-file")
            except ValueError as exc:
                raise ValueError(
                    f"Invalid line {line_number} in metadata file "
                    f"'{metadata_file}': {exc}"
                ) from exc

            if file_name in metadata_by_file:
                raise ValueError(
                    f"Duplicate file name '{file_name}' in metadata file "
                    f"'{metadata_file}'."
                )

            metadata_by_file[file_name] = pairs

    return metadata_by_file


def handle_batch_import(
    collection: str,
    directory: str,
    metadata_file: str | None = None,
) -> int:
    """
    Import every regular file in ``directory`` with per-file metadata.

    All input is validated before anything is imported: with a metadata file,
    every file in the directory (except the metadata file itself) must be
    listed, and every listed file must exist in the directory.

    Args:
        collection (str): The target collection.
        directory (str): Directory whose top-level files are imported.
        metadata_file (str | None): Optional file mapping file names to
            comma-separated key=value pairs.

    Returns:
        int: 0 on success, 1 if any file failed to import.

    Raises:
        FileNotFoundError: If the metadata file does not exist or is not a
            file.
        ValueError: If the directory is missing/invalid, or any validation
            rule is violated.
    """

    directory_path = Path(directory)

    if not directory_path.is_dir():
        raise ValueError(
            f"The directory '{directory}' does not exist or is not a directory."
        )

    resolved_metadata_file: Path | None = None

    if metadata_file is not None:
        if not Path(metadata_file).is_file():
            raise FileNotFoundError(
                f"The metadata file '{metadata_file}' was not found."
            )

        resolved_metadata_file = Path(metadata_file).resolve()

    entries = _scan_directory(directory_path, resolved_metadata_file)

    metadata_by_file: dict[str, dict[str, str]] = {}

    if resolved_metadata_file is not None:
        metadata_by_file = _parse_metadata_file(resolved_metadata_file)

        unlisted = [name for name in entries if name not in metadata_by_file]
        if unlisted:
            raise ValueError(
                "File(s) in the directory are not listed in the metadata "
                f"file '{metadata_file}': " + ", ".join(sorted(unlisted))
            )

        missing = [name for name in metadata_by_file if name not in entries]
        if missing:
            raise ValueError(
                "File(s) listed in the metadata file "
                f"'{metadata_file}' are not in the directory '{directory}': "
                + ", ".join(sorted(missing))
            )

    files: list[str] = [str(entries[name]) for name in sorted(entries)]
    metadata_for: dict[str, dict[str, str] | None] = {
        str(path): metadata_by_file.get(name) for name, path in entries.items()
    }

    successful_imports = 0
    failed_imports = 0

    show_progress = import_data._should_show_progress(len(files))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=True,
        disable=not show_progress,
    ) as progress:
        task_id = progress.add_task("Importing files...", total=len(files))

        if not files:
            progress.console.print("Imported 0 file(s) successfully; 0 failed.")
            return SUCCESS_EXIT_CODE

        embedding_function, chunk_size = get_collection_embedding_context(collection)

        for file in files:
            file_name = import_data._truncate_file_name(Path(file).name)
            description = f"Importing [bold]{file_name}[/]..."
            progress.update(task_id, description=description)
            try:
                records_added = import_data._import_one(
                    collection,
                    file,
                    metadata_for[file],
                    embedding_function=embedding_function,
                    chunk_size=chunk_size,
                )
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
