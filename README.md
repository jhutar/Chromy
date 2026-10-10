<div align="center">
  <img src="logo.png" width="300" alt="Chromy logo: a detective evidence board" />

  # Chromy

  **A small local retrieval-augmented generation (RAG) CLI powered by Chroma.**
</div>

Chromy is a command-line utility for working with a local Chroma vector
database. It lets you create collections, ingest text files as chunked
embeddings, and run semantic similarity queries against stored documents. All
data is persisted locally, so a hosted database is not required. Chromy can also
be used by agentic coding tools through a skill (see the
[included example](./skills/chromy/SKILL.md)).

## Contents

- [What it does](#what-it-does)
- [Technology](#technology)
- [Codebase structure](#codebase-structure)
- [Installation](#installation)
- [Running the CLI](#running-the-cli)
- [Configuration](#configuration)
- [Commands](#commands)
- [How ingestion works](#how-ingestion-works)
- [Import metadata and filtering](#import-metadata-and-filtering)
- [Local development](#local-development)

## What it does

- manages local Chroma collections
- chunks files with `semchunk`
- generates embeddings with Chroma's default embedding function
- stores chunk text plus the source file's absolute path in metadata
- queries collections and prints readable results

## Requirements

- Python 3.12+
- a local environment able to install the project dependencies in `pyproject.toml`

## Technology

Chromy is a Python 3.12+ application packaged with setuptools and exposed as a
[Typer](https://typer.tiangolo.com/) command-line application. It stores vectors
in a persistent local [Chroma](https://www.trychroma.com/) database, uses
`semchunk` to split input, and uses Chroma's default embedding function to turn
chunks into vectors. `uv.lock` provides reproducible dependency resolution for
development and builds.

### Runtime libraries

- `chromadb` — persistent vector database used to store collections, embeddings, documents, and metadata.
- `openai` — declared for model and API integrations; the CLI does not call the
  OpenAI API directly.
- `pymupdf4llm` — declared for PDF-to-Markdown support; PDF ingestion is not yet
  wired into the text-only `import` command.
- `python-dotenv` — loads environment variables from local `.env` files.
- `rich` — provides styled terminal output and progress bars.
- `semchunk` — splits source documents into chunks before embedding.
- `tiktoken` — tokenization support used during chunking and embedding preparation.
- `transformers` — declared for model and tokenizer support; the CLI's current
  embedding implementation uses Chroma's default embedding function.
- `typer` — powers the CLI commands and argument parsing.

### Development libraries

- `mypy` — static type checking.
- `nuitka[onefile]` — builds standalone one-file executables.
- `pytest` — test runner for the project.
- `ruff` — linting and formatting.

## Codebase structure

```text
.
├── chromy/
│   ├── main.py                  # console entrypoint and .env loading
│   ├── cli.py                   # Typer commands, aliases, and CLI error handling
│   ├── chroma_functions.py      # Chroma client and collection/data operations
│   ├── utilities.py             # ingestion, querying, and text-file detection
│   ├── output.py                # Rich terminal formatting
│   ├── errors.py                # shared application exceptions
│   ├── chunking/service.py      # semantic text chunking
│   ├── embedding/service.py     # Chroma default embedding integration
│   └── handlers/                # command-specific orchestration
├── skills/chromy/SKILL.md       # example coding-agent integration
├── tests/                       # pytest unit tests
├── pyproject.toml               # package metadata and tool configuration
├── uv.lock                      # locked dependency graph
├── romeo_and_juliet.txt         # sample text for manual runs and tests
└── logo.png                     # project logo used above
```

The usual execution path is `main.py` → `cli.py` → a command handler. Handlers
coordinate reusable operations from `utilities.py` and `chroma_functions.py`;
chunking and embedding remain isolated in their respective service modules.

## Installation

Clone the repository, enter it, and install the locked runtime and development
dependencies with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone <repository-url>
cd Chromy
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Build

Build the source distribution and wheel with `uv`:

```bash
uv build
```

The build artifacts are written to `dist/`.

## Install as a tool with uv

The project exposes a `chromy` command through the Python packaging entrypoint.
Install it as a standalone `uv` tool from the project directory:

```bash
uv tool install .
```

After installation, run the CLI directly:

```bash
chromy --help
```

To install from a built wheel instead:

```bash
uv build
uv tool install dist/chromy-1.1.0-py3-none-any.whl
```

During development, install the tool in editable mode so changes in the working
tree are picked up without reinstalling:

```bash
uv tool install --editable .
```

## Running the CLI

The project entrypoint is available as the `chromy` command after installing the
tool:

```bash
chromy --help
```

You can also run it from the source tree without installing the tool:

```bash
uv run python -m chromy.main --help
```

## Configuration

`chromy.main` loads a local `.env` file via `python-dotenv`. No API key is
required for the current local embedding and storage workflow. If you want to
keep the database outside the working directory, add the optional setting below
to `.env` (which should remain uncommitted):

```dotenv
CHROMA_FOLDER=/absolute/path/to/a/parent-directory
```

### Chroma storage location

By default, Chromy uses Chroma's default persistent location behavior (a local
`chroma/` directory based on your current working directory when you run the
command).

You can override this with `CHROMA_FOLDER`.

- `CHROMA_FOLDER` must point to a **parent directory**.
- Chromy will store data in `<CHROMA_FOLDER>/chroma`.
- Relative paths are supported and are resolved from the current working directory.
- If `CHROMA_FOLDER` is set, it takes precedence over the default behavior.
- If the configured location is invalid or not writable, the command fails with an explicit error (no fallback to the default location).

Setting the variable once in `.zprofile` or `.profile` ensures a consistent usage of the variable.

Examples:

```bash
# absolute parent path
CHROMA_FOLDER=/tmp/chromy-data chromy list-collections

# relative parent path (resolved from current directory)
CHROMA_FOLDER=.local-data chromy create-collection notes
```

## Local development

After running `uv sync`, commands can be executed in the managed environment
without activating its virtual environment. A typical validation cycle is:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy chromy tests
uv build
```

### Running tests

Run the test suite with pytest:

```bash
uv run pytest -q
```

### Development checks

Run Ruff linting:

```bash
uv run ruff check .
```

Check Ruff formatting:

```bash
uv run ruff format --check .
```

Run static type checking with mypy:

```bash
uv run mypy chromy tests
```

## Commands

```text
list-collections | lc
create-collection <collection> | cc <collection>
delete-collection <collection> | dc <collection>
count <collection> | c <collection>
import <collection> <file> [<file> ...] [--metadata <key>=<value>[,...]] | i <collection> <file> [<file> ...] [--metadata <key>=<value>[,...]]
query <collection> <query_text> [--where <key>=<value>[,...]] | q <collection> <query_text> [--where <key>=<value>[,...]]
delete <collection> --where <key>=<value>[,...] | del <collection> --where <key>=<value>[,...]
```

### Aliases

- `lc` → `list-collections`
- `cc` → `create-collection`
- `dc` → `delete-collection`
- `c` → `count`
- `i` → `import`
- `q` → `query`
- `del` → `delete`

### Examples

Create a collection:

```bash
chromy create-collection notes
# alias
chromy cc notes
```

Add one or more text files:

```bash
chromy import notes ./docs/example.txt
chromy import notes ./docs/intro.md ./docs/setup.md
chromy import notes *.md
# alias
chromy i notes ./docs/example.txt
```

Attach metadata to every chunk, overriding the document identity:

```bash
chromy import notes /tmp/randomname \
    --metadata ticket=PROJ-123,file_name=/files/PROJ-123.md
```

Import a large batch of files with `find`:

```bash
find ./docs -type f \( -name '*.md' -o -name '*.txt' \) -exec chromy import notes {} +
```

Count stored records:

```bash
chromy count notes
# alias
chromy c notes
```

Search the collection:

```bash
chromy query notes "How do I configure this project?"
# filter by metadata (multiple pairs are ANDed)
chromy query notes "what is the status?" --where ticket=PROJ-123
chromy query notes "any open items?" --where ticket=PROJ-123,content_type=comment
# alias
chromy q notes "How do I configure this project?"
```

List collections:

```bash
chromy list-collections
# alias
chromy lc
```

Delete a collection:

```bash
chromy delete-collection notes
# alias
chromy dc notes
```

Delete records by metadata (all pairs must match, AND):

```bash
chromy delete notes --where file_name=/absolute/path/to/docs/example.txt
chromy delete notes --where ticket=PROJ-123,content_type=stale
# alias
chromy del notes --where file_name=/absolute/path/to/docs/example.txt
```

## How ingestion works

When you run `import`, each file is:

1. read from disk
2. split into chunks
3. embedded with Chroma's default embedding function
4. inserted into the target collection with the source file's absolute path stored
   in the `file_name` metadata field, plus any extra `--metadata` pairs.

Importing the same absolute path again replaces that file's existing records in
the collection (delete-then-add). Query results include the stored document
chunk, its ID, distance, and the `file_name` and any extra metadata fields when
available.

## Import metadata and filtering

### `import --metadata`

Attach comma-separated `key=value` pairs to every chunk of every imported file

```bash
chromy import tasks /tmp/tmp.123abc --metadata task_id=PROJ-123,file_name=/files/PROJ-123.md
```

- By default, `file_name` key is taken from actual file name.
- The reserved `file_name` key overrides the document's logical identity: the
  value becomes the chunk-replacement key and the stored `file_name` metadata
  instead of the real absolute path. It is stored verbatim — it is a logical
  identity (`/files/PROJ-123.md`, not necessarily a real file) — and must be
  unique within a collection.
- All non-reserved keys pass through to each chunk's metadata and are returned
  by `query`.
- Keys and values are trimmed; duplicate keys resolve to the last occurrence.
- Values are stored as strings only, so filtering via `--where` is exact string
  match; numeric or range comparisons are not supported.
- A literal comma inside a value is not supported.
- **Important:** two unrelated files imported with the same `file_name` override are
  the same logical document — importing the second replaces the first's chunks
  (the same delete-then-add behavior as re-importing one absolute path). This is
  intentional.

### `query --where` and `delete --where`

Filter results or records by metadata with comma-separated `key=value` pairs;
two or more pairs are combined with AND semantics:

```bash
chromy query tasks "what is the status?" --where task_id=PROJ-123
chromy delete tasks --where task_id=PROJ-123,content_type=stale
```

- Parsing rules match `--metadata` (trimming, last-wins, malformed entries
  rejected with exit code 1).
- `query` without `--where` queries the whole database.
- `delete` still requires `--where` and now accepts multiple pairs.

## Notes

- by default, collections are stored in a local persistent Chroma database in the current directory
- set `CHROMA_FOLDER` to override the parent location; Chromy will use `<CHROMA_FOLDER>/chroma`
- `import` requires the target collection to already exist
- `import` accepts one or more text-file paths; directories, missing files, and
  files detected as non-text are reported as failures
- unquoted glob patterns such as `*.md` are expanded by the shell before `chromy` starts
- quoted glob patterns such as `"*.md"` are treated as literal paths and are not expanded by `chromy`
- unmatched unquoted globs may behave differently by shell: `zsh` commonly fails before `chromy` starts, while `bash` may pass the literal pattern through depending on shell settings
- the CLI reports file-specific import failures and continues with the remaining files
- when importing multiple files in an interactive terminal, the CLI shows a Rich progress bar
