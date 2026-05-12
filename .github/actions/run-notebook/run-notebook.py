#!/usr/bin/env python
"""Execute a notebook end-to-end against a real Jupyter kernel.

Replaces the cell-partitioning approach in convert-notebook.py with a
straight nbclient drive. This means:

  - !pip install and other shell magics run via the kernel's normal magic
    handling (the kernel spawns a subshell), exactly as in Colab/Jupyter Lab.
  - A code cell may freely mix `!pip install` and Python imports.
  - Errors surface with cell and traceback context, not as
    `line N: import: command not found`.

Notebook deps are installed into the same Python environment as the
runner (the kernel and runner share an interpreter), so `!pip install foo`
in cell 1 means cell 2 can `import foo`.

Usage:
  run-notebook.py <notebook-path>

Exits non-zero on any cell failure.
"""

import os
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

CELL_TIMEOUT = int(os.environ.get("NOTEBOOK_CELL_TIMEOUT", "600"))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run-notebook.py <notebook-path>", file=sys.stderr)
        return 2

    notebook_path = sys.argv[1]
    print(f"Executing {notebook_path}")

    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": os.path.dirname(notebook_path) or "."}},
    )

    try:
        client.execute()
    except CellExecutionError as exc:
        print(f"\nCell execution failed:\n{exc}", file=sys.stderr)
        return 1

    print(f"PASS — {notebook_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
