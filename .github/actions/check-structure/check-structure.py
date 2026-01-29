#!/usr/bin/env python
import nbformat
import os
import re
import sys


def check_notebook(notebook_path: str) -> list[str]:
    issues = []
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        if not nb.cells:
            issues.append("Notebook is empty")
            return issues

        # Check first cell is markdown (introduction)
        if nb.cells[0].cell_type != "markdown":
            issues.append("Missing introduction: first cell should be markdown")

        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        if not code_cells:
            return issues

        # Check imports are grouped in first code cell
        first_code_cell = code_cells[0]
        first_code_source = first_code_cell.source

        # Look for import statements in non-first code cells
        for i, cell in enumerate(code_cells[1:], start=2):
            source = cell.source.strip()
            lines = source.split("\n")
            for line in lines:
                line = line.strip()
                # Skip pip install lines, comments, and empty lines
                if (
                    line.startswith("#")
                    or line.startswith("!")
                    or line.startswith("%")
                    or not line
                ):
                    continue
                # Check for import statements at the start of lines
                if re.match(r"^(import |from \S+ import )", line):
                    # Allow conditional imports and inline imports
                    if "if " not in source and "try:" not in source:
                        issues.append(
                            f"Import found in code cell {i}: imports should be grouped in the first code cell"
                        )
                        break

        # Check for cleanup if notebook creates indexes
        all_code = "\n".join(c.source for c in code_cells)
        creates_index = (
            "create_index" in all_code
            or "pc.create_index" in all_code
            or ".create_index(" in all_code
        )

        if creates_index:
            last_cells_code = "\n".join(c.source.lower() for c in code_cells[-3:])
            has_cleanup = (
                "delete_index" in last_cells_code
                or "delete(" in last_cells_code
                or "cleanup" in last_cells_code
            )
            if not has_cleanup:
                issues.append(
                    "Missing cleanup: notebooks that create indexes should delete them at the end"
                )

    except Exception as e:
        issues.append(f"Error reading notebook: {e}")

    return issues


def main():
    has_error = False
    issues_by_notebook = {}

    for root, _, files in os.walk(".", topdown=True):
        if ".git" in root:
            continue
        for file in files:
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root, file)
                issues = check_notebook(notebook_path)
                if issues:
                    issues_by_notebook[notebook_path] = issues
                    has_error = True
                else:
                    print(f"OK: {notebook_path}")

    if has_error:
        print()
        print("Structure issues found in the following notebooks:")
        for notebook, issues in issues_by_notebook.items():
            print(f"\n{notebook}:")
            for issue in issues:
                print(f"  - {issue}")
        print()
        print("Please fix these issues following the notebook guidelines.")
        sys.exit(1)
    else:
        print("\nAll notebooks have valid structure.")


if __name__ == "__main__":
    main()
