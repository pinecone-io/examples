#!/usr/bin/env python
import re
import sys

import nbformat

DATED_PATTERNS = [
    (
        r"\b(as of|since|starting)\s+(january|february|march|april|may|june|july|august|september|october|november|december)?\s*20\d{2}\b",
        "Date reference",
    ),
    (
        r"\brecently (released|added|announced|introduced|launched|updated)\b",
        "Recently phrase",
    ),
    (r"\bcoming soon\b", "Coming soon"),
    (r"\bnew (feature|release|version|update)\b", "New feature reference"),
    (
        r"\b(cutting-edge|state-of-the-art|latest and greatest|bleeding edge)\b",
        "Hyperbolic phrase",
    ),
    (r"\bjust (released|launched|announced|added)\b", "Just released phrase"),
    (r"\b(exciting|amazing|revolutionary) new\b", "Marketing language"),
]


def check_notebook(notebook_path: str) -> list[str]:
    findings = []
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        for cell_idx, cell in enumerate(nb.cells):
            if cell.cell_type != "markdown":
                continue

            source = cell.source.lower()
            for pattern, issue_type in DATED_PATTERNS:
                matches = list(re.finditer(pattern, source, re.IGNORECASE))
                for match in matches:
                    context = match.group(0)
                    findings.append(f'  Cell {cell_idx}: {issue_type} - "{context}"')

    except Exception as e:
        findings.append(f"Error reading notebook: {e}")

    return findings


def main():
    notebooks = sys.argv[1:] if len(sys.argv) > 1 else []

    if not notebooks:
        print("No notebooks to check.")
        return

    has_error = False
    findings_by_notebook = {}

    for notebook_path in notebooks:
        if not notebook_path.endswith(".ipynb"):
            continue
        findings = check_notebook(notebook_path)
        if findings:
            findings_by_notebook[notebook_path] = findings
            has_error = True
        else:
            print(f"OK: {notebook_path}")

    if has_error:
        print()
        print("Time-sensitive language found in the following notebooks:")
        for notebook, findings in findings_by_notebook.items():
            print(f"\n{notebook}:")
            for finding in findings:
                print(finding)
        print()
        print("Please use timeless phrasing instead:")
        print('  - Instead of "as of 2024" -> just state the fact')
        print('  - Instead of "recently released" -> "This example demonstrates..."')
        print('  - Instead of "new feature" -> describe what it does')
        print('  - Instead of "cutting-edge" -> describe the capability objectively')
        sys.exit(1)
    else:
        print("\nAll notebooks use timeless content.")


if __name__ == "__main__":
    main()
