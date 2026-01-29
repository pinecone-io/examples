#!/usr/bin/env python
import re
import sys

import nbformat

# Flags that should be ignored when checking for unpinned packages
IGNORED_FLAGS = {
    "-q",
    "-qq",
    "-qqq",
    "--quiet",
    "-U",
    "--upgrade",
    "-e",
    "--editable",
    "-r",
    "--requirement",
    "--no-cache-dir",
    "--user",
    "--pre",
    "--force-reinstall",
    "--no-deps",
}


def extract_packages(install_line: str) -> list[str]:
    packages = []

    # Remove the pip install prefix
    line = re.sub(r"^[!%]pip\s+install\s+", "", install_line.strip())

    # Split by whitespace
    parts = line.split()

    skip_next = False
    for part in parts:
        if skip_next:
            skip_next = False
            continue

        # Skip flags
        if part.startswith("-"):
            # Some flags take an argument
            if part in {"-r", "--requirement", "-e", "--editable", "-c", "--constraint"}:
                skip_next = True
            continue

        # Skip git+ and other URL-based installs
        if "://" in part or part.startswith("git+"):
            continue

        # Skip local paths
        if part.startswith(".") or part.startswith("/"):
            continue

        # This should be a package name
        packages.append(part)

    return packages


def is_pinned(package: str) -> bool:
    # Check for version specifiers
    return bool(re.search(r"[<>=!~]", package))


def check_notebook(notebook_path: str) -> list[str]:
    unpinned = []
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        for cell_idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

            source = cell.source
            lines = source.split("\n")

            for line in lines:
                line = line.strip()
                if re.match(r"^[!%]pip\s+install\s+", line):
                    packages = extract_packages(line)
                    for pkg in packages:
                        if pkg and not is_pinned(pkg):
                            unpinned.append(f"  Cell {cell_idx}: {pkg}")

    except Exception as e:
        print(f"Error reading {notebook_path}: {e}", file=sys.stderr)

    return unpinned


def main():
    notebooks = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if not notebooks:
        print("No notebooks to check.")
        return

    has_error = False
    unpinned_by_notebook = {}

    for notebook_path in notebooks:
        if not notebook_path.endswith(".ipynb"):
            continue
        unpinned = check_notebook(notebook_path)
        if unpinned:
            unpinned_by_notebook[notebook_path] = unpinned
            has_error = True
        else:
            print(f"OK: {notebook_path}")

    if has_error:
        print()
        print("Unpinned dependencies found in the following notebooks:")
        for notebook, unpinned in unpinned_by_notebook.items():
            print(f"\n{notebook}:")
            for pkg in unpinned:
                print(pkg)
        print()
        print("Please pin dependency versions for reproducibility.")
        print("Example: %pip install pinecone==5.0.0")
        sys.exit(1)
    else:
        print("\nAll dependencies are pinned.")


if __name__ == "__main__":
    main()
