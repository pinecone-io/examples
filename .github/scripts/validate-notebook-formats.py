#!/usr/bin/env python
import nbformat
import os
import sys

def validate_notebook(notebook_path):
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        nbformat.validate(nb)
        print(f"Validated: {notebook_path}")
    except Exception as e:
        print(f"Validation failed for {notebook_path}: {e}", file=sys.stderr)
        return False
    return True

def main():
    has_error = False
    # Walk through the repository to find all .ipynb files
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root, file)
                if not validate_notebook(notebook_path):
                    has_error = True

    if has_error:
        sys.exit(1)

if __name__ == "__main__":
    main()
