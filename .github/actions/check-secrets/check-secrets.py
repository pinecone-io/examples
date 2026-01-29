#!/usr/bin/env python
import re
import sys

import nbformat

SECRET_PATTERNS = [
    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API key"),
    (r"sk-proj-[a-zA-Z0-9_-]{20,}", "OpenAI project API key"),
    (r"pcsk_[a-zA-Z0-9_]{20,}", "Pinecone API key"),
    (r"hf_[a-zA-Z0-9]{20,}", "Hugging Face token"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub personal access token"),
    (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key ID"),
    (r"xox[baprs]-[0-9a-zA-Z]{10,}", "Slack token"),
]

ALLOWLIST_PATTERNS = [
    r"os\.environ",
    r"os\.getenv",
    r"getpass",
    r"\$\{",
    r"<your[_-]",
    r"your[_-]api[_-]key",
    r"xxx",
    r"\.\.\.+",
]


def is_allowlisted(source: str, match_start: int, match_end: int) -> bool:
    context_start = max(0, match_start - 50)
    context_end = min(len(source), match_end + 50)
    context = source[context_start:context_end].lower()

    for pattern in ALLOWLIST_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            return True
    return False


def check_notebook(notebook_path: str) -> list[str]:
    findings = []
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        for cell_idx, cell in enumerate(nb.cells):
            source = cell.source
            for pattern, secret_type in SECRET_PATTERNS:
                for match in re.finditer(pattern, source):
                    if not is_allowlisted(source, match.start(), match.end()):
                        findings.append(
                            f"  Cell {cell_idx}: Possible {secret_type} detected"
                        )

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
        print("Possible secrets detected in the following notebooks:")
        for notebook, findings in findings_by_notebook.items():
            print(f"\n{notebook}:")
            for finding in findings:
                print(finding)
        print()
        print("Please remove hardcoded secrets and use environment variables instead.")
        print("Example: api_key = os.environ['PINECONE_API_KEY']")
        sys.exit(1)
    else:
        print("\nNo secrets detected.")


if __name__ == "__main__":
    main()
