# Contributing to Pinecone Examples

We appreciate your contributions to help us improve and maintain this community resource!

## Quick Contributions

For quick fixes like correcting a typo or patching an obvious bug, feel free to open a pull request directly.

## Larger Changes

If you're considering a larger or more involved change to this repository, its organization, or the functionality of one of the examples, please first [open a new issue](https://github.com/pinecone-io/examples/issues/new) and describe your proposed changes so we can discuss them together before you invest a ton of time or effort into making changes.

## Development Setup

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

The hooks will automatically format and lint your code when you commit.

### Running Checks Manually

```bash
# Lint a file
poetry run ruff check path/to/file.py

# Format a file
poetry run ruff format path/to/file.py
```

## Notebook Guidelines

- Ensure the notebook runs cleanly with "Restart Kernel and Run All"
- Include cleanup cells at the end to delete any indexes created
- See `.cursor/rules/notebook-guidelines.mdc` for detailed guidelines
