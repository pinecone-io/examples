# Bugbot Configuration

This repository contains example notebooks and documentation for Pinecone.

## Review Focus

When reviewing changes, prioritize:

1. **Code clarity** - Example code should be easy to understand. Flag overly complex or clever patterns that may confuse readers. Prefer to use named keyword arguments over positional arguments.

2. **Correctness** - Ensure code examples are accurate and follow current best practices for Python and the Pinecone SDK.

3. **Documentation quality** - Markdown explanations should be clear, professional, and helpful. Avoid marketing language or superlatives.

4. **Timeless content** - Flag any language that will become stale:
   - Specific dates or years
   - Phrases like "recently released", "new", "cutting-edge", "state-of-the-art"
   - Version numbers in prose (unless required for compatibility)

5. **Notebook structure** - Notebooks should flow logically from introduction through conclusion, with clear section headers and explanatory markdown cells.

## Less Critical for Examples

- Strict type hints are not required in example code
- Comprehensive error handling is optional unless demonstrating error handling patterns
- Test coverage requirements do not apply to example notebooks

## Common Patterns to Accept

- Hardcoded API keys placeholders (e.g., `"YOUR_API_KEY"`) are expected
- Print statements for demonstrating output are intentional
- Simplified code that prioritizes readability over production optimization
