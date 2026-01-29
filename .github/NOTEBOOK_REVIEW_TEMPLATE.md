# Notebook Review Checklist

## Notebook Information

| Field | Value |
|-------|-------|
| **Notebook Path** | `path/to/notebook.ipynb` |
| **Reviewer** | @username |
| **Review Date** | YYYY-MM-DD |

---

## 1. Metadata and Setup

- [ ] Uses current Pinecone SDK version (`pinecone>=7.0.0`, not legacy `pinecone-client`)
- [ ] Other pinned dependencies are relatively up-to-date
- [ ] Colab and nbviewer badges present in header
- [ ] All dependencies pinned with reasonable versions
- [ ] No hardcoded API keys or secrets
- [ ] API key handling follows standard pattern: environment variable → getpass fallback

**Standard API key pattern:**
```python
import os
from getpass import getpass

api_key = os.environ.get("PINECONE_API_KEY") or getpass("Enter your Pinecone API key: ")
```

---

## 2. Structure and Flow

- [ ] Clear title and 1-2 paragraph introduction explaining purpose
- [ ] Prerequisites section listing required packages and API keys
- [ ] Logical section headers with consistent naming
- [ ] Cleanup section at end that deletes any created indexes
- [ ] Content is self-contained (doesn't reference external uncommitted files)

---

## 3. Code Quality

- [ ] All imports grouped in first code cell
- [ ] Named keyword arguments used (not positional)
- [ ] Meaningful variable names reflecting domain (e.g., `query_embedding` not `x`)
- [ ] Helper functions have docstrings
- [ ] No deprecated APIs or patterns
- [ ] Error handling where appropriate

---

## 4. Documentation

- [ ] Markdown cells explain "why" before each major code section
- [ ] Explanations are concise but complete for newcomers
- [ ] Terminology consistent with other notebooks in repository

---

## 5. Written Content Style

Reference: [`.cursor/rules/timeless-content.mdc`](../.cursor/rules/timeless-content.mdc)

### Timeless Language

- [ ] No specific dates, years, or time references (e.g., "as of 2024", "new in January")
- [ ] No phrases like "recently released", "coming soon", "cutting-edge", or "state-of-the-art"
- [ ] No version numbers in prose unless necessary for compatibility
- [ ] Uses evergreen phrasing ("This example demonstrates..." not "This new feature allows...")

### Professional Tone

- [ ] Focuses on what the code does, not marketing language
- [ ] Describes benefits/purpose honestly without exaggeration
- [ ] No superlatives or hype ("the best", "revolutionary", "game-changing")
- [ ] Lets the code speak for itself

---

## 6. Execution and Output

- [ ] Cells execute in order (no out-of-order dependencies)
- [ ] Outputs are meaningful and appropriately truncated
- [ ] Long operations show progress (tqdm or status messages)
- [ ] Random seeds set for reproducibility where applicable

---

## 7. Resource Management

- [ ] Indexes/connections properly cleaned up at end
- [ ] Context managers used where appropriate
- [ ] No orphaned resources if execution is interrupted midway

---

## Issues Found

<!-- List specific issues discovered during review -->

| Issue | Location | Severity |
|-------|----------|----------|
| Example: Uses deprecated `pinecone-client` | Cell 1 | High |
| Example: Missing cleanup section | End of notebook | Medium |

---

## Recommendations

<!-- Specific suggestions for improvement -->

1. 
2. 
3. 

---

## Review Summary

- **Overall Status:** ☐ Pass ☐ Pass with minor issues ☐ Needs revision
- **Priority Issues:** (count)
- **Minor Issues:** (count)

**Notes:**

<!-- Any additional context or observations -->
