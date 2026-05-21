---
name: convert-to-marimo
description: This skill should be used when the user asks to "convert a notebook to marimo", "migrate a Jupyter notebook to marimo", "rewrite a notebook in marimo", or wants to modernize an existing .ipynb file into a high-quality marimo .py notebook.
version: 0.1.0
---

# Convert Jupyter Notebook to Marimo

Convert an existing `.ipynb` Jupyter notebook into a high-quality marimo `.py` notebook, updating dependencies, adopting marimo affordances, improving code quality, and revising prose to meet the Pinecone examples writing guidelines.

**Writing guidelines reference:** See **`.ai/writing-guidelines.md`** for voice, tone, and style.

## Phase 1: Initial Conversion

### Convert with marimo

```bash
uv run marimo convert path/to/notebook.ipynb -o docs/notebook-name.py
```

### Start in sandbox mode for development

```bash
uvx marimo edit --sandbox docs/notebook-name.py --no-token
```

Sandbox mode creates an isolated environment from the notebook's `# /// script` inline metadata — none of the project's root dependencies bleed in. Always develop in sandbox mode so the dependency list stays honest.

### Explore the code_mode API at the start of each session

```python
import marimo._code_mode as cm
help(cm)
```

The API can change between marimo versions; verify it before using it.

---

## Phase 2: Dependencies

### Update the `# /// script` metadata block

The converted file will have a metadata block at the top. This is the source of truth for the notebook's dependencies when running in sandbox mode.

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23.6",
#     "pinecone==9.0.1",
#     "datasets==3.5.1",
# ]
# ///
```

**Rules:**
- Pin every dependency to a specific version with `==`
- Include `marimo>=0.23.6` (or current version)
- Pin the Pinecone SDK to `9.0.1` (or latest)
- Keep this block in the notebook file — **never** add notebook deps to the root `pyproject.toml`

### Remove unused dependencies

After conversion, audit the declared deps against what's actually imported. Common removals:
- `tqdm` — replaced by `mo.status.progress_bar()`
- `numpy` — often imported by Jupyter cells that don't need it directly
- `pinecone-notebooks` — Colab-only authentication widget, not needed in marimo

### Watch for library compatibility breaks

Check whether newer versions of dependencies break with the data sources used. A known example: `datasets>=4.0` dropped support for custom loading scripts (e.g. `Helsinki-NLP/tatoeba`). Pin to the last working version and note why in a comment.

---

## Phase 3: Remove Jupyter/Colab Artifacts

Delete or replace:

- **Colab/nbviewer badges** — strip from the header markdown cell
- **`!pip install` cells** — dependencies are declared in `# /// script`, not installed at runtime
- **Colab authentication cells** — `pinecone_notebooks.colab.Authenticate()` and similar widgets
- **"Note: pip install is formatted for Jupyter" markdown** — not relevant in marimo
- **`## Installation` section headings** — no installation step needed
- **`trust_remote_code=True` notes** — keep the argument but remove surrounding Jupyter-specific explanation
- **References to "this notebook", "run this cell", "Jupyter"** — rewrite as plain prose

---

## Phase 4: Update the Pinecone SDK to 9.0.1

Replace deprecated method calls with the `pc.indexes.*` namespace:

| Old | New |
|-----|-----|
| `pc.has_index(name=x)` | `pc.indexes.exists(name=x)` |
| `pc.create_index(name=x, ...)` | `pc.indexes.create(name=x, ...)` |
| `pc.describe_index(name=x)` | `pc.indexes.describe(name=x)` |
| `pc.delete_index(name=x)` | `pc.indexes.delete(name=x)` |
| `pc.Index(host=desc.host)` | `pc.index(name=x)` |
| `index.search(namespace=ns, query={"top_k": k, "inputs": {...}})` | `index.search(namespace=ns, top_k=k, inputs={...})` |
| `results["result"]["hits"]` / `result["_score"]` | `results.result.hits` / `hit.score` |

**Always use keyword argument names** in all Pinecone API calls — positional args are harder to read and more fragile across SDK versions.

---

## Phase 5: Adopt Marimo Affordances

### Replace `print()` output with tables

```python
# Before
for result in results:
    print(f"{result['text']} (score: {result['score']})")

# After
mo.ui.table([{"text": r["text"], "score": r["score"]} for r in results])
```

Use `mo.vstack()` to combine a heading with a table:
```python
mo.vstack([
    mo.md(f"**Query:** {query}"),
    mo.ui.table(data, show_column_summaries=False),
])
```

### Replace `tqdm` with `mo.status.progress_bar()`

```python
# Before
for batch in tqdm(batches):
    index.upsert(batch)

# After
for batch in mo.status.progress_bar(batches, title="Upserting", show_rate=True, show_eta=True):
    index.upsert(batch)
```

When passing a `range`, omit `total` — marimo infers it from `len(range(...))`.

### Wrap destructive operations in `mo.ui.run_button()`

Marimo's reactive model means all cells run automatically. A cleanup cell that deletes an index will fire immediately — gate it with a button:

```python
# Cell 1 — display the button
delete_button = mo.ui.run_button(label="Delete index")
delete_button

# Cell 2 — action (separate cell — can't read .value in the same cell that creates it)
mo.stop(not delete_button.value)
pc.indexes.delete(name=index_name)
```

### Use `mo.callout()` for status messages

```python
mo.callout(mo.md("API key loaded from environment."), kind="success")
mo.callout(mo.md("Enter your API key to continue."), kind="info")
mo.callout(mo.md("**Error:** index not found."), kind="danger")
```

Kinds: `neutral`, `info`, `warn`, `success`, `danger`.

### Handle API key input

Users running locally can set `PINECONE_API_KEY` in their environment or a `.env` file (marimo reads `.env` on startup). Users in molab need a password input:

```python
# Cell 1 — input (hide_code=True)
env_key = os.environ.get("PINECONE_API_KEY", "")
api_key_input = mo.ui.text(
    kind="password",
    placeholder="pcsk_...",
    label="Pinecone API Key",
    value=env_key,
    full_width=True,
)
(
    mo.callout(mo.md("API key loaded from environment."), kind="success")
    if env_key
    else mo.vstack([
        mo.callout(mo.md("Enter your Pinecone API key. Get a free key at [app.pinecone.io](https://app.pinecone.io)."), kind="info"),
        api_key_input,
    ])
)

# Cell 2 — validate and create client (hide_code=True for the stop check; visible for pc = Pinecone(...))
api_key = api_key_input.value
mo.stop(
    not api_key,
    mo.callout(mo.md("**API key required.** Enter your key above to continue."), kind="danger"),
)

# Cell 3 — visible: instantiate the client
pc = Pinecone(api_key=api_key, source_tag="pinecone_examples:...")
```

### Display data with `mo.ui.table()`

HuggingFace datasets and lists of dicts both work directly:

```python
mo.ui.table(dataset, page_size=10)
mo.ui.table(records, page_size=10)
```

### Add interactive inputs for exploration

At the end of the notebook, add a "Try It Yourself" section:

```python
query_input = mo.ui.text(value="default query", full_width=True)
lang_select = mo.ui.radio(
    options={"All": None, "English": "en", "Spanish": "es"},
    value="All",
)
mo.vstack([query_input, lang_select])
```

Then in the next cell:
```python
search(query_input.value, lang=lang_select.value)
```

Results update when the user changes either input.

---

## Phase 6: Code Quality

### Remove over-explaining comments

Only comment on the non-obvious WHY — not on what the code does. Delete comments like:
- `# Initialize client`
- `# convert to record format`
- `# flatten and shuffle for ease of use`
- `# Here, we create a record for each sentence in the dataset`

Keep comments that explain constraints, workarounds, or non-obvious choices.

### Decompose monolithic functions

If the converted notebook has a large function doing multiple things, split it:
- Separate filtering from reshaping from formatting
- Name each function after its single responsibility
- Parameterize functions properly — avoid globals captured by closures

### Avoid multiply-defined variables across cells

Marimo's static analysis flags top-level variables defined in more than one cell. When two cells have the same local variable names, either:
- Use different names
- Inline the computation (no assignment)
- Consolidate both cells into one

### Watch for marimo cell configuration issues

Cells created with `code_mode` default to `hide_code=True`. Always explicitly set `hide_code=False` for code cells that should be visible. Verify with:

```python
for cell in ctx.cells:
    kind = "md  " if cell.config.hide_code else "code"
    print(f"[{cell.id}] {kind}: {cell.code[:60]!r}")
```

---

## Phase 7: Prose and Structure

Follow `.ai/writing-guidelines.md`. Key points for marimo conversion:

### Voice and tone
- Use "we" throughout (collaborative tutorial voice)
- Factual and collegial — no "super helpful!", "Neat!", "magic", "Congrats"
- No superlatives, no marketing language
- No time references ("recently added", "new feature")

### Structure
- **Intersperse explanations between code cells** — don't dump all prose at the top
- Put "why" before the code it motivates (e.g. explain why a keyword is ambiguous just before the filter that uses it)
- After showing data, explain what you see before proceeding
- Use `###` subheadings within sections for skimmability

### Merge adjacent text cells
When two or more markdown cells appear next to each other with no code between them, consolidate them into one unless they serve structurally distinct purposes (e.g. a section heading followed by body text can be merged).

### Remove Jupyter-specific prose
- "Run the cell below" → remove or rewrite
- "This notebook will..." → "This example demonstrates..."
- References to Colab, Google Colab, nbviewer → remove entirely
- "In this notebook" → rewrite without the word "notebook"

### Section heading guidelines
- Headings should be short noun phrases, not full sentences
- "Meaning Over Keywords" not "Semantic Search considers the meaning of the query"
- "How It Works" not "Wait, how is this working?"
- "Cleanup" not "Demo Cleanup"

---

## Phase 8: Final Checks

### Run ruff
```bash
uv run ruff check docs/notebook-name.py
uv run ruff format docs/notebook-name.py
```

The CI pipeline runs `ruff check` and `ruff format --check` on changed `.py` files. Fix all issues before committing.

### Verify sandbox runs
```bash
uvx marimo edit --sandbox docs/notebook-name.py --no-token
```

Run through the notebook end-to-end to confirm all cells execute correctly in the isolated environment.

### Verify no root pyproject.toml changes
Notebook dependencies belong in the `# /// script` block only. If marimo's package manager added anything to `pyproject.toml` during development, revert those changes and restore `uv.lock` from main:
```bash
git checkout origin/main -- uv.lock
```

---

## Common Pitfalls

| Problem | Fix |
|---------|-----|
| `mo.ui.run_button().value` read in same cell | Split button creation and value access into separate cells |
| Multiply-defined variable names across cells | Inline the call or use distinct names |
| Cells created with `code_mode` are hidden | Explicitly set `hide_code=False` |
| marimo package manager edits `pyproject.toml` | Revert — deps belong in `# /// script` only |
| `datasets>=4` breaks dataset loading scripts | Pin to last working version (e.g. `datasets==3.5.1`) |
| Old SDK calls (`pc.has_index`, `pc.Index(host=...)`) | Replace with `pc.indexes.*` namespace |
| `tqdm` still imported but unused | Remove it — use `mo.status.progress_bar()` |
| `source_tag` in `pc = Pinecone(...)` | Keep it, but note in prose it's for internal Pinecone analytics — users should not include it in their own apps |
| Index deletion cell auto-fires on notebook load | Wrap in `mo.ui.run_button()` |
