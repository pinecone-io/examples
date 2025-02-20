#! /usr/bin/env python

# Convert a notebook to a Python script

import os
import sys
import nbformat
import shutil
from tempfile import mkdtemp
from tempfile import TemporaryDirectory

filename = '../../learn/search/semantic-search/semantic-search.ipynb'
print(f"Processing notebook: {filename}")
nb_source_path = os.path.join(os.path.dirname(__file__), filename)

temp_dir = mkdtemp()
venv_path = os.path.join(temp_dir, 'venv')
os.makedirs(venv_path, exist_ok=True)

# Copy file into temp directory
temp_nb_path = os.path.join(temp_dir, 'notebook.ipynb')
print(f"Copying notebook to {temp_nb_path}")
shutil.copy(nb_source_path, temp_nb_path)

with open(temp_nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Extract pip install commands (assumes they are written as "!pip install ..." or "%pip install ...")
# This grabs any line containing "pip install" in the script.
activate_venv = """
#!/bin/bash

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create new virtual environment
python -m venv "${SCRIPT_DIR}/venv"

# Activate the virtual environment
source "${SCRIPT_DIR}/venv/bin/activate"
"""
run_commands = [activate_venv]
for cell in nb.cells:
    if cell.cell_type == "code":
        if cell.source.startswith("!") or cell.source.startswith("%"):
            # Remove the leading "!" or "%" and the literal "pip install"
            command = cell.source[1:].strip()
            run_commands.append(command)

run_commands.append("""
# Run the notebook executable code
python "${SCRIPT_DIR}/notebook.py"
""")

run_commands.append("""
# Deactivate the virtual environment
deactivate
""")

# Save pip install commands to a setup.sh script
run_script_path = os.path.join(temp_dir, 'run.sh')
with open(run_script_path, 'w', encoding="utf-8") as f:
    f.write("\n".join(run_commands))

print(f"Setup script saved to {run_script_path}")

# Collect cells that are not pip install commands
executable_cells = []
for cell in nb.cells:
    if cell.cell_type == "code":
        if "pip" not in cell.source:
            executable_cells.append(cell)

# Save executable cells to a notebook.py file
script_path = os.path.join(temp_dir, 'notebook.py')
with open(script_path, 'w', encoding="utf-8") as f:
    for cell in executable_cells:
        f.write(cell.source + '\n')

print(f"Script saved to {script_path}")

# Output script path to github actions output
with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
    f.write(f"script_path={run_script_path}\n")