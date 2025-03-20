#! /usr/bin/env python

# Convert a notebook to a Python script

import os
import sys
import nbformat
import shutil
from tempfile import mkdtemp
from tempfile import TemporaryDirectory

# Get the notebook filename from the command line
filename = "../../../" + sys.argv[1]
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
pip install --upgrade pip
pip install ipython
"""
run_commands = [activate_venv]
for cell in nb.cells:
    if cell.cell_type == "code":
        if "!pip" in cell.source or "%pip" in cell.source:
            # Replace all instances of "!pip" and "%pip" with "pip"
            command = cell.source.replace("!pip", "pip").replace("%pip", "pip")
            run_commands.append(command)

run_commands.append("""
# Run the notebook executable code
python "${SCRIPT_DIR}/notebook.py"
""")

run_commands.append("""
# Deactivate the virtual environment
deactivate
""")

# Save pip install commands to a run.sh script
run_script_path = os.path.join(temp_dir, 'run.sh')
with open(run_script_path, 'w', encoding="utf-8") as f:
    f.write("\n".join(run_commands))

print(f"Setup script saved to {run_script_path}")

# Collect cells that are not pip install commands
executable_cells = ["from IPython.display import display"]
for cell in nb.cells:
    if cell.cell_type == "code":
        if "pip" not in cell.source:
            #  Remove any lines that start with "!" or "%"
            #  These are "magic" commands such as "%matplotlib inline" that 
            #  are not executable outside of a notebook environment.
            executable = "\n".join([line for line in cell.source.split("\n") if not line.strip().startswith("!") and not line.strip().startswith("%")])
            executable_cells.append(executable)

# Save executable cells to a notebook.py file
script_path = os.path.join(temp_dir, 'notebook.py')
with open(script_path, 'w', encoding="utf-8") as f:
    for cell in executable_cells:
        f.write(cell + '\n')

print(f"Script saved to {script_path}")

# Output script and notebook path to github actions output
with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
    f.write(f"script_path={run_script_path}\n")
    f.write(f"notebook_path={script_path}\n")
    