import os
import nbformat
from pinecone import Pinecone

notebook_changed = os.environ['NOTEBOOK']

print(f"Processing modified notebook {notebook_changed}")

with open(notebook_changed, 'r') as f:
    notebook = nbformat.read(f, as_version=4)
    print(notebook)

pc = Pinecone() # Reads PINECONE_API_KEY from environment variable

# TODO: update embeddings to reflect new contents of notebook