import sys
import nbformat
from pinecone import Pinecone

notebook_changed = sys.argv[1]

print(f"Processing modified notebook {notebook_changed}")

with open(notebook_changed, 'r') as f:
    notebook = nbformat.read(f, as_version=4)
    print(notebook)

pc = Pinecone()

idx = pc.Index(host='hosturl') # TODO: adjust for arjun project

# TODO: update embeddings to reflect new contents of notebook