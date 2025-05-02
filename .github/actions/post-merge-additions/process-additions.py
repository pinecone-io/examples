import os
import nbformat
from pinecone import Pinecone

notebook_added = os.environ['NOTEBOOK']

print(f"Processing new notebook {notebook_added}")

with open(notebook_added, 'r') as f:
    notebook = nbformat.read(f, as_version=4)
    print(notebook)

pc = Pinecone() # Reads PINECONE_API_KEY from environment variable

# TODO: add embeddings related to new notebook