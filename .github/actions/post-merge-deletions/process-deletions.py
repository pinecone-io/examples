import os
import nbformat
from pinecone import Pinecone

notebook_deleted = os.environ['NOTEBOOK']

print(f"Processing deletions to {notebook_deleted}")

pc = Pinecone() # Reads PINECONE_API_KEY from environment variable

# TODO: remove embeddings related to deleted notebook