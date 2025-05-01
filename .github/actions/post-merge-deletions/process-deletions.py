import sys
import nbformat
from pinecone import Pinecone

notebook_deleted = sys.argv[1]

print(f"Processing deletions to {notebook_deleted}")

pc = Pinecone()

idx = pc.Index(host='hosturl') # TODO: adjust for arjun project

# TODO: remove embeddings related to deleted notebook