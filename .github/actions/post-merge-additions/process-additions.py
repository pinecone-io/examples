import sys
import nbformat
from pinecone import Pinecone

notebook_added = sys.argv[1]

print(f"Processing new notebook {notebook_added}")

pc = Pinecone()

idx = pc.Index(host='hosturl') # TODO: adjust for arjun project

# TODO: add embeddings related to new notebook