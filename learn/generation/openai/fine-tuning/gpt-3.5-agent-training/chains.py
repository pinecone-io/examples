from uuid import uuid4

import openai
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm

# text-embedding-ada-002 dimension
EMBEDDING_DIM = 1536


class VectorDBChain:
    name: str = "Vector Search Tool"
    description: str = "A tool for finding information about a topic."

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        index_name: str,
        environment: str,
        pinecone_api_key: str,
    ):
        pc = Pinecone(api_key=pinecone_api_key)
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=environment),
            )
        self.index = pc.Index(index_name)

    def _embed(self, texts: list[str]):
        res = openai.Embedding.create(input=texts, engine="text-embedding-ada-002")
        embeds = [x["embedding"] for x in res["data"]]
        return embeds

    def query(self, text: str) -> list[str]:
        # create query vector
        xq = self._embed([text])[0]
        res = self.index.query(vector=xq, top_k=3, include_metadata=True)
        # get documents
        documents = [x.metadata["text"] for x in res.matches]
        return documents

    def build_index(self, documents: list[str], batch_size: int = 100):
        for i in tqdm(range(0, len(documents), batch_size)):
            # get end of batch
            i_end = min(i + batch_size, len(documents))
            batch = documents[i:i_end]
            # create document/context embeddings
            xd = self._embed(batch)
            # create metadata
            metadata = [{"text": x} for x in batch]
            ids = [str(uuid4()) for _ in batch]
            # add to index
            self.index.upsert(vectors=zip(ids, xd, metadata))
