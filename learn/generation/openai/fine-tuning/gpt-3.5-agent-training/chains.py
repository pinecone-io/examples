from uuid import uuid4

import openai
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm


class VectorDBChain:
    name: str = "Vector Search Tool"
    description: str = "A tool for finding information about a topic."

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        index_name: str,
        pinecone_api_key: str,
        *,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        pc = Pinecone(api_key=pinecone_api_key)
        if index_name not in pc.list_indexes().names():
            spec = ServerlessSpec(cloud=cloud, region=region)
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=spec,
            )
        self.index = pc.Index(name=index_name)

    def _embed(self, texts: list[str]):
        res = openai.Embedding.create(input=texts, engine="text-embedding-ada-002")
        embeds = [x["embedding"] for x in res["data"]]
        return embeds

    def query(self, text: str) -> list[str]:
        xq = self._embed([text])[0]
        res = self.index.query(vector=xq, top_k=3, include_metadata=True)
        matches = res.get("matches", res.matches if hasattr(res, "matches") else [])
        documents = [
            m.get("metadata", m.metadata if hasattr(m, "metadata") else {}).get(
                "document"
            )
            for m in matches
        ]
        return [d for d in documents if d is not None]

    def build_index(self, documents: list[str], batch_size: int = 100):
        for i in tqdm(range(0, len(documents), batch_size)):
            # get end of batch
            i_end = min(i + batch_size, len(documents))
            batch = documents[i:i_end]
            # create document/context embeddings
            xd = self._embed(batch)
            # create metadata
            metadata = [{"document": x} for x in batch]
            ids = [str(uuid4()) for _ in batch]
            vectors = list(zip(ids, xd, metadata))
            self.index.upsert(vectors=vectors)
