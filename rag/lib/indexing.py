import os
from langchain_qdrant import FastEmbedSparse
from qdrant_client import QdrantClient, models
from uuid import uuid4
from itertools import islice
from rag.lib.config import DB_DENSE_VECTORS_CONFIG, DB_SPARSE_VECTORS_CONFIG, EMBEDDING_MODEL_CONFIG

def batched(iterable, batch_size):
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch

def index_chunks(chunks, collection_name):

    # TODO : securiser avec credentials la connexion vers la db
    client = QdrantClient(
        url="http://localhost:6333"
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=DB_DENSE_VECTORS_CONFIG,
        sparse_vectors_config=DB_SPARSE_VECTORS_CONFIG
    )

    # TODO : reflechir comment gerer ca (param ou config) ?
    embeddings = EMBEDDING_MODEL_CONFIG
    sparse_fr = FastEmbedSparse(model_name="Qdrant/bm25", avg_len=float(os.getenv("AVG_LEN_FR")), language="french")
    sparse_en = FastEmbedSparse(model_name="Qdrant/bm25", avg_len=float(os.getenv("AVG_LEN_EN")), language="english")

    texts = [chunk.page_content for chunk in chunks]
    dense_vectors = embeddings.embed_documents(texts)
    # FIXME : ok de gerer les langues comme ca ?
    sparse_fr_vectors = sparse_fr.embed_documents(texts)
    sparse_en_vectors = sparse_en.embed_documents(texts)

    points = []

    for i, chunk in enumerate(chunks):
        fr_vec = sparse_fr_vectors[i]
        en_vec = sparse_en_vectors[i]

        # TODO : est dependant de la config initiale de la DB, rendre generique !!
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector={
                    "dense": dense_vectors[i],
                    "sparse_fr": models.SparseVector(
                        indices=fr_vec.indices,
                        values=fr_vec.values,
                    ),
                    "sparse_en": models.SparseVector(
                        indices=en_vec.indices,
                        values=en_vec.values,
                    ),
                },
                payload={
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata,
                },
            )
        )

    for batch in batched(points, 64):
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )

__all__ = ["index_chunks"]
