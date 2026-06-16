import os
from qdrant_client import QdrantClient, models
from uuid import uuid4
from itertools import islice
from .config import (DB_COLLECTION_NAME, DB_DENSE_INDEX_CONFIG, DB_SPARSE_INDEX_CONFIG, DB_SPARSE_INDEX_CONFIG_FR, DB_SPARSE_INDEX_CONFIG_EN,
                     EMBEDDING_MODEL_CONFIG, SPARSE_MODEL_CONFIG_FR, SPARSE_MODEL_CONFIG_EN
                     )

def batched(iterable, batch_size):
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch

def index_chunks(chunks):

    # TODO : securiser avec credentials la connexion vers la db
    client = QdrantClient(
        url=os.getenv("QDRANT_URL")
    )

    client.create_collection(
        collection_name=DB_COLLECTION_NAME,
        vectors_config=DB_DENSE_INDEX_CONFIG,
        sparse_vectors_config=DB_SPARSE_INDEX_CONFIG
    )

    embeddings = EMBEDDING_MODEL_CONFIG

    texts = [chunk.page_content for chunk in chunks]
    dense_vectors = embeddings.embed_documents(texts)

    # FIXME : ok de gerer les langues comme ca ?
    sparse_fr_vectors = SPARSE_MODEL_CONFIG_FR.embed_documents(texts)
    sparse_en_vectors = SPARSE_MODEL_CONFIG_EN.embed_documents(texts)

    points = []

    for i, chunk in enumerate(chunks):
        fr_vec = sparse_fr_vectors[i]
        en_vec = sparse_en_vectors[i]

        dense_vector_name = list(DB_DENSE_INDEX_CONFIG.keys())[0]
        sparse_vector_name_fr = list(DB_SPARSE_INDEX_CONFIG_FR.keys())[0]
        sparse_vector_name_en = list(DB_SPARSE_INDEX_CONFIG_EN.keys())[0]

        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector={
                    dense_vector_name: dense_vectors[i],
                    sparse_vector_name_fr: models.SparseVector(
                        indices=fr_vec.indices,
                        values=fr_vec.values,
                    ),
                    sparse_vector_name_en: models.SparseVector(
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
            collection_name=DB_COLLECTION_NAME,
            points=batch,
        )

__all__ = ["index_chunks"]
