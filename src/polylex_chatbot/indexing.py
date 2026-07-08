import os
from uuid import uuid4
from itertools import islice
from dotenv import find_dotenv, set_key
from qdrant_client import QdrantClient, models

from .constants import DB_SPARSE_INDEX_CONFIG, DB_SPARSE_INDEX_CONFIG_FR, DB_SPARSE_INDEX_CONFIG_EN
from .config import get_db_dense_index_config, get_embeddings_model_config, get_sparse_model_config_fr, get_sparse_model_config_en

def save_collection_name(collection_name, env_file):
    var_name = "DB_COLLECTION_NAME"
    value = str(collection_name)
    set_key(
        dotenv_path=find_dotenv(filename=env_file),
        key_to_set=var_name,
        value_to_set=value
    )
    os.environ[var_name] = value

def batched(iterable, batch_size):
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch

def index_chunks(chunks, collection_name, collection_description, env_file):
    save_collection_name(collection_name, env_file)

    client = QdrantClient(
        url=os.getenv("QDRANT_URL")
    )

    collection_metadata = {
        "description": collection_description
    }

    client.create_collection(
        collection_name=os.getenv("DB_COLLECTION_NAME"),
        vectors_config=get_db_dense_index_config(),
        sparse_vectors_config=DB_SPARSE_INDEX_CONFIG,
        metadata=collection_metadata
    )

    texts = [chunk.page_content for chunk in chunks]
    dense_vectors = get_embeddings_model_config().embed_documents(texts)

    sparse_fr_vectors = get_sparse_model_config_fr().embed_documents(texts)
    sparse_en_vectors = get_sparse_model_config_en().embed_documents(texts)

    points = []

    for i, chunk in enumerate(chunks):
        fr_vec = sparse_fr_vectors[i]
        en_vec = sparse_en_vectors[i]

        dense_vector_name = list(get_db_dense_index_config().keys())[0]
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
            collection_name=os.getenv("DB_COLLECTION_NAME"),
            points=batch,
        )

__all__ = ["index_chunks"]
