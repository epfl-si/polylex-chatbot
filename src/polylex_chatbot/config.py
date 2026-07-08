from qdrant_client import models
from langchain_qdrant import FastEmbedSparse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter

from polylex_chatbot.llm_context_utils import *
from polylex_chatbot.constants import ARTICLE_PATTERN, DB_SPARSE_INDEX_CONFIG_FR, DB_SPARSE_INDEX_CONFIG_EN, NB_MAX_ITEMS_SENT

# chunking

def create_documents_splitter():
    chunk_size = int(os.getenv("CHUNK_SIZE_NB_CHARS"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP_NB_CHARS"))
    if chunk_overlap == 0:
        separators = [
            ARTICLE_PATTERN,
            "\n\n",
            "\n",
            " ",
            ""
        ]
    else:
        separators = [
            "\n\n",
            "\n",
            " ",
            ""
        ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        is_separator_regex=True,
        keep_separator=True,
        add_start_index=True
    )
    return splitter.split_documents


# database

def get_db_dense_index_config():
    return {
        "dense": models.VectorParams(
            size=int(os.getenv("MODEL_EMBEDDINGS_DIM_VECTOR")),
            distance=models.Distance.COSINE
        )
    }

def init_db_client(lang):
    if lang == "fr":
        return QdrantVectorStore.from_existing_collection(
            url=os.getenv("QDRANT_URL"),
            embedding=get_embeddings_model_config(),
            sparse_embedding=get_sparse_model_config_fr(),
            collection_name=os.getenv("DB_COLLECTION_NAME"),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=list(get_db_dense_index_config().keys())[0],
            sparse_vector_name=list(DB_SPARSE_INDEX_CONFIG_FR.keys())[0]
        )
    elif lang == "en":
        return QdrantVectorStore.from_existing_collection(
            url=os.getenv("QDRANT_URL"),
            embedding=get_embeddings_model_config(),
            sparse_embedding=get_sparse_model_config_en(),
            collection_name=os.getenv("DB_COLLECTION_NAME"),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=list(get_db_dense_index_config())[0],
            sparse_vector_name=list(DB_SPARSE_INDEX_CONFIG_EN.keys())[0]
        )
    return None


# embeddings (sparse + dense)

def get_embeddings_model_config():
    model_embeddings_name = os.getenv("MODEL_EMBEDDINGS_NAME")
    if model_embeddings_name == "Alibaba-NLP/gte-multilingual-base":
        return OpenAIEmbeddings(
            model=os.getenv("MODEL_EMBEDDINGS_NAME"),
            base_url=os.getenv("MODELS_BASE_URL"),
            api_key=os.getenv("MODEL_EMBEDDINGS_API_KEY"),
            check_embedding_ctx_length=False
        )
    else:
        return OpenAIEmbeddings(
            model=os.getenv("MODEL_EMBEDDINGS_NAME"),
            base_url=os.getenv("MODELS_BASE_URL"),
            api_key=os.getenv("MODEL_EMBEDDINGS_API_KEY")
        )

def get_sparse_model_config_fr():
    return FastEmbedSparse(model_name=os.getenv("MODEL_SPARSE_NAME"), avg_len=float(os.getenv("AVG_LEN_FR")), language="french")

def get_sparse_model_config_en():
    return FastEmbedSparse(model_name=os.getenv("MODEL_SPARSE_NAME"), avg_len=float(os.getenv("AVG_LEN_EN")), language="english")


# generation

def get_llm_model_config():
    return ChatOpenAI(
        model=os.getenv("MODEL_LLM_NAME"),
        base_url=os.getenv("MODELS_BASE_URL"),
        api_key=os.getenv("MODEL_LLM_API_KEY"),
        max_tokens=2000,
        temperature=0.0
    )

def prepare_llm_context(chunks, scores):
    #return prepare_llm_context_n_chunks(chunks, scores, NB_MAX_ITEMS_SENT)
    #return prepare_llm_context_max_n_chunks(chunks, scores, NB_MAX_ITEMS_SENT)
    #return prepare_llm_context_n_documents_or_chunks(chunks, scores, NB_MAX_ITEMS_SENT)
    return prepare_llm_context_modular_context(chunks, scores, NB_MAX_ITEMS_SENT)
