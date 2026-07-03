import re
import os
from pathlib import Path
from qdrant_client import models
from langchain_qdrant import FastEmbedSparse
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter

# paths
DATA_PATH = Path.cwd() / "data"
STATS_PATH = Path.cwd() / "stats"
CHUNKS_PATH = Path.cwd() / "stats"
EVALUATION_RESULTS_PATH = Path.cwd() / "evaluations"
COMPARISONS_RESULTS_PATH = Path.cwd() / "comparisons"

# api
LEXES_API_URL = "https://polylex-admin.epfl.ch/api/v1/lexes?isAbrogated=0"

# rcp
RCP_MODEL_NOT_LOADED_TIMEOUT_SECONDS = 30

# languages
LANGUAGES = ["fr", "en"]
# TODO : change name
HARD_CODED_LANGS = {
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/1.1.7-Ruling-EPFL-Geneve.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/1.1.7-Ruling-EPFL-Vaud-.pdf": "fr",
    "https://fedlex.data.admin.ch/filestore/fedlex.data.admin.ch/eli/cc/2008/857/20250501/fr/pdf-a/fedlex-data-admin-ch-eli-cc-2008-857-20250501-fr-pdf-a-1.pdf": "fr",
    "https://ethrat.stage.mxm.agency/wp-content/uploads/2021/10/Reglement_comites_CEPF_0.pdf": "fr",
    "https://ethrat.stage.mxm.agency/wp-content/uploads/2021/10/Directives_ombudsman_CEPF.pdf": "fr",
    "https://ethrat.stage.mxm.agency/wp-content/uploads/2021/10/Directives_information.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/3.1.4_fondation_usa_en-1.pdf": "en",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/LEX-3.1.7_annexe2.pdf": "en",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2020/09/LEX-3.1.7_annexe4.pdf": "en",
    "https://ethrat.stage.mxm.agency/wp-content/uploads/2021/10/210101_Richtlinien_NB_ETHR_F.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/4.4.1_Regles_applications_fr_an-1.pdf": "fr", # FIXME : fr + en, grave ?
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/4.4.1_Description_fonction_fr_an.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/LEX-5.1.0.3.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/5.7.1_conv_tresorerie_all.pdf": "fr", # FIXME : fr + de, grave ?
    "https://ethrat.ch/wp-content/uploads/2021/09/Immobilienweisung_ETH-Bereich_2016_F_0.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2020/07/LEX-1.1.9.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2021/12/LEX-1.1.12.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2022/03/LEX-5.1.0.4.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2026/01/LEX-1.1.17.pdf": "en",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/5.7.2_dir_placement_all.pdf": "fr", # FIXME : fr + de, grave ?
    "https://www.epfl.ch/about/overview/wp-content/uploads/2026/05/LEX-4.6.0.1.pdf": "fr"
}

# chunking
ARTICLE_PATTERN = re.compile(r"\b(?:Article\s+\d+|Art\.\s*\d+)\b")
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=2000, # TODO : comment definir (dans ce cas nb caracteres et pas tokens) ? Avant 1000, mieux (au max 2196 car ajout du titre) ?
    chunk_overlap=200, # TODO : comment definir (dans ce cas nb caracteres et pas tokens) ?
    separators=[
        ARTICLE_PATTERN,
        "\n\n",
        "\n",
        ".",
        " ",
        "",
    ],
    #length_function=count_nb_tokens, # TODO : si necessaire de compter en tokens alors modifier ici et modifier size + overlap
    is_separator_regex=True,
    keep_separator=True,
    add_start_index=True,
)

# database
def get_db_dense_index_config():
    return {
        "dense": models.VectorParams(
            size=int(os.getenv("MODEL_EMBEDDINGS_DIM_VECTOR")),
            distance=models.Distance.COSINE
        )
    }
DB_SPARSE_INDEX_CONFIG_FR = {"sparse_fr": models.SparseVectorParams(modifier=models.Modifier.IDF)}
DB_SPARSE_INDEX_CONFIG_EN = {"sparse_en": models.SparseVectorParams(modifier=models.Modifier.IDF)}
DB_SPARSE_INDEX_CONFIG = {
    **DB_SPARSE_INDEX_CONFIG_FR,
    **DB_SPARSE_INDEX_CONFIG_EN,
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
    return OpenAIEmbeddings(
        model=os.getenv("MODEL_EMBEDDINGS_NAME"),
        base_url=os.getenv("MODELS_BASE_URL"),
        api_key=os.getenv("MODEL_EMBEDDINGS_API_KEY")
    )
def get_sparse_model_config_fr():
    return FastEmbedSparse(model_name=os.getenv("MODEL_SPARSE_NAME"), avg_len=float(os.getenv("AVG_LEN_FR")), language="french")
def get_sparse_model_config_en():
    return FastEmbedSparse(model_name=os.getenv("MODEL_SPARSE_NAME"), avg_len=float(os.getenv("AVG_LEN_EN")), language="english")

# retrieval
NB_CHUNKS_RETRIEVED = 100
NB_CHUNKS_RERANKED = 20

# generation
MAX_USER_MESSAGE_LEN = 1500
NB_CHUNKS_SENT = 5
def get_llm_model_config():
    return OpenAI(
        model=os.getenv("MODEL_LLM_NAME"),
        base_url=os.getenv("MODELS_BASE_URL"),
        api_key=os.getenv("MODEL_LLM_API_KEY"),
        max_tokens=500, # TODO : a augmenter ?
        temperature=0.0
    )
RELEVANCE_THRESHOLD = 0.2

# evaluation
EVALUATION_DATASET_NAME = "20250520_clean_dev_dataset"
COLS_ORDER_IN_EVALUATION_DF = [
    "trace_id",
    "question",
    # retrieval
    "hit_at_1",
    "hit_at_2",
    "hit_at_3",
    "hit_at_4",
    "hit_at_5",
    "hit_at_10",
    "hit_at_15",
    "hit_at_20",
    "Context Relevance (Contextrelevance-Langfuse)", # evaluate if useful context retrieved
    "mrr_doc",
    "ratio_correct_docs",
    # generation (based on ground truth)
    "semantic_similarity",
    "len_answers_quality",
    "Answer Correctness - RAGAS",
    "chrf_score",
    # generation (LLM feeling)
    "Groundedness (Faithfulness-RAGAS)", # evaluate if answer based on context retrieved
    "Answer Relevance (Relevance-Langfuse)" # evaluate if answer is relevant to the question
]
DICT_METRIC_LABELS = {
    "hit_at_1": "Recall@1",
    "hit_at_2": "Recall@2",
    "hit_at_3": "Recall@3",
    "hit_at_4": "Recall@4",
    "hit_at_5": "Recall@5",
    "hit_at_10": "Recall@10",
    "hit_at_15": "Recall@15",
    "hit_at_20": "Recall@20",
    "Context Relevance (Contextrelevance-Langfuse)": "Context relevance",
    "mrr_doc": "Reciprocal rank",
    "ratio_correct_docs": "Precision",
    "semantic_similarity": "Semantic similarity",
    "len_answers_quality": "Answer lengths comparison",
    "Answer Correctness - RAGAS": "Answer correctness",
    "chrf_score": "Score chrF",
    "Groundedness (Faithfulness-RAGAS)": "Groundedness",
    "Answer Relevance (Relevance-Langfuse)": "Answer relevance"
}
