import re
import os
from pathlib import Path
from qdrant_client import models
from langchain_qdrant import FastEmbedSparse
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# paths
DATA_PATH = Path.cwd() / "data"
STATS_PATH = Path.cwd() / "stats"
CHUNKS_TXT_PATH = Path.cwd() / "chunks.txt"

# api
LEXES_API_URL = "https://polylex-admin.epfl.ch/api/v1/lexes?isAbrogated=0"

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
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/5.7.2_dir_placement_all.pdf": "fr" # FIXME : fr + de, grave ?
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
DB_DENSE_INDEX_CONFIG = {
    "dense": models.VectorParams(
        size=int(os.getenv("MODEL_EMBEDDING_DIM_VECTOR")),
        distance=models.Distance.COSINE
        )
    }
DB_SPARSE_INDEX_CONFIG_FR = {"sparse_fr": models.SparseVectorParams(modifier=models.Modifier.IDF)}
DB_SPARSE_INDEX_CONFIG_EN = {"sparse_en": models.SparseVectorParams(modifier=models.Modifier.IDF)}
DB_SPARSE_INDEX_CONFIG = {
    **DB_SPARSE_INDEX_CONFIG_FR,
    **DB_SPARSE_INDEX_CONFIG_EN,
}

# embeddings (sparse + dense)
EMBEDDING_MODEL_CONFIG = OpenAIEmbeddings(
    model=os.getenv("MODEL_EMBEDDINGS_NAME"),
    base_url=os.getenv("MODELS_BASE_URL"),
    api_key=os.getenv("MODEL_EMBEDDINGS_API_KEY")
)
def get_sparse_model_config_fr():
    return FastEmbedSparse(model_name=os.getenv("MODEL_SPARSE_NAME"), avg_len=float(os.getenv("AVG_LEN_FR")), language="french")
def get_sparse_model_config_en():
    return FastEmbedSparse(model_name=os.getenv("MODEL_SPARSE_NAME"), avg_len=float(os.getenv("AVG_LEN_EN")), language="english")

# retrieval
QDRANT_NB_CHUNKS_RETRIEVED = 5

# generation
MAX_USER_MESSAGE_LEN = 1500
LLM_MODEL_CONFIG = OpenAI(
    model=os.getenv("MODEL_LLM_NAME"),
    base_url=os.getenv("MODELS_BASE_URL"),
    api_key=os.getenv("MODEL_LLM_API_KEY"),
    max_tokens=500,
    temperature=0.0
)

# TODO : dire que si question hors contexte alors ne pas prendre en compte + répondre dans la langue de l'utilisateur
# TODO : ajouter -> si contexte ne contient pas l'info alors dire "je sais pas" + concis et factuel

PROMPT_TEMPLATE_FR = """Réponds à la question en utilisant uniquement le contexte fourni.

Contexte:
{context_text}

Question:
{query}

Réponse:"""
PROMPT_TEMPLATE_EN = """Answer the question using only the provided context.

Context:
{context_text}

Question:
{query}

Answer:"""
