from pathlib import Path
from qdrant_client import models

# paths

DOCUMENTS_PATH = Path.cwd() / "documents"
STATS_PATH = Path.cwd() / "stats"
CHUNKS_PATH = Path.cwd() / "stats"
EVALUATION_RESULTS_PATH = Path.cwd() / "evaluations"
COMPARISONS_RESULTS_PATH = Path.cwd() / "comparisons"
TEXTUAL_CONTENTS_PATH_RAG = Path.cwd() / "textual_contents"
TEXTUAL_CONTENTS_PATH_CHATBOT = Path.cwd().parent / "textual_contents"


# api

LEXES_API_URL = "https://polylex-admin.epfl.ch/api/v1/lexes?isAbrogated=0"


# rcp

RCP_MODEL_NOT_LOADED_TIMEOUT_SECONDS = 30


# languages

LANGUAGES = ["fr", "en"]

DICT_MATCH_LANG_FOR_DOC = {
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
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/4.4.1_Regles_applications_fr_an-1.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/4.4.1_Description_fonction_fr_an.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/LEX-5.1.0.3.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/5.7.1_conv_tresorerie_all.pdf": "fr",
    "https://ethrat.ch/wp-content/uploads/2021/09/Immobilienweisung_ETH-Bereich_2016_F_0.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2020/07/LEX-1.1.9.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2021/12/LEX-1.1.12.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2022/03/LEX-5.1.0.4.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2026/01/LEX-1.1.17.pdf": "en",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/5.7.2_dir_placement_all.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2026/05/LEX-4.6.0.1.pdf": "fr"
}

ARTICLE_PATTERN = r"\b(?:Article\s+\d+|Art\.\s*\d+)\b"


# database

DB_SPARSE_INDEX_CONFIG_FR = {"sparse_fr": models.SparseVectorParams(modifier=models.Modifier.IDF)}

DB_SPARSE_INDEX_CONFIG_EN = {"sparse_en": models.SparseVectorParams(modifier=models.Modifier.IDF)}

DB_SPARSE_INDEX_CONFIG = {
    **DB_SPARSE_INDEX_CONFIG_FR,
    **DB_SPARSE_INDEX_CONFIG_EN,
}


# retrieval

NB_CHUNKS_RETRIEVED = 100

NB_CHUNKS_RERANKED = 20


# generation


MAX_USER_MESSAGE_LEN = 1500

NB_MAX_ITEMS_SENT = 10

RELEVANCE_THRESHOLD = 0.2


# evaluation

EVALUATION_DATASET_NAME = "20260704_dev_dataset"

COLS_ORDER_IN_EVALUATION_DF = [
    "trace_id",
    "question",
    "generated_answer",
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
