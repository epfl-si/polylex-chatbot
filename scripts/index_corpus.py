import logging

from polylex_chatbot.env import load_project_env
env_path = load_project_env()

from polylex_chatbot.indexing import index_chunks
from polylex_chatbot.config import DATA_PATH, STATS_PATH, CHUNKS_TXT_PATH, LANGUAGES, DB_COLLECTION_NAME
from polylex_chatbot.metadata import load_metadata, build_language_matched_metadata_by_doc_id
from polylex_chatbot.chunking import create_chunks, save_chunks, divide_chunks_per_lang

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

def index_corpus(data_dir, metadata_dir, chunks_log_path, collection_name):
    logger.info("Start to index corpus")

    logger.info("Reading metadata and building lookup tables on it")
    metadata = load_metadata(metadata_dir)
    language_matched_metadata_by_doc_id = build_language_matched_metadata_by_doc_id(metadata)

    logger.info("Creating chunks and saving it to %s (human-readable format)", chunks_log_path)
    chunks = create_chunks(data_dir, language_matched_metadata_by_doc_id)
    save_chunks(chunks_log_path, chunks)

    logger.info("Computing average chunk length per language for BM25 indices")
    # FIXME : pas utilise car BM25_lang utilise indexe chacun tous les chunks, ok ou ko ?
    chunks_splitted_by_lang = divide_chunks_per_lang(chunks, LANGUAGES)

    logger.info("Indexing chunks in %s collection", collection_name)
    index_chunks(chunks)

    logger.info("Corpus indexed successfully")

if __name__ == "__main__":
    index_corpus(
        data_dir=DATA_PATH,
        metadata_dir=STATS_PATH,
        chunks_log_path=CHUNKS_TXT_PATH,
        collection_name=DB_COLLECTION_NAME
    )
