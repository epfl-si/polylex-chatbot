import os
import logging
import argparse
from datetime import datetime

from polylex_chatbot.env import load_project_env
from polylex_chatbot.indexing import index_chunks
from polylex_chatbot.constants import TEXTUAL_CONTENTS_PATH_RAG
from polylex_chatbot.config import STATS_PATH, CHUNKS_PATH, LANGUAGES, create_documents_splitter
from polylex_chatbot.metadata import load_metadata, build_language_matched_metadata_by_doc_id
from polylex_chatbot.chunking import create_chunks, save_chunks, divide_chunks_per_lang

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

def index_corpus(textual_contents_dir, metadata_dir, chunks_log_path, collection_name, collection_description, env_file):
    logger.info("Start to index corpus")

    logger.info("Reading metadata from %s and building lookup tables on it", metadata_dir)
    metadata = load_metadata(metadata_dir)
    language_matched_metadata_by_doc_id = build_language_matched_metadata_by_doc_id(metadata)

    logger.info("Creating chunks...")
    split_document_function = create_documents_splitter()
    chunks = create_chunks(textual_contents_dir, language_matched_metadata_by_doc_id, split_document_function)

    logger.info("Computing average chunk length per language for BM25 indices")
    _ = divide_chunks_per_lang(chunks, LANGUAGES, env_file)

    logger.info("Indexing chunks in %s collection", collection_name)
    index_chunks(chunks, collection_name, collection_description, env_file)

    chunks_filename = save_chunks(chunks_log_path, chunks)
    logger.info("Chunks saved in a human-readable format to %s and save in plot chunks distribution", chunks_filename)

    logger.info("Corpus indexed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index corpus")
    parser.add_argument("--env-path", required=True, help="Path to the environment file")
    parser.add_argument("--collection-description", help="Description of the collection to be created (corpus name, chunking strategy, contextualisation strategy, embedding model, ...)", default="")
    parser.add_argument("--corpus-dir", help="Directory where textual contents from documents are stored", default=None)
    parser.add_argument("--metadata-dir", help="Directory where metadata are stored", default=None)
    parser.add_argument("--collection-name", help="Name of the collection to create", default=None)
    args = parser.parse_args()

    env_file = load_project_env(args.env_path)

    corpus_name = os.getenv("CORPUS_NAME")
    textual_contents_dir = args.corpus_dir or TEXTUAL_CONTENTS_PATH_RAG / corpus_name
    metadata_dir = args.metadata_dir or STATS_PATH / corpus_name
    collection_name = args.collection_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_collection"

    index_corpus(
        textual_contents_dir=textual_contents_dir,
        metadata_dir=metadata_dir,
        chunks_log_path=CHUNKS_PATH,
        collection_name=collection_name,
        collection_description=args.collection_description,
        env_file=env_file
    )
