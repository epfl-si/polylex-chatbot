import logging
import argparse
from datetime import datetime

from polylex_chatbot.env import load_project_env
env_path = load_project_env()

from polylex_chatbot.constants import TEXTUAL_CONTENTS_PATH_RAG
from polylex_chatbot.metadata import build_metadata, save_textual_content_and_complete_metadata, save_metadata
from polylex_chatbot.downloads import fetch_polylex_api, download_documents
from polylex_chatbot.config import DOCUMENTS_PATH, STATS_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

def build_corpus(documents_dir, textual_contents_dir, metadata_dir, corpus_name):
    logger.info("Start to build corpus")

    logger.info("Collecting data")
    data = fetch_polylex_api()

    logger.info("Creating metadata dict")
    metadata = build_metadata(data)

    logger.info("Downloading documents to %s ...", documents_dir)
    download_documents(metadata, documents_dir, corpus_name)

    logger.info("Extract and save textual content from documents in %s, and add nb_tokens and indexing flag in metadata dict", textual_contents_dir)
    metadata = save_textual_content_and_complete_metadata(documents_dir, textual_contents_dir, metadata)

    logger.info("Writing metadata to %s ...", metadata_dir)
    save_metadata(metadata, metadata_dir)

    logger.info("Corpus built successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build corpus")
    parser.add_argument(
        "--corpus-name",
        default=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_corpus",
        help="Name of the corpus to be created"
    )

    args = parser.parse_args()
    corpus_name = args.corpus_name

    documents_dir = DOCUMENTS_PATH / corpus_name
    textual_contents_dir = TEXTUAL_CONTENTS_PATH_RAG / corpus_name
    metadata_dir = STATS_PATH / corpus_name

    documents_dir.mkdir(parents=True, exist_ok=True)
    textual_contents_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    build_corpus(
        documents_dir=documents_dir,
        textual_contents_dir=textual_contents_dir,
        metadata_dir=metadata_dir,
        corpus_name=corpus_name
    )
