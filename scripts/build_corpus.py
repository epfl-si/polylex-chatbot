import logging
import argparse
from datetime import datetime

from polylex_chatbot.env import load_project_env
env_path = load_project_env()

from polylex_chatbot.metadata import build_metadata, add_indexing_flag, save_metadata
from polylex_chatbot.downloads import fetch_polylex_api, download_documents
from polylex_chatbot.config import DATA_PATH, STATS_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

def build_corpus(data_dir, metadata_dir, corpus_name):
    logger.info("Start to build corpus")

    logger.info("Collecting data")
    data = fetch_polylex_api()

    logger.info("Creating metadata dict")
    metadata = build_metadata(data)

    logger.info("Downloading documents...")
    corpus_dir = download_documents(metadata, data_dir, corpus_name)
    logger.info("Documents downloaded to %s", corpus_dir)

    logger.info("Compute document statistics to determine which documents should be indexed")
    metadata = add_indexing_flag(metadata, corpus_dir)

    corpus_metadata_dir = save_metadata(metadata, metadata_dir, corpus_name)
    logger.info("Writing metadata to %s", corpus_metadata_dir)

    logger.info("Corpus built successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build corpus")
    parser.add_argument(
        "--corpus-name",
        default=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_corpus",
        help="Name of the corpus to be created"
    )

    args = parser.parse_args()

    build_corpus(
        data_dir=DATA_PATH,
        metadata_dir=STATS_PATH,
        corpus_name=args.corpus_name
    )
