import logging

from polylex_chatbot.env import load_project_env
env_path = load_project_env()

from polylex_chatbot.metadata import build_metadata, add_indexing_flag, save_metadata
from polylex_chatbot.downloads import download_documents
from polylex_chatbot.polylex import fetch_polylex_api
from polylex_chatbot.config import DATA_PATH, STATS_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

def build_corpus(data_dir, metadata_dir):
    logger.info("Start to build corpus")

    logger.info("Collecting data")
    data = fetch_polylex_api()

    logger.info("Creating metadata dict")
    metadata = build_metadata(data)

    logger.info("Downloading documents to %s", data_dir)
    download_documents(metadata, data_dir)

    logger.info("Compute document statistics to determine which documents should be indexed")
    metadata = add_indexing_flag(metadata, data_dir)

    logger.info("Writing metadata to %s", metadata_dir)
    save_metadata(metadata, metadata_dir)

    logger.info("Corpus built successfully")

if __name__ == "__main__":
    build_corpus(
        data_dir=DATA_PATH,
        metadata_dir=STATS_PATH,
    )
