import logging
import argparse
from pathlib import Path

from rag.lib.metadata import build_metadata, add_indexing_flag, save_metadata
from rag.lib.downloads import download_documents
from rag.lib.polylex import fetch_polylex_api
from rag.lib.config import DATA_PATH, STATS_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the RAG corpus by collecting metadata and downloading documents."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_PATH,
        help="Directory where downloaded documents will be stored.",
    )

    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=STATS_PATH,
        help="Directory where metadata file will be written.",
    )

    return parser.parse_args()

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
    args = parse_args()

    build_corpus(
        data_dir=args.data_dir,
        metadata_dir=args.metadata_dir,
    )
