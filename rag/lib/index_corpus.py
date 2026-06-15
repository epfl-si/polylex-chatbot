import os
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from rag.lib.indexing import index_chunks
from rag.lib.config import DATA_PATH, STATS_PATH, CHUNKS_TXT_PATH, LANGUAGES, ENV_PATH
from rag.lib.metadata import load_metadata, build_metadata_lookup_tables
from rag.lib.chunking import create_chunks, save_chunks, divide_chunks_per_lang

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Index documents from data-dir using metadata from metadata-dir."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_PATH,
        help="Directory containing the documents to index.",
    )

    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=STATS_PATH,
        help="Directory containing the metadata",
    )

    parser.add_argument(
        "--chunks-log-path",
        type=Path,
        default=CHUNKS_TXT_PATH,
        help="Path where the human-readable chunks will be written.",
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default=os.getenv("COLLECTION_NAME"), # TODO : f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_collection",
        help="Name of the collection to create.",
    )

    return parser.parse_args()

def index_corpus(data_dir, metadata_dir, chunks_log_path, collection_name):
    logger.info("Start to index corpus")

    logger.info("Reading metadata and building lookup tables on it")
    metadata = load_metadata(metadata_dir)
    doc_id_to_metadata, metadata_to_title = build_metadata_lookup_tables(metadata)

    logger.info("Creating chunks and saving it to %s (human-readable format)", chunks_log_path)
    chunks = create_chunks(data_dir, doc_id_to_metadata, metadata_to_title)
    save_chunks(chunks_log_path, chunks)

    logger.info("Computing average chunk length per language for BM25 indices")
    # FIXME : pas utilise car BM25_lang utilise indexe chacun tous les chunks, ok ou ko ?
    chunks_splitted_by_lang = divide_chunks_per_lang(chunks, LANGUAGES, ENV_PATH)

    logger.info("Indexing chunks in %s collection", collection_name)
    index_chunks(chunks, collection_name)

    logger.info("Corpus indexed successfully")

if __name__ == "__main__":
    args = parse_args()

    index_corpus(
        data_dir=args.data_dir,
        metadata_dir=args.metadata_dir,
        chunks_log_path=args.chunks_log_path,
        collection_name=args.collection_name
    )
