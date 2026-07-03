import os
import logging
import argparse

from polylex_chatbot.env import load_project_env
from polylex_chatbot.config import STATS_PATH, DATA_PATH
from polylex_chatbot.metadata import load_metadata
from polylex_chatbot.stats import compute_corpus_metadata_stats, compute_corpus_content_stats, save_stats, compute_content_lengths, compute_and_save_nb_occ_article_plot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

def compute_stats(corpus_dir, metadata_dir):
    logger.info("Start to compute stats on corpus")

    logger.info("Reading metadata from %s", metadata_dir)
    metadata = load_metadata(metadata_dir)
    logger.info("Number of documents inside the corpus: %s", len(metadata))

    logger.info("Compute stats on corpus metadata")
    corpus_metadata_stats = compute_corpus_metadata_stats(metadata)
    corpus_metadata_stats_path = save_stats(metadata_dir, "stats_corpus_metadata", "json", corpus_metadata_stats)
    logger.info("Stats on corpus metadata saved at %s", corpus_metadata_stats_path)

    logger.info("Compute stats on corpus content")
    corpus_content_stats = compute_corpus_content_stats(corpus_dir)
    corpus_content_stats_path = save_stats(metadata_dir, "stats_corpus_content", "csv", corpus_content_stats)
    content_lengths_stats = compute_content_lengths(corpus_content_stats)
    content_lengths_stats_path = save_stats(metadata_dir, "stats_content_lengths", "csv", content_lengths_stats)
    logger.info("Stats on corpus content saved at %s and %s", corpus_content_stats_path, content_lengths_stats_path)

    logger.info("Compute and save plot showing distribution of article pattern in documents")
    compute_and_save_nb_occ_article_plot(metadata_dir, corpus_content_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute stats on corpus metadata and corpus content")
    parser.add_argument("--env-path", required=True, help="Path to the environment file")
    parser.add_argument("--corpus-dir", help="Directory where documents are stored", default=None)
    parser.add_argument("--metadata-dir", help="Directory where metadata are stored", default=None)
    args = parser.parse_args()

    env_file = load_project_env(args.env_path)

    corpus_name = os.getenv("CORPUS_NAME")
    corpus_dir = args.corpus_dir or DATA_PATH / corpus_name
    metadata_dir = args.metadata_dir or STATS_PATH / corpus_name

    compute_stats(
        corpus_dir=corpus_dir,
        metadata_dir=metadata_dir
    )
