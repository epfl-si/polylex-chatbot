import logging

from polylex_chatbot.env import load_project_env
env_path = load_project_env()

from polylex_chatbot.config import STATS_PATH, DATA_PATH
from polylex_chatbot.metadata import load_metadata
from polylex_chatbot.stats import compute_corpus_metadata_stats, compute_corpus_content_stats, save_stats, compute_content_lengths, compute_and_save_nb_occ_article_plot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Start to compute stats on corpus")

    logger.info("Reading metadata")
    metadata = load_metadata(STATS_PATH, only_indexed_documents=True)
    logger.info("Number of documents inside the corpus: %s", len(metadata))

    logger.info("Compute stats on corpus metadata")
    corpus_metadata_stats = compute_corpus_metadata_stats(metadata)
    corpus_metadata_stats_path = save_stats(STATS_PATH, "corpus_metadata", "json", corpus_metadata_stats)
    logger.info("Stats on corpus metadata saved at %s", corpus_metadata_stats_path)

    logger.info("Compute stats on corpus content")
    corpus_content_stats = compute_corpus_content_stats(DATA_PATH)
    corpus_content_stats_path = save_stats(STATS_PATH, "corpus_content", "csv", corpus_content_stats)
    content_lengths_stats = compute_content_lengths(corpus_content_stats)
    content_lengths_stats_path = save_stats(STATS_PATH, "content_lengths_stats", "csv", content_lengths_stats)
    logger.info("Stats on corpus content saved at %s and %s", corpus_content_stats_path, content_lengths_stats_path)

    logger.info("Compute and save plot showing distribution of article pattern in documents")
    compute_and_save_nb_occ_article_plot(STATS_PATH, corpus_content_stats)
