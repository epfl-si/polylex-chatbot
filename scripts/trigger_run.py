# TODO : tester retrieval_task et generation_task separement

import logging
import argparse
from datetime import datetime
from langfuse import get_client

from polylex_chatbot.env import load_project_env
env_path = load_project_env()

from polylex_chatbot.config import (EVALUATION_DATASET_NAME, init_db_client, NB_CHUNKS_RETRIEVED, NB_CHUNKS_RERANKED,
                                    LLM_MODEL_CONFIG, NB_CHUNKS_SENT, PROMPT_TEMPLATE_FR)
from polylex_chatbot.tasks import make_rag_task
from polylex_chatbot.evaluators import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TODO")
    # TODO : a voir mais pour lancer script argument qui indique quel .env lire pour permettre d'avoir toutes les bonnes variables d'environnement (a creer manuellement avant de run)
    # TODO : si c'est le cas alors ne pas importer par defaut .env dans les imports mais importer ensuite ici
    parser.add_argument("env-path", help="Path to the env file to use")
    parser.add_argument("run-description", help="Description of the run")
    parser.add_argument("--dataset-name", default=EVALUATION_DATASET_NAME)
    args = parser.parse_args()

    langfuse = get_client()
    dataset = langfuse.get_dataset(args.dataset_name)

    logger.info("Collecting metadata to describe the run")
    collection_name = os.getenv("DB_COLLECTION_NAME")
    llm_name = os.getenv("MODEL_LLM_NAME")
    run_name = f"{collection_name}_{llm_name}_{datetime.now().isoformat()}"

    logger.info("Create db client")
    qdrant = init_db_client("fr")

    logger.info("Running experiment...")
    rag_result = dataset.run_experiment(
        name=run_name,
        description=f"{args.run_description} (using '{collection_name}' collection and '{llm_name}' llm)",
        task=make_rag_task(qdrant, NB_CHUNKS_RETRIEVED, os.getenv("MODEL_RERANKER_NAME"),
                           os.getenv("MODEL_RERANKER_API_KEY"), os.getenv("MODELS_BASE_URL"), NB_CHUNKS_RERANKED,
                           LLM_MODEL_CONFIG, NB_CHUNKS_SENT, PROMPT_TEMPLATE_FR),
        evaluators=[
            # retrieval
            make_hit_at_x_evaluator(1),
            make_hit_at_x_evaluator(2),
            make_hit_at_x_evaluator(3),
            make_hit_at_x_evaluator(4),
            make_hit_at_x_evaluator(5),
            make_hit_at_x_evaluator(10),
            make_hit_at_x_evaluator(15),
            make_hit_at_x_evaluator(20),
            mrr_doc_evaluator,
            ratio_correct_docs_evaluator,
            # generation
            chrf_evaluator,
            len_answers_quality_evaluator,
            semantic_similarity_evaluator
        ],
        metadata={
            "collection_name": collection_name,
            "llm_name": llm_name,
            "nb_chunks_retrieved": NB_CHUNKS_RETRIEVED,
            "nb_chunks_reranked": NB_CHUNKS_RERANKED,
            "nb_chunks_sent": NB_CHUNKS_SENT
        }
    )

    logger.info("Experiment completed!\nResults:\n%s", rag_result.format())
