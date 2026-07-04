import logging
import argparse
import pandas as pd
from datetime import datetime
from langfuse import Langfuse
from dataclasses import dataclass

from polylex_chatbot.env import load_project_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

@dataclass
class DatasetItem:
    question: str
    answer: str
    doc_id: str
    paragraph: str

def create_datasets_in_langfuse(langfuse_client, dev_dataset_name, test_dataset_name):
    date = datetime.now().strftime("%Y-%m-%d")
    langfuse_client.create_dataset(
        name=dev_dataset_name,
        description="Dataset containg dev questions",
        metadata={
            "date": date,
            "type": "dev"
        }
    )

    langfuse_client.create_dataset(
        name=test_dataset_name,
        description="Dataset containg test questions",
        metadata={
            "date": date,
            "type": "test"
        }
    )

def add_item_in_dataset(langfuse_client, dataset_name, item):
    langfuse_client.create_dataset_item(
            dataset_name=dataset_name,
            input={
                "query": item.question
            },
            expected_output={
                "answer": item.answer
            },
            metadata={
                "expected_doc_id": item.doc_id,
                "expected_paragraph": item.paragraph
            }
        )

def populate_langfuse_datasets(langfuse_client, dataset, dev_dataset_name, test_dataset_name):
    for _, row in dataset.iterrows():
        item = DatasetItem(
            question=row["Question"],
            answer=row["Réponse"],
            doc_id=row["Id document"],
            paragraph=(row["Article / Paragraphe"] if pd.notna(row["Article / Paragraphe"]) else "")
        )
        if row['Dataset'] == "dev":
            add_item_in_dataset(langfuse_client, dev_dataset_name, item)
        elif row['Dataset'] == "test":
            add_item_in_dataset(langfuse_client, test_dataset_name, item)

def create_langfuse_datasets(dataset_path, dev_dataset_name, test_dataset_name):
    logger.info("Init Langfuse client...")
    langfuse_client = Langfuse()

    logger.info("Read dataset...")
    dataset = pd.read_csv(dataset_path, header=0)

    logger.info("Create datasets in Langfuse...")
    create_datasets_in_langfuse(langfuse_client, dev_dataset_name, test_dataset_name)

    logger.info("Populate created datasets...")
    populate_langfuse_datasets(langfuse_client, dataset, dev_dataset_name, test_dataset_name)

    logger.info("Datasets created")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create datasets in Langfuse for evaluation")
    parser.add_argument("--env-path", required=True, help="Path to the environment file")
    parser.add_argument("--dataset-path", required=True, help="Path where dataset is stored (csv)")
    parser.add_argument("--dev-dataset-name", required=True, help="Name of the dev dataset to create in Langfuse")
    parser.add_argument("--test-dataset-name", required=True, help="Name of the test dataset to create in Langfuse")
    args = parser.parse_args()

    env_file = load_project_env()

    create_langfuse_datasets(
        dataset_path=args.dataset_path,
        dev_dataset_name=args.dev_dataset_name,
        test_dataset_name=args.test_dataset_name
    )
