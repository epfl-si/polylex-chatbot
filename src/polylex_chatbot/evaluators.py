import os
from openai import AsyncOpenAI
from langfuse import Evaluation
from ragas.metrics.collections import CHRFScore
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import SemanticSimilarity

def make_hit_at_x_evaluator(x):
    def hit_at_x_evaluator(*, output, metadata, **kwargs):
        expected_doc_id = metadata.get("expected_doc_id")
        retrieved_doc_ids_top_k = output.get("retrieved_doc_ids")[:x]
        retrieved_scores_top_k = output.get("retrieved_scores")[:x]
        value = 1.0 if expected_doc_id in retrieved_doc_ids_top_k else 0.0
        return Evaluation(
            name=f"hit_at_{x}",
            value=value,
            comment=f"expected_doc_id={expected_doc_id}, top{x}={retrieved_doc_ids_top_k} with score {retrieved_scores_top_k}"
        )
    return hit_at_x_evaluator

def mrr_doc_evaluator(*, output, metadata, **kwargs):
    expected_doc_id = metadata.get("expected_doc_id")
    retrieved_doc_ids = output.get("retrieved_doc_ids")
    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == expected_doc_id:
            reciprocal_rank = 1.0 / rank
            break
    return Evaluation(
        name="mrr_doc",
        value=reciprocal_rank,
        comment=f"expected_doc_id={expected_doc_id} and retrieved={retrieved_doc_ids}",
    )

def ratio_correct_docs_evaluator(*, output, metadata, **kwargs):
    expected_doc_id = metadata.get("expected_doc_id")
    retrieved_doc_ids = output.get("retrieved_doc_ids")
    nb_correct_docs = 0
    nb_total_docs = len(retrieved_doc_ids)
    for retrieved_doc_id in retrieved_doc_ids:
        if retrieved_doc_id == expected_doc_id:
            nb_correct_docs += 1
    ratio_correct_docs = nb_correct_docs / nb_total_docs
    return Evaluation(
        name="ratio_correct_docs",
        value=ratio_correct_docs,
        comment=f"expected_doc_id={expected_doc_id} and retrieved={retrieved_doc_ids}",
    )

async def chrf_evaluator(*, output, expected_output, **kwargs):
    reference = expected_output.get("answer")
    response = output.get("generated_response")
    score = await CHRFScore().ascore(reference=reference, response=response)
    return Evaluation(
        name="chrf_score",
        value=score
    )

def len_answers_quality_evaluator(*, output, expected_output, **kwargs):
    ground_truth = expected_output.get("answer")
    len_ground_truth = len(ground_truth)
    generated_response = output.get("generated_response")
    len_generated_response = len(generated_response)
    # 0 if len_generated_response << len_ground_truth, 1 if len_generated_response == len_ground_truth and 0 if len_generated_response >> len_ground_truth
    len_answers_quality = min(len_ground_truth, len_generated_response) / max(len_ground_truth, len_generated_response)
    return Evaluation(
        name="len_answers_quality",
        value=len_answers_quality
    )

async def semantic_similarity_evaluator(*, output, expected_output, **kwargs):
    client = AsyncOpenAI(api_key=os.getenv("MODEL_EMBEDDINGS_JUDGE_API_KEY"), base_url=os.getenv("MODELS_BASE_URL"))
    embeddings = embedding_factory(
        model=os.getenv("MODEL_EMBEDDINGS_JUDGE_NAME"),
        client=client
    )
    reference = expected_output.get("answer")
    response = output.get("generated_response")
    score  = await SemanticSimilarity(embeddings=embeddings).ascore(reference=reference, response=response)
    return Evaluation(
        name="semantic_similarity",
        value=score
    )
