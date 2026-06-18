from langfuse import Evaluation
from ragas.metrics.collections import CHRFScore

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

def nb_correct_doc_evaluator(*, output, metadata, **kwargs):
    expected_doc_id = metadata.get("expected_doc_id")
    retrieved_doc_ids = output.get("retrieved_doc_ids")
    nb_correct_docs = 0
    for retrieved_doc_id in retrieved_doc_ids:
        if retrieved_doc_id == expected_doc_id:
            nb_correct_docs += 1
    return Evaluation(
        name="nb_correct_doc",
        value=nb_correct_docs,
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

def len_ratio_answers_evaluator(*, output, expected_output, **kwargs):
    ground_truth = expected_output.get("answer")
    generated_response = output.get("generated_response")
    len_ratio = len(ground_truth) / len(generated_response)
    return Evaluation(
        name="len_ratio_answers",
        value=len_ratio
    )
