from experiment.generation import generate_response
from experiment.retrieval import retrieve_documents

def make_rag_task(db, reranker_model_name, reranker_api_key, base_url, k, generation_top_k):
    def rag_task(*, item, **kwargs):
        query = item.input.get("query")

        retrieval_result = retrieve_documents(db, query, reranker_model_name, reranker_api_key, base_url, k)

        context_for_llm = retrieval_result.get("retrieved_contexts")[:generation_top_k]
        generated_response = generate_response(query, context_for_llm)

        return {
            **retrieval_result,
            "used_contexts": context_for_llm,
            "generated_response": generated_response
        }

    return rag_task
