from .retrieval import retrieve_documents
from .generation import generate_response

def make_retrieval_task(db, nb_chunks_retrieved, reranker_model_name, reranker_api_key, base_url, nb_chunks_reranked):
    def retrieval_task(*, item, **kwargs):
        query = item.input.get("query")
        return retrieve_documents(db, query, reranker_model_name, reranker_api_key, base_url, nb_chunks_retrieved, nb_chunks_reranked)
    return retrieval_task

def make_generation_task(llm_model_config, generation_top_k, prompt):
    def generation_task(*, item, **kwargs):
        query = item.input.get("query")
        contexts = item.input.get("retrieved_contexts")
        context_for_llm = contexts[:generation_top_k]
        answer = generate_response(llm_model_config, query, prompt, context_for_llm)
        return {
            "generated_response": answer,
            "used_contexts": context_for_llm
        }
    return generation_task

def make_rag_task(db, nb_chunks_retrieved, reranker_model_name, reranker_api_key, base_url, nb_chunks_reranked, llm_model_config, nb_chunks_sent, prompt):
    def rag_task(*, item, **kwargs):
        query = item.input.get("query")
        retrieval_result = retrieve_documents(db, query, reranker_model_name, reranker_api_key, base_url, nb_chunks_retrieved, nb_chunks_reranked)
        context_for_llm = retrieval_result.get("retrieved_contexts")[:nb_chunks_sent]
        answer = generate_response(llm_model_config, query, prompt, context_for_llm)
        return {
            **retrieval_result,
            "used_contexts": context_for_llm,
            "generated_response": answer
        }
    return rag_task
