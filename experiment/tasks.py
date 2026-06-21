from polylex_chatbot.retrieval import retrieve_documents
from polylex_chatbot.generation import generate_response

def make_retrieval_task(db, reranker_model_name, reranker_api_key, base_url, k):
    def retrieval_task(*, item, **kwargs):
        query = item.input.get("query")
        return retrieve_documents(db, query, reranker_model_name, reranker_api_key, base_url, k)
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

def make_rag_task(db, reranker_model_name, reranker_api_key, base_url, k, llm_model_config, generation_top_k, prompt):
    def rag_task(*, item, **kwargs):
        query = item.input.get("query")
        retrieval_result = retrieve_documents(db, query, reranker_model_name, reranker_api_key, base_url, k)
        context_for_llm = retrieval_result.get("retrieved_contexts")[:generation_top_k]
        answer = generate_response(llm_model_config, query, prompt, context_for_llm)
        return {
            **retrieval_result,
            "used_contexts": context_for_llm,
            "generated_response": answer
        }
    return rag_task
