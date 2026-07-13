from .config import prepare_llm_context
from .constants import TEXTUAL_CONTENTS_PATH_RAG
from .retrieval import retrieve_documents
from .generation import generate_response

def make_rag_task(db, nb_chunks_retrieved, reranker_model_name, reranker_api_key, base_url, nb_chunks_reranked, llm_model_config, prompt):
    def rag_task(*, item, **kwargs):
        query = item.input.get("query")
        doc_ids, scores, chunks = retrieve_documents(db, query, reranker_model_name, reranker_api_key, base_url, nb_chunks_retrieved, nb_chunks_reranked)
        context_for_llm, items_in_llm_context, nb_relevant_chunks, nb_max_items = prepare_llm_context(chunks, scores, TEXTUAL_CONTENTS_PATH_RAG)
        answer, _, _ = generate_response(llm_model_config, query, prompt, context_for_llm)
        return {
            "retrieved_doc_ids": doc_ids,
            "retrieved_scores": scores,
            "retrieved_contexts": chunks,
            "nb_max_items": nb_max_items,
            "context_for_llm": context_for_llm,
            "chunks_in_llm_context": items_in_llm_context,
            "nb_relevant_chunks": nb_relevant_chunks,
            "generated_response": answer
        }
    return rag_task
