from reranker import rerank_documents
from qdrant_client.http.models import SearchParams

def retrieve_documents(db, query, reranker_model_name, reranker_api_key, base_url, k):
    hits = db.similarity_search_with_score(query, k=k, search_params=SearchParams(exact=True))

    documents = [document.page_content for document, _ in hits]

    reranked_hits = rerank_documents(reranker_api_key, base_url, query, documents, reranker_model_name)

    retrieved_doc_ids = []
    reranked_scores = []
    retrieved_contexts = []

    for rank, hit in enumerate(reranked_hits, start=1):
        original_index = hit["index"]
        rerank_score = hit["relevance_score"]

        document, _ = hits[original_index]
        doc_id = document.metadata.get("doc_id")

        retrieved_doc_ids.append(doc_id)
        reranked_scores.append(rerank_score)

        retrieved_contexts.append(
            {
                "rank": rank,
                "content": document.page_content,
                "metadata": document.metadata
            }
        )

    return {
        "retrieved_doc_ids": retrieved_doc_ids,
        "retrieved_scores": reranked_scores,
        "retrieved_contexts": retrieved_contexts,
    }

def make_retrieval_task(db, reranker_model_name, reranker_api_key, base_url, k):
    def retrieval_task(*, item, **kwargs):
        query = item.input.get("query")
        return retrieve_documents(db, query, reranker_model_name, reranker_api_key, base_url, k)
    return retrieval_task

__all__ = ["retrieve_documents", "make_retrieval_task"]
