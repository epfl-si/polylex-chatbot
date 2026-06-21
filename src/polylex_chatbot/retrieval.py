import requests
from qdrant_client.http.models import SearchParams

# TODO : top_n doit etre une constante qq part
# source : https://gitlab.epfl.ch/rcp/aiaas/client-usage-examples/-/blob/main/requests-reranker.py
def rerank_documents(api_key, base_url, query, documents, model, top_n=20):
    """
    Reranks documents based on their relevance to a given query.

    Args:
        api_key (str): Your API key
        base_url (str): The base URL for the reranking API
        query (str): The query to rerank documents for
        documents (list[str]): A list of documents to rerank
        model (str): The model to use for reranking.
        top_n (int, optional): The number of top documents to return. Defaults to 3.

    Returns:
        list[str]: The reranked documents
    """
    url = f"{base_url}/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    ranked_documents = sorted(result["results"], key=lambda x: x["relevance_score"], reverse=True)

    return ranked_documents

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
        "retrieved_contexts": retrieved_contexts
    }
