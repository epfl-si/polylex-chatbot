import os
import copy
from collections import Counter

from .generation import build_context_for_llm
from .constants import TEXTUAL_CONTENTS_PATH_CHATBOT, RELEVANCE_THRESHOLD

def prepare_llm_context_n_chunks(chunks, scores, nb_chunks_sent):
    chunks_in_llm_context = chunks[:nb_chunks_sent]
    nb_relevant_chunks = -1
    context_for_llm = build_context_for_llm(chunks_in_llm_context)
    return context_for_llm, chunks_in_llm_context, nb_relevant_chunks, nb_chunks_sent

def prepare_llm_context_max_n_chunks(chunks, scores, nb_max_chunks_sent):
    chunks_in_llm_context = []

    for i, score in enumerate(scores):
        if score >= RELEVANCE_THRESHOLD and i < nb_max_chunks_sent:
            chunks_in_llm_context.append(chunks[i])

    nb_relevant_chunks = len(chunks_in_llm_context)
    context_for_llm = build_context_for_llm(chunks_in_llm_context)

    return context_for_llm, chunks_in_llm_context, nb_relevant_chunks, nb_max_chunks_sent

def should_send_documents_to_llm(nb_tokens_in_context):
    nb_max_tokens = int(os.getenv("NB_MAX_TOKENS_IN_LLM_CONTEXT"))
    nb_tokens_already_used = int(os.getenv("NB_TOKENS_TO_KEEP_AVAILABLE"))
    nb_tokens_available = nb_max_tokens - nb_tokens_already_used
    return nb_tokens_in_context <= nb_tokens_available

def get_doc_content_from_chunk(chunk):
    filename = f"{chunk.get("metadata").get("filename")}.txt"
    title = chunk.get("content").split("\n\n", 1)[0]
    filepath = TEXTUAL_CONTENTS_PATH_CHATBOT / os.getenv("CORPUS_NAME") / filename
    textual_content = filepath.read_text(encoding="utf-8")
    content = f"{title}\n\n{textual_content}" if title else textual_content
    return content

def prepare_llm_context_n_documents_or_chunks(chunks, scores, nb_items_sent):
    chunks_to_use = chunks[:nb_items_sent]
    nb_relevant_items = -1

    nb_tokens_in_context = 0
    for chunk in chunks_to_use:
        # if chunk is a summary, nb_tokens == 0 but seems fine because small contents
        nb_tokens_in_context += chunk.get("metadata").get("nb_tokens")
    should_send_documents = should_send_documents_to_llm(nb_tokens_in_context)

    if should_send_documents:
        doc_ids_already_in_llm_context = set()
        docs_in_llm_context = []

        for chunk in chunks_to_use:
            metadata = chunk.get("metadata")
            doc_id = metadata.get("doc_id")

            if doc_id in doc_ids_already_in_llm_context:
                print(f"Chunk from same document retrieved, nothing added in LLM context")
                continue

            doc_ids_already_in_llm_context.add(doc_id)
            docs_in_llm_context.append(chunk)
            docs_in_llm_context[-1]["content"] = get_doc_content_from_chunk(chunk)

        context_for_llm = build_context_for_llm(docs_in_llm_context)
        nb_items_really_sent = len(docs_in_llm_context)
        return context_for_llm, docs_in_llm_context, nb_relevant_items, nb_items_really_sent
    else:
        for chunk in chunks_to_use:
            print(f"Context too long if sending '{chunk.get("metadata").get("filename")}' document to LLM")

        context_for_llm = build_context_for_llm(chunks_to_use)

        return context_for_llm, chunks_to_use, nb_relevant_items, nb_items_sent

def prepare_llm_context_modular_context(chunks, scores, nb_items_sent):
    FULL_DOCUMENT_TOKEN_LIMIT = 50000
    APPROX_TOKENS_PER_CHUNK = 2000

    def can_add(nb_tokens):
        return should_send_documents_to_llm(approxim_nb_tokens_in_context + nb_tokens)

    chunks_to_use = chunks[:nb_items_sent]
    scores_to_use = scores[:nb_items_sent]

    approxim_nb_tokens_in_context = 0
    items_in_llm_context = []
    handled_indices = set()

    doc_ids = [chunk.get("metadata").get("doc_id") for chunk in chunks_to_use]
    count_doc_ids = Counter(doc_ids)

    context_full = False
    for doc_id, count in count_doc_ids.most_common():
        if context_full:
            break

        if count == 1:
            print(f"No more chunks retrieved from the same document")
            break

        indices_of_chunks_from_current_doc_id = [i for i, chunk in enumerate(chunks_to_use) if chunk.get("metadata").get("doc_id") == doc_id]
        ref_chunk = chunks_to_use[indices_of_chunks_from_current_doc_id[0]]
        nb_tokens = ref_chunk.get("metadata").get("nb_tokens")

        if nb_tokens <= FULL_DOCUMENT_TOKEN_LIMIT:
            if not can_add(nb_tokens):
                print(f"Document with doc id '{doc_id}' can not be added in context (no more space left) but is referenced {count} times")
                break
            ref_chunk_for_context = copy.deepcopy(ref_chunk)
            items_in_llm_context.append(ref_chunk_for_context)
            items_in_llm_context[-1]["content"] = get_doc_content_from_chunk(ref_chunk_for_context)
            approxim_nb_tokens_in_context += nb_tokens
            handled_indices.update(indices_of_chunks_from_current_doc_id)
            print(f"Document with doc id '{doc_id}' added in context (referenced {count} times)")
        else:
            print(f"Document with doc id '{doc_id}' referenced {count} times but too large to get in context ({nb_tokens})")
            for index in indices_of_chunks_from_current_doc_id:
                chunk = chunks_to_use[index]
                score = scores_to_use[index]
                if score < RELEVANCE_THRESHOLD:
                    print(f"Chunk is not relevant, so not added in context (score: {score})")
                    continue
                if not can_add(APPROX_TOKENS_PER_CHUNK):
                    print(f"Chunk is relevant, but no more space left (score: {score})")
                    context_full = True
                    break
                items_in_llm_context.append(chunk)
                approxim_nb_tokens_in_context += APPROX_TOKENS_PER_CHUNK
                handled_indices.add(index)
                print(f"Chunk is added in context with a score of {score}")

    for index, chunk in enumerate(chunks_to_use):
        score = scores_to_use[index]

        if index in handled_indices:
            continue

        if scores_to_use[index] < RELEVANCE_THRESHOLD:
            print(f"Chunk referenced once is not relevant, so not added in context (score: {score})")
        else:
            if not can_add(APPROX_TOKENS_PER_CHUNK):
                print(f"Chunk referenced once is relevant, but no more space left (score: {score})")
                break

            items_in_llm_context.append(chunk)
            approxim_nb_tokens_in_context += APPROX_TOKENS_PER_CHUNK
            print(f"Chunk referenced once is added in context with a score of {score}")

    context_for_llm = build_context_for_llm(items_in_llm_context)
    nb_relevant_items = sum(1 for index, chunk in enumerate(chunks_to_use) if scores_to_use[index] >= RELEVANCE_THRESHOLD)
    nb_items_really_sent = len(items_in_llm_context)

    print(f"Context successfully built ({approxim_nb_tokens_in_context} tokens in context)")

    return context_for_llm, items_in_llm_context, nb_relevant_items, nb_items_really_sent
