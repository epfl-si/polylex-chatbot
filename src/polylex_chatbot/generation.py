from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_context_for_llm(chunks):
    if len(chunks) == 0:
        return "[]"

    contents = [chunk.get("content") for chunk in chunks]
    context = "\n\n".join(f"[Chunk {i}]\n{content}" for i, content in enumerate(contents, start=1))

    return context

def prepare_llm_context(chunks, scores, nb_chunks_max, relevance_threshold):
    chunks_in_llm_context = []

    for i, score in enumerate(scores):
        if score >= relevance_threshold and i < nb_chunks_max:
            chunks_in_llm_context.append(chunks[i])

    nb_relevant_chunks = len(chunks_in_llm_context)
    context_for_llm = build_context_for_llm(chunks_in_llm_context)

    return context_for_llm, chunks_in_llm_context, nb_relevant_chunks, nb_chunks_max

def generate_response(llm, query, prompt, context, monitoring_config=None):
    prompt_template = ChatPromptTemplate.from_template(prompt)
    chain = prompt_template | llm | StrOutputParser()

    inputs = {
        "query": query,
        "context_text": context
    }

    if monitoring_config is None:
        response = chain.invoke(inputs).strip()
    else:
        response = chain.invoke(inputs, config={"callbacks": [monitoring_config]}).strip()

    return response

__all__=[generate_response, prepare_llm_context]
