from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_response(llm, query, prompt, context, monitoring_config=None):
    contents = [chunk.get("content") for chunk in context]
    context_text = "\n\n".join(f"[Chunk {i}]\n{content}" for i, content in enumerate(contents, start=1))

    prompt_template = ChatPromptTemplate.from_template(prompt)
    chain = prompt_template | llm | StrOutputParser()

    inputs = {
        "query": query,
        "context_text": context_text
    }

    if monitoring_config is None:
        response = chain.invoke(inputs).strip()
    else:
        response = chain.invoke(inputs, config={"callbacks": [monitoring_config]}).strip()

    return response

def prepare_llm_context(chunks, scores, nb_chunks_max, relevance_threshold):
    context = []
    for i, score in enumerate(scores):
        if score >= relevance_threshold and i < nb_chunks_max:
            context.append(chunks[i])
    return context

__all__=[generate_response, prepare_llm_context]
