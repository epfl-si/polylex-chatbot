from langchain_core.prompts import ChatPromptTemplate

def build_context_for_llm(items):
    if len(items) == 0:
        return "-"

    contents = [item.get("content") for item in items]

    formatted_refs = []
    for item in items:
        metadata = item.get("metadata")
        lex_type = metadata.get("lex_type")
        lex_number = metadata.get("lex_number")
        category = metadata.get("cat")
        formatted_ref = f"{lex_type} {lex_number}"
        if category == "appendix":
            formatted_ref = formatted_ref + " (appendix)"
        formatted_refs.append(formatted_ref)

    context = "\n\n".join(f"[{item_ref} {i}]\n{content}" for i, (content, item_ref) in enumerate(zip(contents, formatted_refs), start=1))

    return context

def generate_response(llm, query, prompt, context, monitoring_config=None):
    prompt_template = ChatPromptTemplate.from_template(prompt)
    raw_chain = prompt_template | llm

    inputs = {
        "query": query,
        "context_text": context
    }

    if monitoring_config is None:
        llm_results = raw_chain.invoke(inputs)
    else:
        llm_results = raw_chain.invoke(inputs, config={"callbacks": [monitoring_config]})

    response = getattr(llm_results, "content")

    finish_reason = getattr(llm_results, "response_metadata").get("finish_reason")
    usage_metadata = getattr(llm_results, "usage_metadata")

    return response, finish_reason, usage_metadata

__all__ = ["build_context_for_llm", "generate_response"]
