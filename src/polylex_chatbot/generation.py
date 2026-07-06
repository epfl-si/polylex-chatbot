from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_context_for_llm(items, item_type):
    if len(items) == 0:
        return "-"

    contents = [item.get("content") for item in items]
    # TODO : a la place de item_type, remplacer par le nom de la LEX ?
    context = "\n\n".join(f"[{item_type} {i}]\n{content}" for i, content in enumerate(contents, start=1))

    return context

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

__all__ = ["build_context_for_llm", "generate_response"]
