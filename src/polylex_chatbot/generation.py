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
