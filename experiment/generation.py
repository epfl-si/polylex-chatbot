from langchain_core.output_parsers import StrOutputParser

from polylex_chatbot.config import LLM_MODEL_CONFIG

# TODO : merger avec code du chatbot
def build_generation_prompt(query, context):
    contents = [chunk.get("content") for chunk in context]
    context_text = "\n\n".join(f"[Chunk {i}]\n{content}" for i, content in enumerate(contents, start=1))
    prompt = f"""Réponds à la question en utilisant uniquement le contexte fourni.

Contexte:
{context_text}

Question:
{query}

Réponse:"""

    return prompt

# TODO : merger avec code du chatbot
def generate_response(query, context):
    llm = LLM_MODEL_CONFIG
    prompt = build_generation_prompt(query, context)
    chain = llm | StrOutputParser()
    response = chain.invoke(prompt).strip()
    return response

def make_generation_task(generation_top_k=5):
    def generation_task(*, item, **kwargs):
        query = item.input.get("query")
        contexts = item.input.get("retrieved_contexts")
        context_for_llm = contexts[:generation_top_k]
        generated_response = generate_response(query, context_for_llm)
        return {
            "generated_response": generated_response,
            "used_contexts": context_for_llm
        }
    return generation_task

__all__ = ["make_generation_task"]
