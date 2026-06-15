import os
import chainlit as cl
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from qdrant_client.http.models import SearchParams
from langdetect import detect

langfuse = Langfuse()

# TODO : utiliser @cl.on_chat_start pour historique ?
@cl.on_message
async def main(message: cl.Message):
    langfuse_handler = CallbackHandler()

    # TODO : secure content send from the user (for language detection + send to LLM)
    lang = detect(message.content)
    if lang not in ["fr", "en"]: # TODO : lire fichier de config !
        # TODO : handle this error with call to LLM + same for question out of context
        print(f"Language {lang} is not supported.")  # TODO
        return

    embeddings = OpenAIEmbeddings(
        model=os.getenv("MODEL_EMBEDDINGS_NAME"),
        base_url=os.getenv("MODELS_BASE_URL"),
        api_key=os.getenv("MODEL_EMBEDDINGS_API_KEY")
    )

    # TODO : improve prompts (simple to complex)
    prompt_template_en = """
You are an AI assistant specialized in answering questions using provided context.

Rules:
- You MUST base your answer ONLY on the provided context.
- If the context does not contain the answer, say "I don't know".
- Do NOT use prior knowledge unless explicitly allowed.
- Cite relevant parts of the context when possible.
- Be concise and factual.

Context:
{context}

Question:
{input}
"""

    prompt_template_fr = """
Vous êtes un assistant IA spécialisé dans la réponse aux questions à partir d’un contexte fourni.

Règles :
- Vous DEVEZ baser votre réponse UNIQUEMENT sur le contexte fourni.
- Si le contexte ne contient pas la réponse, dites "Je ne sais pas".
- N’utilisez PAS de connaissances externes sauf si cela est explicitement autorisé.
- Citez les parties pertinentes du contexte lorsque c’est possible.
- Soyez concis et factuel.

Contexte :
{context}

Question :
{input}
    """

    config_by_lang = {
        "fr": {
            "model": FastEmbedSparse(
                model_name=os.getenv("MODEL_SPARSE_NAME"),
                avg_len=os.getenv("MODEL_SPARSE_AVG_LEN_FR"),
                language="french"
            ),
            "vector_name" : os.getenv("MODEL_SPARSE_VECTOR_NAME_FR"),
            "prompt": prompt_template_fr
        },
        "en": {
            "model": FastEmbedSparse(
                model_name=os.getenv("MODEL_SPARSE_NAME"),
                avg_len=os.getenv("MODEL_SPARSE_AVG_LEN_EN"),
                language="english"
            ),
            "vector_name" : os.getenv("MODEL_SPARSE_VECTOR_NAME_EN"),
            "prompt": prompt_template_en
        }
    }

    retrievers = {
        lang: QdrantVectorStore.from_existing_collection(
            url=os.getenv("QDRANT_URL"),
            embedding=embeddings,
            sparse_embedding=cfg["model"],
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name=cfg["vector_name"],
        ).as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": int(os.getenv("QDRANT_NB_CHUNKS_RETRIEVED")),
                "search_params": SearchParams(exact=True)
            }
        )
        for lang, cfg in config_by_lang.items()
    }

    llm = OpenAI(
        model=os.getenv("MODEL_LLM_NAME"),
        base_url=os.getenv("MODELS_BASE_URL"),
        api_key=os.getenv("MODEL_LLM_API_KEY"),
        max_tokens=int(os.getenv("MAX_TOKENS_LLM")),
        temperature=float(os.getenv("TEMPERATURE_LLM"))
    )

    prompt = ChatPromptTemplate.from_template(config_by_lang[lang]["prompt"])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retrievers[lang], combine_docs_chain)
    response = retrieval_chain.invoke(
        {"input": message.content},
        config={"callbacks": [langfuse_handler]}
    )

    answer = response.get("answer")
    retrieved_chunks = response.get("context")

    sources = []
    actions = []

    for retrieved_chunk in retrieved_chunks:
        lex_type = retrieved_chunk.metadata.get("lex_type")
        lex_number = retrieved_chunk.metadata.get("lex_number")
        src_url = retrieved_chunk.metadata.get("src_url")
        cat = retrieved_chunk.metadata.get("cat")
        label = f"{lex_type} {lex_number}{' (appendix)' if cat == 'appendix' else ''}"
        actions.append(
            cl.Action(
                name="open_source",
                label=label,
                tooltip="Display source content", # TODO : a traduire
                payload={
                    "label": label,
                    "chunk": retrieved_chunk.page_content,
                    "url": src_url,
                },
            )
        )

        sources.append(
            cl.Text(
                content=retrieved_chunk.page_content, name=label, display="side"
            )
        )

    trace_id = langfuse_handler.last_trace_id

    actions.extend([
        cl.Action(
            name="like",
            payload={"trace_id": trace_id},
            label="👍",
            tooltip="The answer was useful!",
        ),
        cl.Action(
            name="dislike",
            payload={"trace_id": trace_id},
            label="👎",
            tooltip="The answer was useless!",
        ),
    ])

    await cl.Message(
        content=answer,
        actions=actions
    ).send()

    # TODO : trouver meilleur moyen pour acceder aux sources

@cl.action_callback("like")
async def like(action):
    langfuse.create_score(
        name="user-feedback",
        value=1,
        trace_id=action.payload.get('trace_id')
    )

@cl.action_callback("dislike")
async def dislike(action):
    langfuse.create_score(
        name="user-feedback",
        value=0,
        trace_id=action.payload.get('trace_id'),
    )

@cl.action_callback("open_source")
async def open_source(action: cl.Action):
    label = action.payload["label"]
    chunk = action.payload["chunk"]
    url = action.payload["url"]

    await cl.ElementSidebar.set_title(label)
    await cl.ElementSidebar.set_elements([
        cl.Text(
            name=label,
            content=chunk,
            display="side",
        )
    ])

    await cl.Message(
        content=f"Source `{label}` : [open in browser]({url})" # TODO : nul
    ).send()
