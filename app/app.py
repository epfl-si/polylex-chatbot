import os
import sys
import uuid
import logging
import chainlit as cl
from langdetect import detect
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from qdrant_client.http.models import SearchParams
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from polylex_chatbot.env import load_project_env
env_path = load_project_env()

from polylex_chatbot.config import (LANGUAGES,
                                    EMBEDDING_MODEL_CONFIG, LLM_MODEL_CONFIG,
                                    DB_DENSE_INDEX_CONFIG, SPARSE_MODEL_CONFIG_FR, SPARSE_MODEL_CONFIG_EN,
                                    DB_SPARSE_INDEX_CONFIG_FR, DB_SPARSE_INDEX_CONFIG_EN,
                                    QDRANT_NB_CHUNKS_RETRIEVED,
                                    MAX_USER_MESSAGE_LEN
                                    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

TRANSLATIONS = {
    "en": {
        "message_too_long": "The message is too long ({len_message} / {max_len} characters).",
        "unsupported_language": "The detected language `{lang}` is not supported. Only questions in French or English are accepted.",
        "generic_error": "An error occurred while processing the question. Please try again.",
        "source_display_error": "Unable to display this source.",
        "like_tooltip": "The answer was useful!",
        "dislike_tooltip": "The answer was not useful.",
        "appendix": "appendix"
    },
    "fr": {
        "message_too_long": "Le message est trop long ({len_message} / {max_len} caractères).",
        "unsupported_language": "La langue détectée `{lg}` n'est pas supportée. Seules les questions posées en français ou en anglais sont acceptées.",
        "generic_error": "Une erreur est survenue pendant le traitement de la question. Veuillez réessayer.",
        "source_display_error": "Impossible d'afficher cette source.",
        "like_tooltip": "La réponse était utile !",
        "dislike_tooltip": "La réponse n'était pas utile.",
        "appendix": "annexe"
    }
}

def get_ui_lang():
    languages = cl.user_session.get("languages") or "" # TODO : selon https://github.com/Chainlit/chainlit/issues/879 pas possible de recuperer la langue du navigateur...
    first_lang = languages.split(",")[0].split(";")[0].strip().lower()
    selected_lang = "en" if first_lang.startswith("en") else "fr"
    return selected_lang

def translate(key, lang, **kwargs):
    translations = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    template = translations.get(key, TRANSLATIONS["fr"].get(key, key))
    return template.format(**kwargs)

langfuse = Langfuse()

# TODO : utiliser @cl.on_chat_start pour historique ?
@cl.on_message
async def main(message: cl.Message):
    ui_lang = get_ui_lang()
    cl.user_session.set("ui_lang", ui_lang)

    len_message = len(message.content)
    logger.info("New message received: content_length=%s", len_message)

    if len_message > MAX_USER_MESSAGE_LEN:
        logger.warning("Message from user too long: %s / %s characters", len_message, MAX_USER_MESSAGE_LEN)
        await cl.Message(content=translate("message_too_long", ui_lang, len_message=len_message, max_len=MAX_USER_MESSAGE_LEN)).send()
        return

    langfuse_handler = CallbackHandler()

    try:
        lang = detect(message.content)

        logger.info("Detected language: %s", lang)

        if lang not in LANGUAGES:
            logger.warning("Unsupported language detected: %s", lang)
            await cl.Message(content=translate("unsupported_language", ui_lang, lg=lang)).send()
            return

        embeddings = EMBEDDING_MODEL_CONFIG

        # TODO : dire que si question hors contexte alors ne pas prendre en compte + répondre dans la langue de l'utilisateur
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
                "model": SPARSE_MODEL_CONFIG_FR,
                "vector_name" : list(DB_SPARSE_INDEX_CONFIG_FR.keys())[0],
                "prompt": prompt_template_fr
            },
            "en": {
                "model": SPARSE_MODEL_CONFIG_EN,
                "vector_name" : list(DB_SPARSE_INDEX_CONFIG_EN.keys())[0],
                "prompt": prompt_template_en
            }
        }

        retrievers = {
            lang: QdrantVectorStore.from_existing_collection(
                url=os.getenv("QDRANT_URL"),
                embedding=embeddings,
                sparse_embedding=cfg["model"],
                collection_name=os.getenv("DB_COLLECTION_NAME"),
                retrieval_mode=RetrievalMode.HYBRID,
                vector_name=list(DB_DENSE_INDEX_CONFIG.keys())[0],
                sparse_vector_name=cfg["vector_name"],
            ).as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": QDRANT_NB_CHUNKS_RETRIEVED,
                    "search_params": SearchParams(exact=True)
                }
            )
            for lang, cfg in config_by_lang.items()
        }

        llm = LLM_MODEL_CONFIG

        prompt = ChatPromptTemplate.from_template(config_by_lang[lang]["prompt"])
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retrievers[lang], combine_docs_chain)

        logger.info("Invoking retrieval chain.")
        response = retrieval_chain.invoke(
            {"input": message.content},
            config={"callbacks": [langfuse_handler]}
        )

        answer = response.get("answer")
        retrieved_chunks = response.get("context") or []
        logger.info(
            "Retrieval chain completed: nb_chunks=%s and len_answer=%s",
            len(retrieved_chunks),
            len(answer or "")
        )

        source_refs = []
        source_registry = cl.user_session.get("source_registry") or {}

        for i, retrieved_chunk in enumerate(retrieved_chunks, start=1):
            lex_type = retrieved_chunk.metadata.get("lex_type")
            lex_number = retrieved_chunk.metadata.get("lex_number")
            src_url = retrieved_chunk.metadata.get("src_url")
            cat = retrieved_chunk.metadata.get("cat")
            appendix_label = translate("appendix", ui_lang)
            label = f"{lex_type} {lex_number}{f' ({appendix_label})' if cat == 'appendix' else ''}"

            source_id = str(uuid.uuid4())

            source_registry[source_id] = {
                "label": label,
                "chunk": retrieved_chunk.page_content,
                "url": src_url
            }

            source_refs.append({
                "id": source_id,
                "label": label,
                "url": src_url,
            })

        cl.user_session.set("source_registry", source_registry)

        elements = []
        if source_refs:
            source_element = cl.CustomElement(
                name="SourceReferences",
                props={"sources": source_refs},
                display="inline"
            )
            elements.append(source_element)

        trace_id = langfuse_handler.last_trace_id

        actions = [
            cl.Action(
                name="like",
                payload={"trace_id": trace_id},
                label="👍",
                tooltip=translate("like_tooltip", ui_lang)
            ),
            cl.Action(
                name="dislike",
                payload={"trace_id": trace_id},
                label="👎",
                tooltip=translate("dislike_tooltip", ui_lang)
            ),
        ]

        await cl.Message(
            content=answer,
            elements=elements,
            actions=actions
        ).send()

        logger.info("Answer sent successfully: trace_id=%s", trace_id)

    except Exception:
        logger.exception("Unhandled error while processing message")
        await cl.Message(content=translate("generic_error", ui_lang)).send()

@cl.action_callback("like")
async def like(action):
    trace_id = action.payload.get("trace_id")
    langfuse.create_score(
        name="user-feedback",
        value=1,
        trace_id=trace_id
    )
    logger.info("User feedback received (value=1): trace_id=%s", trace_id)

@cl.action_callback("dislike")
async def dislike(action):
    trace_id = action.payload.get("trace_id")
    langfuse.create_score(
        name="user-feedback",
        value=0,
        trace_id=trace_id
    )
    logger.info("User feedback received (value=0): trace_id=%s", trace_id)

@cl.action_callback("open_source")
async def open_source(action):
    ui_lang = cl.user_session.get("ui_lang")
    source_id = action.payload.get("source_id")

    source_registry = cl.user_session.get("source_registry") or {}
    source = source_registry.get(source_id)

    label = source["label"]
    chunk = source["chunk"]
    url = source["url"]

    logger.info("Source clicked: label=%s and url=%s", label, url)

    try:
        await cl.ElementSidebar.set_title(label)
        await cl.ElementSidebar.set_elements([
            cl.Text(
                name=label,
                content=chunk,
                display="side"
            )
        ])

        logger.info("Source displayed in sidebar: label=%s", label)

    except Exception:
        logger.exception("Failed to open source: label=%s and url=%s", label, url)
        await cl.Message(content=translate("source_display_error", ui_lang)).send()
