import os
import sys
import uuid
import logging
import chainlit as cl
from langdetect import detect
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from polylex_chatbot.env import load_project_env
env_path = load_project_env()

from polylex_chatbot.config import LANGUAGES, init_db_client, NB_CHUNKS_RETRIEVED, NB_CHUNKS_RERANKED, NB_CHUNKS_SENT, LLM_MODEL_CONFIG, MAX_USER_MESSAGE_LEN, PROMPT_TEMPLATE_FR, PROMPT_TEMPLATE_EN, RELEVANCE_THRESHOLD
from polylex_chatbot.retrieval import retrieve_documents
from polylex_chatbot.generation import generate_response

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
        "unsupported_language": "The detected language `{lg}` is not supported. Only questions in French or English are accepted.",
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

    query = message.content

    len_message = len(query)
    logger.info("New message received: content_length=%s", len_message)

    if len_message > MAX_USER_MESSAGE_LEN:
        logger.warning("Message from user too long: %s / %s characters", len_message, MAX_USER_MESSAGE_LEN)
        await cl.Message(content=translate("message_too_long", ui_lang, len_message=len_message, max_len=MAX_USER_MESSAGE_LEN)).send()
        return

    langfuse_handler = CallbackHandler()

    try:
        lang = detect(query)

        logger.info("Detected language: %s", lang)

        if lang not in LANGUAGES:
            logger.warning("Unsupported language detected: %s", lang)
            await cl.Message(content=translate("unsupported_language", ui_lang, lg=lang)).send()
            return

        config_by_lang = {
            "fr": {
                "qdrant_config": init_db_client(lang),
                "prompt": PROMPT_TEMPLATE_FR
            },
            "en": {
                "qdrant_config": init_db_client(lang),
                "prompt": PROMPT_TEMPLATE_EN
            }
        }

        retrieval_result = retrieve_documents(config_by_lang[lang]["qdrant_config"], query, os.getenv("MODEL_RERANKER_NAME"), os.getenv("MODEL_RERANKER_API_KEY"), os.getenv("MODELS_BASE_URL"), NB_CHUNKS_RETRIEVED, NB_CHUNKS_RERANKED)[:NB_CHUNKS_SENT]
        context_for_llm = retrieval_result.get("retrieved_contexts")
        retrieved_scores = retrieval_result.get("retrieved_scores")
        relevant_chunks = []
        for i, score in enumerate(retrieved_scores):
            if score >= RELEVANCE_THRESHOLD:
                relevant_chunks.append(context_for_llm[i])
        logger.info("Context retrieved: %s / %s relevant chunks with scores %s", len(relevant_chunks), len(context_for_llm), retrieved_scores)

        answer = generate_response(LLM_MODEL_CONFIG, query, config_by_lang[lang]["prompt"], context_for_llm, langfuse_handler)
        logger.info("Answer generated: len_answer=%s", len(answer or ""))

        source_refs = []
        source_registry = cl.user_session.get("source_registry") or {}

        for retrieved_chunk in relevant_chunks:
            content = retrieved_chunk.get("content")
            metadata = retrieved_chunk.get("metadata")
            lex_type = metadata.get("lex_type")
            lex_number = metadata.get("lex_number")
            src_url = metadata.get("src_url")
            cat = metadata.get("cat")
            appendix_label = translate("appendix", ui_lang)
            label = f"{lex_type} {lex_number}{f' ({appendix_label})' if cat == 'appendix' else ''}"

            source_id = str(uuid.uuid4())

            source_registry[source_id] = {
                "label": label,
                "chunk": content,
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
