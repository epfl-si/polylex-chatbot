import hashlib
from rag.lib.config import LANGUAGES
from rag.lib.html_utils import transform_html_in_text, get_urls_from_html
from rag.lib.documents import resolve_document_url

def make_doc_id(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]

def upsert_doc(data, key, cat, source, content_format, ref):
    if key not in data:
        data[key] = {
            "doc_id": make_doc_id(key),
            "cats": [],
            "source": source,
            "content_format": content_format,
            "refs": []
        }
    if cat not in data[key]["cats"]:
        data[key]["cats"].append(cat)
    data[key]["refs"].append(ref)

def build_metadata(response, debugging=False):
    data = {}

    for lex in response.json():
        lex_id = lex.get('_id')
        lex_type = lex.get("type")
        lex_number = lex.get("number")

        for lang in LANGUAGES:
            cap_lang = lang.capitalize()
            lex_url = lex.get(f"url{cap_lang}")
            lex_description = lex.get(f"description{cap_lang}")
            lex_summary = lex.get(f"title{cap_lang}") + "\n" + transform_html_in_text(lex_description)
            lex_appendix_urls = get_urls_from_html(lex_description)
            base_ref = {
                "lex_id": lex_id,
                "lex_type": lex_type,
                "lex_number": lex_number,
                "lex_lang": lang,
                "lex_url": lex_url
            }
            upsert_doc(data, lex_summary, "summary", "polylex", "txt", {**base_ref, "cat": "summary"})

            transformed_lex_url, lex_source, lex_format = resolve_document_url(lex_url, lang)
            if transformed_lex_url != "" and lex_source != "" and lex_format != "":
                upsert_doc(data, transformed_lex_url, "lex", lex_source, lex_format, {**base_ref, "cat": "lex"})

            for lex_appendix_url in lex_appendix_urls:
                transformed_lex_appendix_url, lex_appendix_source, lex_appendix_format = resolve_document_url(
                    lex_appendix_url, lang)
                if transformed_lex_appendix_url != "" and lex_appendix_source != "" and lex_appendix_format != "":
                    upsert_doc(data, transformed_lex_appendix_url, "appendix", lex_appendix_source, lex_appendix_format,
                               {**base_ref, "cat": "appendix"})
        if debugging:
            break

    return data

__all__ = [build_metadata]
