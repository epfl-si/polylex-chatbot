import hashlib
import re
import os
import json
from tika import parser
from docx import Document
from langdetect import detect

from .stats import count_nb_tokens
from .chunking import get_doc_id_from_file, clean_text
from .config import LANGUAGES, HARD_CODED_LANGS
from .downloads import resolve_document_url, write_txt
from .html_utils import transform_html_in_text, get_urls_from_html

def make_doc_id(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]

def upsert_doc(metadata_dict, redirected_url, src_url, cat, source, content_format, ref, title, description):
    doc_id = make_doc_id(redirected_url)
    if doc_id not in metadata_dict:
        metadata_dict[doc_id] = {
            "filename": f"{doc_id}.{content_format}",
            "src_url": src_url,
            "redirected_url": redirected_url,
            "cats": [],
            "source": source,
            "content_format": content_format,
            "refs": [],
            "summaries": {}
        }

    if cat not in metadata_dict[doc_id]["cats"]:
        metadata_dict[doc_id]["cats"].append(cat)
    metadata_dict[doc_id]["refs"].append(ref)

    # no summary to index if document is only an appendix
    # and no override either because never several lexes in refs
    if cat != "appendix":
        lang = ref.get("lex_lang")
        metadata_dict[doc_id]["summaries"][lang] = {
            "title": title,
            "description": description
        }

def join_language(metadata_dict):
    """
    Add unique language for each entry of data
    """

    for doc_id, metadata in metadata_dict.items():
        refs = metadata.get("refs")
        tmp_metadata = metadata
        if len(refs) == 1:
            tmp_metadata["lang"] = refs[0].get("lex_lang")
            metadata_dict[doc_id] = tmp_metadata
        else:
            content_format = metadata.get("content_format")
            tmp_metadata["lang"] = detect_language(metadata.get("redirected_url"), content_format)
            metadata_dict[doc_id] = tmp_metadata
    return metadata_dict

def build_metadata(response, debugging=False):
    metadata_dict = {}

    for lex in response.json():
        lex_id = lex.get('_id')
        lex_type = lex.get("type")
        lex_number = lex.get("number")

        for lang in LANGUAGES:
            cap_lang = lang.capitalize()
            src_url = lex.get(f"url{cap_lang}")
            lex_title = lex.get(f"title{cap_lang}")
            lex_description = lex.get(f"description{cap_lang}")
            appendix_urls = get_urls_from_html(lex_description)
            lex_description_cleaned = transform_html_in_text(lex_description)

            base_ref = {
                "lex_id": lex_id,
                "lex_type": lex_type,
                "lex_number": lex_number,
                "lex_lang": lang
            }

            redirected_url, source, content_format = resolve_document_url(src_url, lang)
            if redirected_url != "" and source != "" and content_format != "":
                upsert_doc(metadata_dict, redirected_url, src_url, "lex", source, content_format, {**base_ref, "cat": "lex"}, lex_title, lex_description_cleaned)

            for appendix_url in appendix_urls:
                redirected_url, source, content_format = resolve_document_url(appendix_url, lang)
                if redirected_url != "" and source != "" and content_format != "":
                    upsert_doc(metadata_dict, redirected_url, appendix_url, "appendix", source, content_format, {**base_ref, "cat": "appendix"}, lex_title, lex_description_cleaned)

        if debugging:
            break

    return join_language(metadata_dict)

def detect_language(content, content_format):
    # TODO : plus de txt donc dead code + lire premiere page du fichier plutot que seulement detecter selon url ?
    # use directly lib to find language from a text
    if content_format == "txt":
        return detect(content)

    # try to find lang tag in url
    lang_pattern = re.compile(r"[_-](fr|en|an)\.[^/]+$", re.IGNORECASE)
    match = re.search(lang_pattern, content)
    detected_lang = match.group(1) if match else ""
    if detected_lang != "":
        detected_lang = detected_lang.lower()
        detected_lang = "en" if detected_lang == "an" else detected_lang
        return detected_lang

    # use hard-coded correct language for exceptions
    real_lang = HARD_CODED_LANGS.get(content)
    if real_lang:
        return real_lang

    # try to detect language from filename -> FIXME : avertir que pas tres fiable ?
    filename_pattern = re.compile(r"[^/]+\.[^/]+$")
    match = re.search(filename_pattern, content)
    filename = match.group() if match else ""
    detected_lang = detect(filename)
    if detected_lang in LANGUAGES:
        return detected_lang

    print(f"Error while trying to detect language for {content}, default to 'fr'")
    return "fr"

def add_metadata_for_entry(metadata_dict, content):
    metadata_dict["nb_tokens"] = count_nb_tokens(content)
    metadata_dict["is_indexed"] = False if len(content) >= 200000 else True
    return metadata_dict

def save_textual_content_and_complete_metadata(path_to_read, path_to_save, metadata):
    for file in path_to_read.iterdir():
        suffix = file.suffix

        if suffix == ".txt":
            content = file.read_text(encoding="utf-8")
        elif suffix == ".docx":
            doc = Document(file)
            content = "\n".join(p.text for p in doc.paragraphs)
        elif suffix == ".pdf":
            parsed = parser.from_file(str(file))
            content = parsed.get("content")
        else:
            # TODO : gerer dans les logs
            content = ""
            print(f"Error while reading {file}: format '{suffix}' not supported")

        # save textual content
        filename = f"{file.stem}.txt"
        cleaned_content = clean_text(content, metadata.get("source"))
        write_txt(filename, path_to_save, cleaned_content)

        # complete metadata
        if suffix != ".txt":
            doc_id = get_doc_id_from_file(file)
            metadata_dict = metadata[doc_id]
            completed_metadata_dict = add_metadata_for_entry(metadata_dict, content)
            metadata[doc_id] = completed_metadata_dict

    return metadata

def save_metadata(metadata, path):
    """
    Save metadata to the specified path
    """
    corpus_metadata_filename = os.path.join(path, "corpus_metadata.json")

    with open(corpus_metadata_filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    return corpus_metadata_filename

def load_metadata(path, only_indexed_documents=False):
    metadata = {}

    corpus_metadata_filename = os.path.join(path, "corpus_metadata.json")

    with open(corpus_metadata_filename, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if only_indexed_documents:
        return {
            url: entry
            for url, entry in metadata.items()
            if entry.get("is_indexed") is True
        }

    return metadata

def find_best_ref(metadata_dict):
    refs = metadata_dict.get("refs")
    lang = metadata_dict.get("lang")

    first_ref = refs[0]
    same = all(
        ref.get("lex_id") == first_ref.get("lex_id") and
        ref.get("lex_type") == first_ref.get("lex_type") and
        ref.get("lex_number") == first_ref.get("lex_number")
        for ref in refs
    )

    # same lex but mistmatch between languages
    if same:
        for ref in refs:
            if ref.get("lex_lang") == lang:
                return ref

    # ref from lex and appendix, best is lex
    for ref in refs:
        if ref.get("lex_type") == "lex":
            return ref

    # ref from appendices, first one chosen but only a few exceptions
    return refs[0]

def build_language_matched_metadata_by_doc_id(metadata):
    language_matched_metadata_by_doc_id = {}

    for doc_id, metadata_dict in metadata.items():
        ref = find_best_ref(metadata_dict)
        lang = metadata_dict.get("lang")
        language_matched_metadata_by_doc_id[doc_id] = {
            "is_indexed": metadata_dict.get("is_indexed"),
            "title": metadata_dict.get("summaries", {}).get(lang, {}).get("title", ""),
            "lex_id": ref.get("lex_id"),
            "lex_type": ref.get("lex_type"),
            "lex_number": ref.get("lex_number"),
            "lex_lang": lang,
            "cat": ref.get("cat"),
            "src_url": metadata_dict.get("src_url"),
            "source": metadata_dict.get("source"),
            "content_format": metadata_dict.get("content_format"),
            "nb_tokens": metadata_dict.get("nb_tokens")
        }

    return language_matched_metadata_by_doc_id

__all__ = ["build_metadata", "save_textual_content_and_complete_metadata", "save_metadata", "load_metadata", "build_language_matched_metadata_by_doc_id"]
