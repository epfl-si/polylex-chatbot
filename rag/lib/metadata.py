import hashlib
import re
import os
import json
import pandas as pd
from tika import parser
from datetime import datetime
from langdetect import detect
from rag.lib.config import LANGUAGES, HARD_CODED_LANGS, ARTICLE_PATTERN
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

def join_language(data):
    '''
    Add unique language for each entry of data
    '''
    for content, metadata_dict in data.items():
        refs = metadata_dict.get("refs")
        metadata = metadata_dict
        if len(refs) == 1:
            metadata["lang"] = refs[0].get("lex_lang")
            data[content] = metadata
        else:
            content_format = metadata_dict.get("content_format")
            metadata["lang"] = detect_language(content, content_format)
            data[content] = metadata
    return data

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

    return join_language(data)

def detect_language(content, content_format):
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

def add_indexing_flag(metadata, data_path):
    '''
    Add an `is_indexed` flag to each entry in metadata based on document statistics
    '''

    # TODO : remove pandas df
    stats_per_doc = []

    for file in data_path.glob("*.pdf"):
        parsed_file = parser.from_file(str(file))
        file_metadata = parsed_file.get("metadata", {})
        nb_pages = int(file_metadata.get("xmpTPg:NPages", 1))
        extracted_text = parsed_file.get("content") or ""
        nb_occ_article = sum(1 for _ in ARTICLE_PATTERN.finditer(extracted_text))
        stats_per_doc.append({
            "doc_id": file.stem,
            "nb_pages": nb_pages,
            "nb_occ_article": nb_occ_article
        })

    stats = pd.DataFrame(stats_per_doc)
    pdfs_not_to_index = stats.loc[(stats["nb_pages"] > 100) | ((stats["nb_pages"] > 50) & (stats["nb_occ_article"] == 0))]["doc_id"].values
    
    # TODO : gerer les logs
    print(pdfs_not_to_index)

    for metadata_dict in metadata.values():
        doc_id = metadata_dict["doc_id"]
        if doc_id in pdfs_not_to_index:
            metadata_dict["is_indexed"] = False
        else:
            metadata_dict["is_indexed"] = True

    return metadata

def save_metadata(metadata, path):
    '''
    Save metadata to disk
    '''

    os.makedirs(path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_filename = os.path.join(path, f"{timestamp}_lexes_metadata.json")

    with open(metadata_filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

def load_metadata(path):
    metadata = {}

    most_recent_file = max(
        (f for f in path.iterdir() if f.suffix == ".json"),
        key=lambda f: f.stat().st_mtime
    )

    with open(most_recent_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
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

    # ref from appendices, first one (random) # FIXME : peut mieux faire ?
    return refs[0]

def build_metadata_lookup_tables(metadata):
    doc_id_to_metadata = {}
    metadata_to_title = {}

    for content, metadata_dict in metadata.items():
        ref = find_best_ref(metadata_dict)
        lex_id = ref.get("lex_id")
        lang = metadata_dict.get("lang")
        if "summary" in metadata_dict.get("cats"):
            metadata_to_title[(lex_id, lang)] = content
        else:
            doc_id = metadata_dict.get("doc_id")
            lex_number = ref.get("lex_number")
            lex_url = ref.get("lex_url")
            source = metadata_dict.get("source")
            is_indexed = metadata_dict.get("is_indexed")
            doc_id_to_metadata[doc_id] = {"lex_id": lex_id, "lex_number": lex_number, "lex_url": lex_url, "lang": lang,
                                          "source": source, "is_indexed": is_indexed}

    return doc_id_to_metadata, metadata_to_title

__all__ = [build_metadata, add_indexing_flag, save_metadata, load_metadata, build_metadata_lookup_tables]
