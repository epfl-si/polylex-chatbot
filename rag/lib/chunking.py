import re
from tika import parser
from pathlib import Path
from dotenv import set_key
from langchain_core.documents import Document

from rag.lib.config import SPLITTER

def clean_text(text, source):
    # TODO : a ameliorer
    cleaner_text = re.sub(r'\s+', ' ', text).strip()
    cleaner_text = re.sub(r"\.{4,}", " ", cleaner_text)
    if source == "fedlex":
        cleaner_text = re.sub(r'\b(?:RO \d{4} \d+|RS \d+(?:\.\d+)+)\b', '', cleaner_text)
    return cleaner_text

def create_chunks(path, doc_id_to_metadata_lookup, metadata_to_title_lookup):
    '''
    Create text chunks for each indexed PDF in the given path and enrich them with metadata
    '''

    chunks = []

    for file in path.glob("*.pdf"):
        doc_id = file.stem
        if not doc_id_to_metadata_lookup[doc_id]["is_indexed"]:
            continue
        lex_id = doc_id_to_metadata_lookup[doc_id]["lex_id"]
        lex_number = doc_id_to_metadata_lookup[doc_id]["lex_number"]
        lex_url = doc_id_to_metadata_lookup[doc_id]["lex_url"]
        lang = doc_id_to_metadata_lookup[doc_id]["lang"]
        source = doc_id_to_metadata_lookup[doc_id]["source"]
        summary = metadata_to_title_lookup.get((lex_id, lang))
        # FIXME : comment faire si txt pas dispo dans la bonne langue (est-ce que ca arrive ?) ?
        if not summary:
            print(f"KO for {file}")
        title = summary.split("\n")[0]
        parsed_pdf = parser.from_file(str(file))
        extracted_text = parsed_pdf.get("content")
        cleaner_text = clean_text(extracted_text, source)
        extracted_metadata = parsed_pdf.get("metadata")

        doc = Document(
            page_content=cleaner_text,
            metadata={
                "doc_id": doc_id,
                "filepath": str(file),
                "language": lang,
                "source": source,
                "lex_number": lex_number,
                "lex_url": lex_url,
                "total_pages": int(extracted_metadata["xmpTPg:NPages"]),
                "creationDate": extracted_metadata.get('xmp:CreateDate')
            },
        )

        chunks_from_struct = SPLITTER.split_documents([doc])

        for chunk in chunks_from_struct:
            page_content = f"{title} \n {chunk.page_content}" if title else chunk.page_content
            metadata = chunk.metadata
            chunks.append(
                Document(
                    page_content=page_content,
                    metadata=metadata  # TODO : ajouter reference aux articles contenus dans ce chunk ?
                )
            )

    return chunks

def save_chunks(txt_path, chunks):
    with open(txt_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            content = f"\n------------ DOC ID: {chunk.metadata["doc_id"]} - LANGUAGE: {chunk.metadata["language"]} - SOURCE: {chunk.metadata["source"]} - LEX NUMBER: {chunk.metadata["lex_number"]} - TOTAL PAGES: {chunk.metadata["total_pages"]} - START INDEX: {chunk.metadata["start_index"]} ------------\n"
            f.write(content + chunk.page_content + "\n")

def save_avg_lens(path, chunks_splitted_by_lang):
    env_path = Path(path)
    env_path.touch(exist_ok=True)
    for lang, data in chunks_splitted_by_lang.items():
        avg_len = round(data["avg_len"], 2)
        var_name = f"AVG_LEN_{lang.upper()}"
        set_key(
            dotenv_path=str(env_path),
            key_to_set=var_name,
            value_to_set=avg_len
        )

def divide_chunks_per_lang(chunks, langs, path):
    result = {
        lang: {
            "chunks": [],
            "avg_len": 0
        }
        for lang in langs
    }

    for chunk in chunks:
        chunk_lang = chunk.metadata.get("language")
        result[chunk_lang]["chunks"].append(chunk)

    for data in result.values():
        chunks_per_lang = data["chunks"]
        data["avg_len"] = sum(len(chunk.page_content.split()) for chunk in chunks_per_lang) / len(chunks_per_lang)

    save_avg_lens(path, result)

    return result

__all__ = ["create_chunks", "save_chunks", "divide_chunks_per_lang"]
