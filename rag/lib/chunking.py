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

def get_doc_id_from_file(file):
    suffix = file.suffix.lower()
    stem = file.stem

    if suffix in [".pdf", ".docx"]:
        return stem

    if suffix == ".txt":
        return stem.split("_summary_")[0]

    return None

def create_chunks(path, language_matched_metadata_by_doc_id):
    '''
    Create text chunks for each document in the given path and enrich them with metadata
    '''

    chunks = []

    for file in path.iterdir():
        doc_id = get_doc_id_from_file(file)

        if doc_id is None or not language_matched_metadata_by_doc_id[doc_id]["is_indexed"]:
            continue

        title = language_matched_metadata_by_doc_id[doc_id]["title"]
        source = language_matched_metadata_by_doc_id[doc_id]["source"]
        suffix = file.suffix.lower()

        if suffix == ".txt":
            extracted_text = file.read_text(encoding="utf-8")
            extracted_metadata = {}
        elif suffix in [".docx", ".pdf"]:
            parsed_doc = parser.from_file(str(file), requestOptions={"timeout": 300})
            extracted_text = parsed_doc.get("content")
            extracted_metadata = parsed_doc.get("metadata")
        else:
            # TODO : mettre dans les logs
            print(f"File '{file}' not chunked (suffix not handled)")
            continue

        cleaner_text = clean_text(extracted_text, source)

        doc = Document(
            page_content=cleaner_text,
            metadata={
                "doc_id": doc_id,
                "filepath": str(file),
                "total_pages": int(extracted_metadata.get("xmpTPg:NPages", 1)),
                "creationDate": extracted_metadata.get("xmp:CreateDate"),
                "src_url": language_matched_metadata_by_doc_id[doc_id]["src_url"],
                "language": language_matched_metadata_by_doc_id[doc_id]["lex_lang"],
                "cat": language_matched_metadata_by_doc_id[doc_id]["cat"],
                "source": source,
                "content_format": language_matched_metadata_by_doc_id[doc_id]["content_format"],
                "lex_id": language_matched_metadata_by_doc_id[doc_id]["lex_id"],
                "lex_type": language_matched_metadata_by_doc_id[doc_id]["lex_type"],
                "lex_number": language_matched_metadata_by_doc_id[doc_id]["lex_number"]
            },
        )

        chunks_from_struct = SPLITTER.split_documents([doc])

        for chunk in chunks_from_struct:
            page_content = f"{title}\n\n{chunk.page_content}" if title else chunk.page_content
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

# TODO : save dans config et pas env...
def save_avg_lens(path, chunks_splitted_by_lang):
    env_path = Path(path)
    env_path.touch(exist_ok=True)
    for lang, data in chunks_splitted_by_lang.items():
        avg_len = round(data["avg_len"], 2)
        var_name = f"AVG_LEN_{lang.upper()}"
        set_key(
            dotenv_path=str(env_path),
            key_to_set=var_name,
            value_to_set=str(avg_len),
            quote_mode="never"
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
