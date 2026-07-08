import os
import re
from matplotlib import pyplot as plt
from dotenv import find_dotenv, set_key
from langchain_core.documents import Document

def clean_text(text, source):
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

def create_chunks(path, language_matched_metadata_by_doc_id, split_document_function):
    chunks = []

    for file in path.iterdir():
        filename = file.stem
        doc_id = get_doc_id_from_file(file)

        if language_matched_metadata_by_doc_id[doc_id]["is_indexed"]:
            title = language_matched_metadata_by_doc_id[doc_id]["title"]
            source = language_matched_metadata_by_doc_id[doc_id]["source"]
            content = file.read_text(encoding="utf-8")

            if "summary" in filename:
                category = "summary"
                content_format = "txt"
                nb_tokens = 0
            else:
                category = language_matched_metadata_by_doc_id[doc_id]["cat"]
                content_format = language_matched_metadata_by_doc_id[doc_id]["content_format"]
                nb_tokens = language_matched_metadata_by_doc_id[doc_id]["nb_tokens"]

            doc = Document(
                page_content=content,
                metadata={
                    "doc_id": doc_id,
                    "filename": filename,
                    "nb_tokens": nb_tokens,
                    "src_url": language_matched_metadata_by_doc_id[doc_id]["src_url"],
                    "language": language_matched_metadata_by_doc_id[doc_id]["lex_lang"],
                    "cat": category,
                    "source": source,
                    "content_format": content_format,
                    "lex_id": language_matched_metadata_by_doc_id[doc_id]["lex_id"],
                    "lex_type": language_matched_metadata_by_doc_id[doc_id]["lex_type"],
                    "lex_number": language_matched_metadata_by_doc_id[doc_id]["lex_number"]
                },
            )

            chunks_from_struct = split_document_function([doc])

            for chunk in chunks_from_struct:
                page_content = f"{title}\n\n{chunk.page_content}" if title else chunk.page_content
                metadata = chunk.metadata

                chunks.append(
                    Document(
                        page_content=page_content,
                        metadata=metadata
                    )
                )

    return chunks

def save_chunks_distribution(path, chunks):
    contents = [chunk.page_content for chunk in chunks]
    content_nb_chars = [len(content) for content in contents]

    plt.hist(content_nb_chars, bins=50)
    plt.xlabel("Nb chars")
    plt.ylabel("Frequency")
    plt.title(f"Nb chunks: {len(chunks)} / min_len={min(content_nb_chars)} and max_len={max(content_nb_chars)}")

    plt.savefig(path / "plot_chunks_distribution.png")

def save_chunks(path, chunks):
    dir_collection = path / os.getenv("CORPUS_NAME") / os.getenv("DB_COLLECTION_NAME")
    dir_collection.mkdir(parents=True, exist_ok=True)
    chunks_filename = os.path.join(dir_collection, "chunks.txt")

    with open(chunks_filename, "w", encoding="utf-8") as f:
        for chunk in chunks:
            content = f"\n------------ DOC ID: {chunk.metadata["doc_id"]} - LANGUAGE: {chunk.metadata["language"]} - SOURCE: {chunk.metadata["source"]} - LEX NUMBER: {chunk.metadata["lex_number"]} - START INDEX: {chunk.metadata["start_index"]} ------------\n"
            f.write(content + chunk.page_content + "\n")

    save_chunks_distribution(dir_collection, chunks)

    return chunks_filename

def save_avg_lens(chunks_splitted_by_lang, env_file):
    for lang, data in chunks_splitted_by_lang.items():
        avg_len = round(data["avg_len"], 2)
        var_name = f"AVG_LEN_{lang.upper()}"
        value = str(avg_len)
        set_key(
            dotenv_path=find_dotenv(filename=env_file),
            key_to_set=var_name,
            value_to_set=value
        )
        os.environ[var_name] = value

def divide_chunks_per_lang(chunks, langs, env_file):
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

    save_avg_lens(result, env_file)

    return result

__all__ = ["clean_text", "get_doc_id_from_file", "create_chunks", "save_chunks", "divide_chunks_per_lang"]
