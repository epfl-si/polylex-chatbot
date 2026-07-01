import re
import os
import requests
from dotenv import find_dotenv, set_key

from .config import LEXES_API_URL
from .fedlex import get_fedlex_pdf_from_sparql

def fetch_polylex_api():
    response = requests.get(LEXES_API_URL)
    if response.status_code != 200:
        # TODO : a gerer dans les logs
        raise Exception(f"Unexpected status code while fetching : {response.status_code}")
    return response

def resolve_document_url(url, lang):
    transformed_url, source, content_format = "", "", ""
    if "inside.epfl.ch" in url:
        print(f"Can not load {url} (restricted access to EPFL members)")
        return transformed_url, source, content_format # empty value
    if "www.admin.ch" in url or "fedlex.admin.ch" in url:
        source = "fedlex"
        content_format = "pdf"
        if url.endswith(".pdf"):
            transformed_url = url
        else:
            if url == "http://www.admin.ch/ch/f/rs/22.html" or url == "https://www.admin.ch/opc/fr/classified-compilation/83.html":
                print(f"This page from Fedlex is not handled: {url}")
            else:
                transformed_url = get_fedlex_pdf_from_sparql(url, lang)
        return transformed_url, source, content_format
    if url.endswith(".pdf") or url.endswith(".docx"):
        transformed_url = url
        source = "others"
        content_format = "pdf" if url.endswith(".pdf") else "docx"
        return transformed_url, source, content_format
    epfl_redirect_urls_pattern = re.compile(r'^https://.*\.epfl\.ch$')
    epfl_websites_pattern = re.compile(r'^https://www\.epfl\.ch/(about|campus|education)/')
    epfl_apps_pattern = re.compile(r'(sac|isa)\.epfl\.ch')
    if url.endswith(".html") or epfl_redirect_urls_pattern.search(url) or epfl_websites_pattern or epfl_apps_pattern.search(url):
        # TODO : si site alors message d'avertissement et rien ou charger dans une cle fake tous les elements non charges ?
        print(f"{url} not loaded (website)")
        return transformed_url, source, content_format # empty value
    # TODO : a gerer dans les logs
    print(f"This is an exception and has to be handled: {url}")
    return transformed_url, source, content_format

def write_txt(filename, dir, content):
    with open(os.path.join(dir, filename), "w", encoding="utf-8") as f:
        f.write(content)

def download_file(url, dir, filename):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:147.0) Gecko/20100101 Firefox/147.0"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(os.path.join(dir, filename), "wb") as f:
            f.write(response.content)
    else:
        print(f"Error: the content for {url} can not be fetched (status: {response.status_code})")

def save_corpus_name(corpus_name):
    var_name = "CORPUS_NAME"
    value = str(corpus_name)
    set_key(
        dotenv_path=find_dotenv(),
        key_to_set=var_name,
        value_to_set=value
    )
    os.environ[var_name] = value

def download_documents(data, path, corpus_name):
    """
    Download the documents referenced in data to the specified directory
    """

    corpus_path = path / corpus_name
    corpus_path.mkdir(parents=True, exist_ok=True)

    save_corpus_name(corpus_name)

    for doc_id, metadata in data.items():
        content_format = metadata.get("content_format")
        redirected_url = metadata.get("redirected_url")

        if content_format in ["docx", "pdf"]:
            filename = f"{doc_id}.{content_format}"
            download_file(redirected_url, corpus_path, filename)
        else:
            print(f"File format '{content_format}' not yet supported for doc_id='{doc_id}'")

        for lang, summary in metadata.get("summaries").items():
            filename = f"{doc_id}_summary_{lang}.txt"
            content = f"{summary.get("title")}\n\n{summary.get("description")}".strip()
            write_txt(filename, corpus_path, content)

    return corpus_path

__all__ = ["fetch_polylex_api", "resolve_document_url", "download_documents"]
