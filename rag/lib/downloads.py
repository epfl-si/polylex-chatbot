import os
import requests
from datetime import datetime

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

def archive_existing_and_create_data_directory(path):
    """
    Create a directory at the specified path.

    If the directory already exists, rename it by appending the current timestamp,
    then create a new empty directory at the original path.
    """

    if path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_path = path.with_name(f"{path.name}_{timestamp}")
        path.rename(archived_path)

    path.mkdir(parents=True, exist_ok=False)

def download_documents(data, path):
    '''
    Download the documents referenced in data to the specified directory
    '''

    archive_existing_and_create_data_directory(path)

    for doc_id, metadata in data.items():
        content_format = metadata.get("content_format")
        redirected_url = metadata.get("redirected_url")

        if content_format in ["docx", "pdf"]:
            filename = f"{doc_id}.{content_format}"
            download_file(redirected_url, path, filename)
        else:
            print(f"File format '{content_format}' not yet supported for doc_id='{doc_id}'")

        for lang, summary in metadata.get("summaries").items():
            filename = f"{doc_id}_summary_{lang}.txt"
            content = f"{summary.get("title")}\n\n{summary.get("description")}".strip()
            write_txt(filename, path, content)

__all__ = ["download_documents"]
