import requests
from rag.lib.config import LEXES_API_URL

def fetch_polylex_api():
    response = requests.get(LEXES_API_URL)
    if response.status_code != 200:
        # TODO : a gerer dans les logs
        raise Exception(f"Unexpected status code while fetching : {response.status_code}")
    return response
