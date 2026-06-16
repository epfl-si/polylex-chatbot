import re
from .fedlex import get_fedlex_pdf_from_sparql

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
