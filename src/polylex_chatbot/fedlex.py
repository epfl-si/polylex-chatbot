import requests
from playwright.sync_api import sync_playwright

def resolve_redirect(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
            return page.url
        finally:
            browser.close()

def get_fedlex_api_style_url(url):
    # manual tricks but seems to work...
    api_url = url.replace("https://www.fedlex.admin.ch/", "https://fedlex.data.admin.ch/")
    api_url = api_url.replace("fedlex.data.admin.ch/eli/oc", "fedlex.data.admin.ch/eli/cc")
    url_end_exceptions = ["/fr", "/en", "/fr#a2"]
    for url_end in url_end_exceptions:
        if api_url.endswith(url_end):
            api_url = api_url.replace(url_end, "")
            break
    return api_url

def get_fedlex_pdf_url(url, lang):
    if lang == "en":
        lang_uri = "ENG"
    else:
        lang_uri = "FRA"
    sparql_query = f"""
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
SELECT ?publicationDate ?dateApplicability ?fileUrl WHERE {{
  ?work jolux:isMemberOf <{url}> ;
        jolux:dateApplicability ?dateApplicability ;
        jolux:isRealizedBy ?expr .
  OPTIONAL {{ ?work jolux:publicationDate ?publicationDate }}
  ?expr jolux:language <http://publications.europa.eu/resource/authority/language/{lang_uri}> ;
        jolux:isEmbodiedBy ?manif .
  ?manif jolux:userFormat <https://fedlex.data.admin.ch/vocabulary/user-format/pdf-a> ;
        jolux:isExemplifiedBy ?fileUrl .
}}
ORDER BY DESC(?publicationDate) DESC(?dateApplicability)
LIMIT 1
""".strip()
    endpoint = "https://fedlex.data.admin.ch/sparqlendpoint"
    r = requests.get(endpoint, params={"query": sparql_query, "format": "application/sparql-results+json"})
    r.raise_for_status()
    data = r.json()
    bindings = data["results"]["bindings"]
    if not bindings:
        print(f"No SPARQL results for {url}")
        return ""
    return bindings[0]["fileUrl"]["value"]

def get_fedlex_pdf_from_sparql(url, lang):
    redirected_url = resolve_redirect(url)
    sparql_url = get_fedlex_api_style_url(redirected_url)
    return get_fedlex_pdf_url(sparql_url, lang)

__all__ = ["get_fedlex_pdf_from_sparql"]
