import re
from bs4 import BeautifulSoup

def get_urls_from_html(text_in_html):
    soup = BeautifulSoup(text_in_html, "html.parser")
    urls = []
    for a in soup.find_all("a"):
        href = a.get("href", "").strip()
        if "mailto:" not in href:
            urls.append(href)
    return urls

def transform_html_in_text(text_with_html):
    soup = BeautifulSoup(text_with_html, "html.parser")
    for a in soup.find_all("a"):
        href = a.get("href", "").strip()
        label = a.get_text()
        a.replace_with(f"{label} ({href})" if href else label)
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

