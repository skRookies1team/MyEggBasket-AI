import requests
from bs4 import BeautifulSoup

def fetch_article_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    article = soup.select_one("#dic_area")
    return article.get_text(separator="\n").strip() if article else ""
