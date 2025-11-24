from elasticsearch import Elasticsearch
from ai_pipeline.config.settings import ES_HOST

es = Elasticsearch(ES_HOST)

def save_news_to_es(url, text, chunks, sentiments):
    doc = {
        "url": url,
        "text": text,
        "chunks": chunks,
        "sentiments": sentiments
    }
    es.index(index="news_articles", document=doc)
