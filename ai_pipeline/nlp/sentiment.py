from transformers import pipeline
from ai_pipeline.config.settings import FINBERT_MODEL

sentiment_model = pipeline(
    "text-classification",
    model=FINBERT_MODEL,
    truncation=True,
    max_length=128,
    device=-1
)

def analyze_sentiment(text_chunks):
    results = sentiment_model(text_chunks)
    scores = [
        r["score"] if r["label"] == "긍정" else -r["score"]
        for r in results
    ]
    return scores
