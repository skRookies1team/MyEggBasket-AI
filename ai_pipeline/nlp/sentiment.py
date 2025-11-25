from transformers import pipeline
from ai_pipeline.config.settings import FINBERT_MODEL

sentiment_model = pipeline(
    "text-classification",
    model=FINBERT_MODEL,
    truncation=True,
    max_length=512,
    device=-1
)

def analyze_sentiment(text_chunks):
    if not text_chunks:
        return []

    results = sentiment_model(text_chunks)
    scores = []

    for r in results:
        label = r['label']  # 예: 'positive', 'neutral', 'negative'
        score = r['score']  # 확신도 (0~1)

    
        if label == 'positive':
            # 긍정: 그대로 양수 
            final_score = score
            
        elif label == 'negative':
            # 부정: 음수로 변환 
            final_score = -score
            
        elif label == 'neutral':
            # 중립: 0점으로 처리 (혹은 0에 가깝게)
            # 확실한 중립이면 0을 주거나, 약간의 노이즈만 남김
            final_score = 0.0
            
        else:
            # 혹시 모를 다른 라벨(LABEL_0 등) 대비
            # 일단 중립 처리
            final_score = 0.0

        scores.append(final_score)

    return scores

