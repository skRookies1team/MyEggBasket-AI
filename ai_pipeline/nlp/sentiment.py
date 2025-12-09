from transformers import pipeline
from ai_pipeline.config.settings import FINBERT_MODEL

# 모델 로드 (한국어 모델)
sentiment_model = pipeline(
    "text-classification",
    model=FINBERT_MODEL,
    truncation=True,
    max_length=512,
    # device=0 # GPU 사용 시 주석 해제
)

def analyze_sentiment(text_chunks):
    """
    텍스트 조각들의 감성을 분석하여 점수 리스트(-1.0 ~ 1.0)를 반환합니다.
    """
    if not text_chunks:
        return []

    results = sentiment_model(text_chunks)
    scores = []

    for r in results:
        label = r['label']
        score = r['score']  # 모델의 확신도 (0~1)
        
        # snunlp/KR-FinBert-SC 모델 라벨 매핑
        # (0: negative, 1: neutral, 2: positive)
        if label == 'positive' or label == 'LABEL_2':
            final_score = score          # 긍정 (+1.0 방향)
        elif label == 'negative' or label == 'LABEL_0':
            final_score = -score         # 부정 (-1.0 방향)
        else: 
            # 'neutral' or 'LABEL_1'
            final_score = 0.0            # 중립 (0점)

        scores.append(final_score)

    return scores