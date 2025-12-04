import pandas as pd
import numpy as np

def calculate_aggregated_features(news_data_list):
    """
    news_data_list 예시:
    [
        {'date': '2025-10-01 10:00', 'code': '005930', 'score': 0.8},
        {'date': '2025-10-01 10:30', 'code': '005930', 'score': 0.5},
        {'date': '2025-10-01 09:00', 'code': '000660', 'score': -0.2},
    ]
    """
    if not news_data_list:
        return pd.DataFrame()

    df = pd.DataFrame(news_data_list)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # 결과를 저장할 리스트
    results = []

    # 종목별로 그룹화하여 처리
    for code, group in df.groupby('code'):
        # 시간 단위 리샘플링을 위해 date를 인덱스로 설정
        group.set_index('date', inplace=True)
        
        # 1. 1시간 단위 단순 평균 (News Sentiment 1H)
        # 1H 단위로 묶어서 평균 계산 -> 결측치는 앞의 값으로 채우거나 0 처리
        hourly_sentiment = group['score'].resample('1h').mean().fillna(0)

        # 2. 시간 가중 감성 점수 (Time-Decay)
        # 최신 뉴스일수록 가중치를 더 줌 (Exponential Weighted Moving Average)
        # span=6 : 대략 최근 6개의 데이터(혹은 시간 구간)에 큰 비중
        decay_sentiment = group['score'].ewm(span=6).mean()

        # 3. 감성 변동성 (Volatility)
        # 감성 점수가 얼마나 들쭉날쭉한지 (표준편차) -> 리스크 지표로 활용
        sentiment_volatility = group['score'].rolling(window='6h').std().fillna(0)

        # 데이터 합치기 (가장 최근 시점 기준의 지표 추출 예시)
        # 실제로는 시계열 전체를 DB에 넣겠지만, 여기선 최신 상태값 예시
        latest_idx = group.index[-1]
        
        results.append({
            'code': code,
            'last_updated': latest_idx,
            'sentiment_1h': hourly_sentiment.loc[latest_idx] if latest_idx in hourly_sentiment else 0,
            'sentiment_decay': decay_sentiment.loc[latest_idx],
            'sentiment_volatility': sentiment_volatility.loc[latest_idx],
            'news_count': group['score'].rolling('6h').count().loc[latest_idx] # ✅ 이름을 'news_count'로 통일
        })

    return pd.DataFrame(results)