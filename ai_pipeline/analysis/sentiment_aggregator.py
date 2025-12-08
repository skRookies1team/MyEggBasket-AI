import pandas as pd
import numpy as np

def calculate_aggregated_features(news_data_list):
    if not news_data_list:
        return pd.DataFrame()

    df = pd.DataFrame(news_data_list)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    results = []

    for code, group in df.groupby('code'):
        group.set_index('date', inplace=True)
        
        # 1. 1시간 단위 평균 (빈 시간은 앞의 값으로 채움: ffill)
        hourly_sentiment = group['score'].resample('1h').mean()
        
        # 2. 시간 가중 감성 점수 (Decay)
        decay_sentiment = group['score'].ewm(span=6).mean()

        # 3. 변동성
        sentiment_volatility = group['score'].rolling(window='6h').std().fillna(0)
        
        # [버그 수정] 정확한 시간이 아니라, '가장 마지막에 계산된 값'을 가져와야 함
        latest_idx = group.index[-1] # 뉴스의 실제 마지막 시간
        
        # Resample된 데이터 중 가장 마지막 유효값 가져오기
        # (마지막 뉴스가 10:15에 있었으면, 10:00 구간의 값을 가져와야 함)
        if not hourly_sentiment.empty:
            last_hourly_val = hourly_sentiment.iloc[-1]
            if pd.isna(last_hourly_val): # 혹시 NaN이면 0
                last_hourly_val = 0.0
        else:
            last_hourly_val = 0.0

        results.append({
            'code': code,
            'last_updated': latest_idx,
            'sentiment_1h': last_hourly_val, # [수정됨]
            'sentiment_decay': decay_sentiment.iloc[-1], # 마지막 값
            'sentiment_volatility': sentiment_volatility.iloc[-1], # 마지막 값
            'news_count': group['score'].rolling('24h').count().iloc[-1] # 최근 24시간 뉴스 수
        })

    return pd.DataFrame(results)