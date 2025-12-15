import pandas as pd
import ta
import pymongo
from datetime import datetime
from ai_pipeline.config.settings import MONGO_URI

class PriceFeatureLoader:
    def __init__(self, MONGO_URI, db_name="stock_db"):
        self.client = pymongo.MongoClient(MONGO_URI)
        self.db = self.client[db_name]
        self.collection = self.db["min_candles"] # 분봉 컬렉션 이름 가정

    def get_latest_technical_features(self, stock_code, lookback=200):
        """
        최신 기술적 지표를 계산하기 위해 최근 N개의 분봉을 가져옵니다.
        """
        # 1. 몽고DB에서 최근 데이터 조회 (TA 계산을 위해 과거 데이터도 필요)
        cursor = self.collection.find(
            {"stock_code": stock_code},
            {"_id": 0, "timestamp": 1, "close": 1, "volume": 1, "open": 1, "high": 1, "low": 1}
        ).sort("timestamp", -1).limit(lookback)

        df = pd.DataFrame(list(cursor))

        if df.empty:
            return None

        # 시간순 정렬 (과거 -> 현재)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 2. 기술적 지표 실시간 계산 (FeatureExpander 로직과 동일하게)
        # (A) RSI
        df['RSI_14'] = ta.momentum.rsi(df['close'], window=14)

        # (B) 이동평균 및 이격도
        for w in [5, 20, 60]:
            ma = ta.trend.sma_indicator(df['close'], window=w)
            df[f'Disparity_{w}'] = df['close'] / (ma + 1e-9)

        # (C) 볼린저 밴드
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['BB_PctB'] = bb.bollinger_pband()

        # (D) MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD_Diff'] = macd.macd_diff()

        # (E) 거래량 비율
        vol_ma20 = ta.trend.sma_indicator(df['volume'], window=20)
        df['Vol_Ratio'] = df['volume'] / (vol_ma20 + 1e-9)

        # 3. 가장 최신(마지막 행) 데이터만 추출
        latest = df.iloc[-1].to_dict()
        
        # 필요한 컬럼만 리턴
        features = {
            'close': latest['close'],
            'volume': latest['volume'],
            'hist_RSI_14': latest['RSI_14'],
            'hist_Disparity_5': latest['Disparity_5'],
            'hist_Disparity_20': latest['Disparity_20'],
            'hist_Disparity_60': latest['Disparity_60'],
            'hist_BB_PctB': latest['BB_PctB'],
            'hist_MACD_Diff': latest['MACD_Diff'],
            'hist_Vol_Ratio': latest['Vol_Ratio']
        }
        
        # NaN 처리
        return {k: (0.0 if pd.isna(v) else v) for k, v in features.items()}