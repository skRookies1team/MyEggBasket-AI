import pandas as pd
import numpy as np
import ta
import pymongo
from ai_pipeline.config import settings

class PriceFeatureLoader:
    def __init__(self, mongo_uri=settings.MONGO_URI, db_name=settings.MONGO_DB_NAME):
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        collection_name = getattr(settings, "MONGO_COLLECTION_NAME", "realtime_price")
        self.collection = self.db[collection_name]
        print(f" [PriceLoader] DB: {db_name} / Collection: {collection_name} 연결")

    def get_latest_technical_features(self, stock_code, lookback=200):
        """
        최신 기술적 지표 및 변동성 피처를 계산합니다. (학습 데이터와 동일한 로직 적용)
        """
        # 1. 몽고DB 데이터 조회
        cursor = self.collection.find(
            {"stckShrnIscd": stock_code},
            {"_id": 0}
        ).sort("timestamp", -1).limit(lookback)

        df = pd.DataFrame(list(cursor))

        if df.empty:
            return None

        # 2. 컬럼 매핑 & 전처리
        rename_map = {
            'stckPrpr': 'close',
            'acmlVol': 'volume',
            'acmlTrPbmn': 'acml_tr_pbmn', # 거래대금 (필요 시)
            'prdyCtrt': 'prdy_ctrt'       # 전일대비율
        }
        df = df.rename(columns=rename_map)

        for col in ['close', 'volume', 'prdy_ctrt', 'acml_tr_pbmn']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = 0.0

        if 'timestamp' in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        else:
            df = df.iloc[::-1].reset_index(drop=True)

        # ---------------------------------------------------------
        # 3. 피처 엔지니어링 (RealtimeFeatureLoader 로직 이식)
        # ---------------------------------------------------------
        try:
            # (1) [기존] TA 라이브러리 지표 (9개)
            df['RSI_14'] = ta.momentum.rsi(df['close'], window=14)

            for w in [5, 20, 60]:
                ma = ta.trend.sma_indicator(df['close'], window=w)
                df[f'Disparity_{w}'] = df['close'] / (ma + 1e-9)

            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['BB_PctB'] = bb.bollinger_pband()

            macd = ta.trend.MACD(df['close'])
            df['MACD_Diff'] = macd.macd_diff()

            vol_ma20 = ta.trend.sma_indicator(df['volume'], window=20)
            df['Vol_Ratio'] = df['volume'] / (vol_ma20 + 1e-9)

            # (2) [추가] RealtimeFeatureLoader 자체 계산 지표 (약 10개)
            # - 가격 변화율
            df['price_change_1'] = df['close'].pct_change(1)
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)

            # - 이동평균 대비율 (Disparity와 유사하지만 RealtimeLoader 이름에 맞춤)
            for w in [5, 20, 60]:
                ma = df['close'].rolling(window=w).mean()
                df[f'price_vs_ma{w}'] = (df['close'] - ma) / (ma + 1e-9)

            # - 거래대금 변화율
            if 'acml_tr_pbmn' in df.columns:
                df['tr_amount_change'] = df['acml_tr_pbmn'].pct_change(1)
            else:
                df['tr_amount_change'] = 0.0

            # - 변동성 & 모멘텀
            df['volatility_5'] = df['price_change_1'].rolling(window=5).std()
            df['volatility_10'] = df['price_change_1'].rolling(window=10).std()

            # (참고: momentum은 price_change와 유사하여 생략되기도 했으나 필요하면 추가)
            # df['momentum_5'] = df['close'] - df['close'].shift(5)

            # 4. 최신 행 추출 및 반환
            latest = df.iloc[-1].to_dict()

            # RealtimeFeatureLoader의 컬럼명과 최대한 매칭
            features = {
                'close': latest['close'],
                'volume': latest['volume'],
                # FeatureExpander 출신 (TA)
                'hist_RSI_14': latest.get('RSI_14', 0),
                'hist_Disparity_5': latest.get('Disparity_5', 0),
                'hist_Disparity_20': latest.get('Disparity_20', 0),
                'hist_Disparity_60': latest.get('Disparity_60', 0),
                'hist_BB_PctB': latest.get('BB_PctB', 0),
                'hist_MACD_Diff': latest.get('MACD_Diff', 0),
                'hist_Vol_Ratio': latest.get('Vol_Ratio', 0),

                # RealtimeFeatureLoader 출신
                'prdy_ctrt': latest.get('prdy_ctrt', 0),
                'price_change_1': latest.get('price_change_1', 0),
                'price_change_5': latest.get('price_change_5', 0),
                'price_change_10': latest.get('price_change_10', 0),
                'price_vs_ma5': latest.get('price_vs_ma5', 0),
                'price_vs_ma20': latest.get('price_vs_ma20', 0),
                'price_vs_ma60': latest.get('price_vs_ma60', 0), # 60일선 추가
                'tr_amount_change': latest.get('tr_amount_change', 0),
                'volatility_5': latest.get('volatility_5', 0),
                'volatility_10': latest.get('volatility_10', 0)
            }

            return {k: (0.0 if pd.isna(v) else v) for k, v in features.items()}

        except Exception as e:
            print(f" [PriceLoader] 지표 계산 중 에러: {e}")
            return None