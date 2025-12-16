import pandas as pd
import ta
import pymongo
from ai_pipeline.config import settings


class PriceFeatureLoader:
    def __init__(self, mongo_uri=settings.MONGO_URI, db_name=settings.MONGO_DB_NAME):
        try:
            self.client = pymongo.MongoClient(mongo_uri)
            self.db = self.client[db_name]

            # 컬렉션 이름 설정
            col_name = getattr(settings, "MONGO_COLLECTION_NAME", "stock_price_1min")
            self.collection = self.db[col_name]

            # [핵심] DB 검색 키 설정 (Java 엔티티 기준: stckShrnIscd)
            self.code_field = getattr(settings, "MONGO_STOCK_CODE_FIELD", "stckShrnIscd")

            print(f" [PriceLoader] DB: {db_name}.{col_name} (Key: {self.code_field})")
        except Exception as e:
            print(f" [PriceLoader Error] 연결 실패: {e}")
            self.collection = None

    def get_latest_technical_features(self, stock_code, lookback=200):
        if self.collection is None: return None

        # 1. DB 조회 (stckShrnIscd 기준)
        query = {self.code_field: str(stock_code)}

        # timestamp 내림차순 정렬 (최신순) -> lookback 개수만큼 가져오기
        cursor = self.collection.find(query, {"_id": 0}).sort("timestamp", -1).limit(lookback)
        df = pd.DataFrame(list(cursor))

        if df.empty:
            # print(f" [Info] 데이터 없음: {stock_code}")
            return None

        # 2. 전처리: 타임스탬프 변환 및 정렬
        # DB의 timestamp가 String일 경우 datetime으로 변환
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # 과거 -> 현재 순으로 재정렬 (지표 계산을 위해 필수)
            df = df.sort_values("timestamp").reset_index(drop=True)
        else:
            # 타임스탬프가 없으면 역순으로라도 뒤집음
            df = df.iloc[::-1].reset_index(drop=True)

        # 3. 컬럼 매핑 (DB 필드명(camelCase) -> 내부 변수명(snake_case))
        # 제공해주신 Java 필드명 기준
        rename_map = {
            'stckPrpr': 'close',  # 현재가
            'acmlVol': 'volume',  # 누적 거래량
            'prdyCtrt': 'prdy_ctrt',  # 전일 대비율
            'acmlTrPbmn': 'acml_tr_pbmn'  # 누적 거래대금
        }
        df = df.rename(columns=rename_map)

        # 4. 필수 컬럼 결측치 처리 (0.0으로 채움)
        required_cols = ['close', 'volume', 'prdy_ctrt']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # 거래대금(acml_tr_pbmn)이 없으면 계산해서라도 채워넣음
        if 'acml_tr_pbmn' not in df.columns:
            df['acml_tr_pbmn'] = df['close'] * df['volume']
        else:
            df['acml_tr_pbmn'] = pd.to_numeric(df['acml_tr_pbmn'], errors='coerce').fillna(0.0)

        # 5. 기술적 지표 계산 (TA-Lib / Pandas)
        try:
            # RSI (14)
            df['hist_RSI_14'] = ta.momentum.rsi(df['close'], window=14).fillna(0)

            # 이동평균(MA) 및 이격도
            for w in [5, 20, 60]:
                ma = ta.trend.sma_indicator(df['close'], window=w).fillna(df['close'])
                # 이격도 계산 방식 통일
                df[f'price_vs_ma{w}'] = (df['close'] - ma) / (ma + 1e-9)
                df[f'hist_Disparity_{w}'] = df['close'] / (ma + 1e-9)

            # 볼린저 밴드 (PctB)
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['hist_BB_PctB'] = bb.bollinger_pband().fillna(0.5)

            # MACD
            macd = ta.trend.MACD(df['close'])
            df['hist_MACD_Diff'] = macd.macd_diff().fillna(0)

            # 거래량 이동평균 비율
            vol_ma20 = ta.trend.sma_indicator(df['volume'], window=20).fillna(df['volume'])
            df['hist_Vol_Ratio'] = df['volume'] / (vol_ma20 + 1e-9)

            # 변동성 및 변화율
            df['price_change_1'] = df['close'].pct_change(1).fillna(0)
            df['price_change_5'] = df['close'].pct_change(5).fillna(0)
            df['price_change_10'] = df['close'].pct_change(10).fillna(0)
            df['tr_amount_change'] = df['acml_tr_pbmn'].pct_change(1).fillna(0)

            df['volatility_5'] = df['price_change_1'].rolling(window=5).std().fillna(0)
            df['volatility_10'] = df['price_change_1'].rolling(window=10).std().fillna(0)

            # 6. 최신(마지막) 데이터 추출
            latest = df.iloc[-1].to_dict()

            # 7. 최종 반환 딕셔너리 (모델 입력용)
            features = {
                'close': latest.get('close', 0),  # 현재가 (결과 확인용)
                'prdy_ctrt': latest.get('prdy_ctrt', 0),
                'price_change_1': latest.get('price_change_1', 0),
                'price_change_5': latest.get('price_change_5', 0),
                'price_change_10': latest.get('price_change_10', 0),
                'price_vs_ma5': latest.get('price_vs_ma5', 0),
                'price_vs_ma20': latest.get('price_vs_ma20', 0),
                'price_vs_ma60': latest.get('price_vs_ma60', 0),
                'tr_amount_change': latest.get('tr_amount_change', 0),
                'volatility_5': latest.get('volatility_5', 0),
                'volatility_10': latest.get('volatility_10', 0),
                'hist_RSI_14': latest.get('hist_RSI_14', 0),
                'hist_Disparity_5': latest.get('hist_Disparity_5', 0),
                'hist_Disparity_20': latest.get('hist_Disparity_20', 0),
                'hist_Disparity_60': latest.get('hist_Disparity_60', 0),
                'hist_BB_PctB': latest.get('hist_BB_PctB', 0),
                'hist_MACD_Diff': latest.get('hist_MACD_Diff', 0),
                'hist_Vol_Ratio': latest.get('hist_Vol_Ratio', 0),
                'timestamp': latest.get('timestamp')
            }
            return features

        except Exception as e:
            print(f" [PriceLoader] 지표 계산 에러: {e}")
            return None