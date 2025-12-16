import pandas as pd
import numpy as np
import os


class RealtimeFeatureLoader:
    """
    1년치 분봉 데이터(CSV)를 로드하여 머신러닝 피쳐로 변환
    - 초 단위 데이터 처리 로직 삭제됨
    - Date, Time, Close, Volume 형식의 분봉 데이터 전용
    """

    def __init__(self, csv_file_path):
        self.csv_path = csv_file_path
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f" CSV 파일을 찾을 수 없습니다: {csv_file_path}")

    def load_and_preprocess(self):
        """CSV 파일 로드 및 전처리"""

        # 1. CSV 로드
        try:
            df = pd.read_csv(self.csv_path, sep=',', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.csv_path, sep=',', encoding='cp949')

        # 2. 컬럼명 표준화 (소문자 변환 및 공백 제거)
        df.columns = df.columns.str.strip().str.lower()

        # 원본 컬럼명 매핑 (Date, Time 등 대소문자 섞여있을 경우 대비)
        # 소문자로 변환된 컬럼명을 기준으로 처리

        # 필수 컬럼 체크 (date, time, close)
        if not all(col in df.columns for col in ['date', 'time', 'close']):
            # 분봉 데이터 형식이 아니면 빈 DF 반환
            return pd.DataFrame()

        # 3. 종목코드 추출 (파일명: 000270_1Year.csv -> 000270)
        filename = os.path.basename(self.csv_path)
        stock_code = filename.split('_')[0]
        if stock_code.isdigit():
            df['stock_code'] = stock_code.zfill(6)
        else:
            return pd.DataFrame()

        # 4. Timestamp 생성 (Date + Time)
        # Date: 20241210, Time: 141500 (HHMMSS)
        try:
            df['date_str'] = df['date'].astype(str)
            df['time_str'] = df['time'].astype(str).str.zfill(6)

            df['timestamp'] = pd.to_datetime(
                df['date_str'] + df['time_str'],
                format='%Y%m%d%H%M%S',
                errors='coerce'
            )
        except Exception as e:
            print(f" [Error] 날짜 변환 실패: {e}")
            return pd.DataFrame()

        # 5. 컬럼 매핑 및 정리
        rename_map = {
            'close': 'stck_prpr',  # 현재가
            'volume': 'acml_vol'  # 거래량
        }
        df = df.rename(columns=rename_map)

        # 거래대금 추정 (가격 * 거래량)
        df['acml_tr_pbmn'] = df['stck_prpr'] * df['acml_vol']

        # 필수 컬럼 수치형 변환
        for col in ['stck_prpr', 'acml_tr_pbmn']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 결측 및 정렬
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values(['stock_code', 'timestamp']).reset_index(drop=True)

        # 필요한 컬럼만 선택
        keep_cols = ['timestamp', 'stock_code', 'stck_prpr', 'acml_tr_pbmn', 'prdy_ctrt']
        df = df[[c for c in keep_cols if c in df.columns]]

        return df

    def create_technical_features(self, df):
        """기술적 지표 생성"""
        if df.empty:
            return pd.DataFrame()

        result_dfs = []
        df = df.set_index('timestamp')

        for stock_code, group in df.groupby('stock_code'):
            # 1분봉 리샘플링 (빈 시간 채우기 & 분 단위 정렬 보장)
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            resampled = group[numeric_cols].resample('1min').last().ffill()
            resampled['stock_code'] = stock_code

            # --- 지표 계산 ---
            # 1. 가격 변화율
            resampled['price_change_1'] = resampled['stck_prpr'].pct_change(1).fillna(0)
            resampled['price_change_5'] = resampled['stck_prpr'].pct_change(5).fillna(0)
            resampled['price_change_10'] = resampled['stck_prpr'].pct_change(10).fillna(0)

            # 2. 이동평균
            resampled['ma_5'] = resampled['stck_prpr'].rolling(window=5).mean()
            resampled['ma_20'] = resampled['stck_prpr'].rolling(window=20).mean()
            resampled['ma_60'] = resampled['stck_prpr'].rolling(window=60).mean()

            # 3. 이격도
            resampled['price_vs_ma5'] = (resampled['stck_prpr'] - resampled['ma_5']) / (resampled['ma_5'] + 1e-8)
            resampled['price_vs_ma20'] = (resampled['stck_prpr'] - resampled['ma_20']) / (resampled['ma_20'] + 1e-8)
            resampled['price_vs_ma60'] = (resampled['stck_prpr'] - resampled['ma_60']) / (resampled['ma_60'] + 1e-8)

            # 4. 거래대금 변화율
            if 'acml_tr_pbmn' in resampled.columns:
                resampled['tr_amount_change'] = resampled['acml_tr_pbmn'].pct_change(1).fillna(0)
            else:
                resampled['tr_amount_change'] = 0.0

            # 5. 변동성
            resampled['volatility_5'] = resampled['price_change_1'].rolling(window=5).std().fillna(0)
            resampled['volatility_10'] = resampled['price_change_1'].rolling(window=10).std().fillna(0)

            # 6. 타겟 생성 (5분 뒤 가격 예측)
            LOOK_AHEAD = 5
            THRESHOLD = 0.003
            resampled['future_price'] = resampled['stck_prpr'].shift(-LOOK_AHEAD)
            resampled['return'] = (resampled['future_price'] - resampled['stck_prpr']) / (resampled['stck_prpr'] + 1e-8)
            resampled['target'] = (resampled['return'] >= THRESHOLD).astype(int)

            if 'prdy_ctrt' not in resampled.columns:
                resampled['prdy_ctrt'] = 0.0

            result_dfs.append(resampled.reset_index())

        if not result_dfs:
            return pd.DataFrame()

        final_df = pd.concat(result_dfs, ignore_index=True)
        return final_df

    def prepare_features(self):
        """최종 피쳐 데이터 준비"""
        # 1. 로드
        df = self.load_and_preprocess()
        if df is None or df.empty:
            return None, None, None

        # 2. 지표 생성
        df = self.create_technical_features(df)
        if df.empty:
            return None, None, None

        # 3. 결측치 제거
        df = df.dropna()
        if len(df) == 0:
            return None, None, None

        # 4. 피쳐 선택
        # 학습에 필요한 컬럼 + 식별자(timestamp, stock_code) 포함
        feature_cols = [
            'timestamp',  # 시계열 매핑용 필수
            'prdy_ctrt',
            'price_change_1', 'price_change_5', 'price_change_10',
            'price_vs_ma5', 'price_vs_ma20', 'price_vs_ma60',
            'tr_amount_change',
            'volatility_5', 'volatility_10'
        ]

        valid_cols = [c for c in feature_cols if c in df.columns]

        # 없는 수치형 컬럼 0.0 처리 (timestamp 제외)
        for c in feature_cols:
            if c not in df.columns and c != 'timestamp':
                df[c] = 0.0

        X = df[valid_cols].copy()
        y = df['target'].copy()
        stock_codes = df['stock_code'].copy()

        return X, y, stock_codes