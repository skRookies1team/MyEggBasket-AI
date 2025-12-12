import pandas as pd
import numpy as np
from datetime import datetime
import os


class RealtimeFeatureLoader:
    """
    실시간 체결 정보 CSV를 로드하고 머신러닝 피쳐로 변환
    (1분 단위 리샘플링 적용, 언더스코어 없는 컬럼명 호환 지원)
    """

    # [중요] 필수 컬럼 목록을 최소화했습니다 (이게 적용되어야 코드가 수정된 것입니다)
    REQUIRED_COLUMNS = [
        'timestamp', 'stock_code', 'stck_prpr', 'askp1', 'bidp1'
    ]

    def __init__(self, csv_file_path):
        print(f" [Loader] 초기화됨 (New Version Check): {csv_file_path}")  # 수정 여부 확인용 로그
        self.csv_path = csv_file_path

        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f" CSV 파일을 찾을 수 없습니다: {csv_file_path}")

    def load_and_preprocess(self):
        """CSV 파일 로드 및 전처리 (유연한 컬럼 처리)"""
        print("\n 체결 정보 CSV 로딩 중...")

        # CSV 로드
        try:
            df = pd.read_csv(self.csv_path, sep=',', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.csv_path, sep=',', encoding='cp949')

        # 1. 컬럼명 표준화 (소문자 변환 및 공백 제거)
        df.columns = df.columns.str.strip().str.lower()

        # 2. 컬럼 매핑 (로그에 나온 '실제 컬럼'들 완벽 대응)
        column_mapping = {
            # 종목코드
            'stck_shrn_iscd': 'stock_code', 'stckshrniscd': 'stock_code',
            'code': 'stock_code', '종목코드': 'stock_code',

            # 현재가
            'stck_prpr': 'stck_prpr', 'stckprpr': 'stck_prpr', '현재가': 'stck_prpr',

            # 호가
            'askp1': 'askp1', '매도1호가': 'askp1',
            'bidp1': 'bidp1', '매수1호가': 'bidp1',
            'total_askp_rsqn': 'total_askp_rsqn', 'totalaskprsqn': 'total_askp_rsqn',
            'total_bidp_rsqn': 'total_bidp_rsqn', 'totalbidprsqn': 'total_bidp_rsqn',

            # 체결량/강도
            'seln_cntg_csnu': 'seln_cntg_csnu', 'selncntgcsnu': 'seln_cntg_csnu',
            'shnu_cntg_csnu': 'shnu_cntg_csnu', 'shnucntgcsnu': 'shnu_cntg_csnu',

            # 거래대금/변동률
            'acml_tr_pbmn': 'acml_tr_pbmn', 'acmltrpbmn': 'acml_tr_pbmn', 'acmlvol': 'acml_vol',
            'prdy_vrss': 'prdy_vrss', 'prdyvrss': 'prdy_vrss',
            'prdy_ctrt': 'prdy_ctrt', 'prdyctrt': 'prdy_ctrt',

            # 시간
            'stck_cntg_hour': 'stck_cntg_hour', 'stckcntghour': 'stck_cntg_hour'
        }
        df = df.rename(columns=column_mapping)

        # 3. stock_code 확인
        if 'stock_code' not in df.columns:
            # 매핑되지 않은 경우 한 번 더 찾기
            for possible_name in ['stck_shrn_iscd', '종목코드', 'code', 'stckshrniscd']:
                if possible_name in df.columns:
                    df = df.rename(columns={possible_name: 'stock_code'})
                    break

        if 'stock_code' in df.columns:
            df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
        else:
            print(f" [DEBUG] 현재 컬럼 목록: {df.columns.tolist()}")
            raise KeyError(f" 'stock_code' 컬럼을 찾을 수 없습니다.")

        # 3.1 timestamp 컬럼 유연 처리: 다양한 이름을 허용하고, 없으면 stck_cntg_hour + 파일명(날짜)로 생성 시도
        timestamp_candidates = ['timestamp', 'time', 'trade_time', 'tr_time', '체결시각', '체결시간', '체결일시', 'stck_cntg_hour']
        found_ts = None
        for c in timestamp_candidates:
            if c in df.columns:
                found_ts = c
                break

        if found_ts and found_ts != 'timestamp':
            df = df.rename(columns={found_ts: 'timestamp'})

        if 'timestamp' not in df.columns:
            # stck_cntg_hour가 있다면 파일명에서 날짜를 추출해 조합 시도
            if 'stck_cntg_hour' in df.columns:
                base = os.path.basename(self.csv_path)
                date_part = os.path.splitext(base)[0]
                # 파일명이 YYYYMMDD 형태라면 시도
                if len(date_part) == 8 and date_part.isdigit():
                    # stck_cntg_hour 값이 HHMMSS 또는 HMMSS 형식일 수 있음
                    hrs = df['stck_cntg_hour'].astype(str).fillna('0')
                    ts_list = []
                    for h in hrs:
                        hstr = h.zfill(6)
                        try:
                            ts = pd.to_datetime(date_part + hstr, format='%Y%m%d%H%M%S', errors='coerce')
                        except Exception:
                            ts = pd.NaT
                        ts_list.append(ts)
                    df['timestamp'] = pd.to_datetime(ts_list)
                else:
                    # 날짜 정보가 없으면 인덱스 기반 임의 timestamp 생성 (경고 출력)
                    print(" [WARN] CSV 파일명에서 날짜를 찾지 못해 인덱스 기반 timestamp 생성(정확하지 않을 수 있음)")
                    df['timestamp'] = pd.to_datetime(df.index, unit='s', origin='unix')
            else:
                print(f" [DEBUG] 현재 컬럼 목록: {df.columns.tolist()}")
                raise KeyError(" 'timestamp' 컬럼을 찾을 수 없습니다. 가능한 대체 컬럼(예: 'stck_cntg_hour')이 없거나 파싱에 실패했습니다.")

        # 4. 필수 및 기술적 지표용 컬럼 결측 처리 (0으로 채움)
        # 이 컬럼들은 없으면 0으로 채워서 계산 오류 방지
        target_cols = [
            'stck_prpr', 'askp1', 'bidp1',
            'acml_tr_pbmn', 'total_askp_rsqn', 'total_bidp_rsqn',
            'seln_cntg_csnu', 'shnu_cntg_csnu', 'prdy_ctrt'
        ]

        for col in target_cols:
            if col not in df.columns:
                # print(f" ⚠️ 컬럼 부재: '{col}' -> 0으로 채움")
                df[col] = 0.0
            else:
                # 숫자로 변환 (오류 발생 시 0)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 5. 가격 데이터 보정 (현재가가 0인 경우 호가로 대체)
        mask_zero_price = (df['stck_prpr'] == 0)
        if mask_zero_price.any():
            df.loc[mask_zero_price, 'stck_prpr'] = df.loc[mask_zero_price, 'askp1']
            # 그래도 0이면 bidp1
            mask_still_zero = (df['stck_prpr'] == 0)
            df.loc[mask_still_zero, 'stck_prpr'] = df.loc[mask_still_zero, 'bidp1']

        # 6. timestamp 변환
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            print(" timestamp 변환 실패. 기본 형식으로 진행합니다.")

        # 정렬
        df = df.sort_values(['stock_code', 'timestamp']).reset_index(drop=True)

        return df

    def create_technical_features(self, df):
        """기술적 지표 생성 (1분 단위 리샘플링 적용)"""

        result_dfs = []

        # 리샘플링을 위해 인덱스 설정
        df = df.set_index('timestamp')

        for stock_code, group in df.groupby('stock_code'):
            # 1분봉 데이터로 리샘플링
            # numeric_only=True로 수치형 데이터만 집계
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            resampled = group[numeric_cols].resample('1min').last()

            # 빈 구간은 앞의 값으로 채움
            resampled = resampled.ffill()

            # stock_code 복원
            resampled['stock_code'] = stock_code

            # --- 지표 계산 (1분 기준) ---

            # 1. 가격 변화율
            resampled['price_change_1'] = resampled['stck_prpr'].pct_change(1)
            resampled['price_change_5'] = resampled['stck_prpr'].pct_change(5)
            resampled['price_change_10'] = resampled['stck_prpr'].pct_change(10)

            # 2. 이동평균
            resampled['ma_5'] = resampled['stck_prpr'].rolling(window=5, min_periods=1).mean()
            resampled['ma_10'] = resampled['stck_prpr'].rolling(window=10, min_periods=1).mean()
            resampled['ma_20'] = resampled['stck_prpr'].rolling(window=20, min_periods=1).mean()

            # 3. 이격도
            resampled['price_vs_ma5'] = (resampled['stck_prpr'] - resampled['ma_5']) / (resampled['ma_5'] + 1e-8)
            resampled['price_vs_ma20'] = (resampled['stck_prpr'] - resampled['ma_20']) / (resampled['ma_20'] + 1e-8)

            # 4. 타겟 생성 (5분 뒤 가격 예측)
            LOOK_AHEAD = 5
            THRESHOLD = 0.003  # 0.3%

            resampled['future_price'] = resampled['stck_prpr'].shift(-LOOK_AHEAD)
            resampled['return'] = (resampled['future_price'] - resampled['stck_prpr']) / (resampled['stck_prpr'] + 1e-8)
            resampled['target'] = (resampled['return'] > THRESHOLD).astype(int)

            # 5. 거래대금 변화율
            resampled['tr_amount_change'] = resampled['acml_tr_pbmn'].pct_change(1)

            # 6. 스프레드
            resampled['spread'] = (resampled['askp1'] - resampled['bidp1']) / (resampled['stck_prpr'] + 1e-8)
            resampled['spread_pct'] = resampled['spread'] * 100

            # 7. 매수/매도 압력
            total_orders = resampled['total_askp_rsqn'] + resampled['total_bidp_rsqn'] + 1e-8
            resampled['buy_pressure'] = resampled['total_bidp_rsqn'] / total_orders

            # 8. 체결 강도
            total_cntg = resampled['seln_cntg_csnu'] + resampled['shnu_cntg_csnu'] + 1e-8
            resampled['buy_strength'] = resampled['shnu_cntg_csnu'] / total_cntg

            # 9. 변동성
            resampled['volatility_5'] = resampled['price_change_1'].rolling(window=5, min_periods=1).std()
            resampled['volatility_10'] = resampled['price_change_1'].rolling(window=10, min_periods=1).std()

            # 10. 모멘텀
            resampled['momentum_5'] = resampled['stck_prpr'] - resampled['stck_prpr'].shift(5)
            resampled['momentum_10'] = resampled['stck_prpr'] - resampled['stck_prpr'].shift(10)

            # 전일 대비율이 없다면 0으로 처리
            if 'prdy_ctrt' not in resampled.columns:
                resampled['prdy_ctrt'] = 0.0

            result_dfs.append(resampled.reset_index())

        if not result_dfs:
            return pd.DataFrame()

        final_df = pd.concat(result_dfs, ignore_index=True)
        return final_df

    def prepare_features(self):
        """최종 피쳐 데이터 준비"""
        # 1. 로드 및 전처리
        df = self.load_and_preprocess()

        if df.empty:
            return None, None, None

        # 2. 기술적 지표 추가
        df = self.create_technical_features(df)

        if df.empty:
            return None, None, None

        # 3. NaN 제거
        df = df.dropna()

        if len(df) == 0:
            return None, None, None

        # 4. 피쳐 선택
        feature_cols = [
            'prdy_ctrt',
            'price_change_1', 'price_change_5', 'price_change_10',
            'price_vs_ma5', 'price_vs_ma20',
            'tr_amount_change',
            'spread', 'spread_pct',
            'buy_pressure',
            'buy_strength',
            'volatility_5', 'volatility_10',
            'momentum_5', 'momentum_10'
        ]

        # 없는 컬럼은 0으로 채워서라도 리턴
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0

        X = df[feature_cols].copy()
        y = df['target'].copy()
        stock_codes = df['stock_code'].copy()

        return X, y, stock_codes


if __name__ == "__main__":
    pass