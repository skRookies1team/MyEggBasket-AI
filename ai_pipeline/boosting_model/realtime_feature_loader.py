# python
import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob


class RealtimeFeatureLoader:
    """
    실시간 체결 정보 CSV를 로드하고 머신러닝 피쳐로 변환
    (1분 단위 리샘플링 적용)
    - 생성자에 파일 경로, 디렉토리, 또는 glob 패턴을 전달 가능
    """

    REQUIRED_COLUMNS = [
        'timestamp', 'stock_code', 'stck_cntg_hour', 'stck_prpr',
        'prdy_vrss', 'prdy_ctrt', 'acml_tr_pbmn',
        'seln_cntg_csnu', 'shnu_cntg_csnu', 'askp1', 'bidp1',
        'total_askp_rsqn', 'total_bidp_rsqn'
    ]

    def __init__(self, csv_path_or_dir):
        self.input_path = csv_path_or_dir

        # 디렉토리인 경우 해당 폴더의 모든 .csv를 사용
        if os.path.isdir(self.input_path):
            files = sorted(glob.glob(os.path.join(self.input_path, "*.csv")))
            if not files:
                raise FileNotFoundError(f" 디렉토리에 CSV 파일이 없습니다: `{self.input_path}`")
            self.csv_files = files
            print(f" CSV 디렉토리 확인: `{self.input_path}` → {len(files)} files")
            return

        # glob 패턴 또는 단일 파일 처리
        matched = sorted(glob.glob(self.input_path))
        if matched:
            # glob이 하나 이상의 파일을 찾음
            self.csv_files = matched
            print(f" 입력 경로 매칭 파일 수: {len(matched)}")
        else:
            # 직접 파일 경로가 없으면 예외
            raise FileNotFoundError(f" CSV 파일/패턴을 찾을 수 없습니다: `{self.input_path}`")

    def load_and_preprocess(self):
        """CSV 파일들 로드 및 전처리 (여러 파일 병합 지원)"""
        print("\n 체결 정보 CSV 로딩 중...")
        df_list = []

        for fpath in self.csv_files:
            try:
                df = pd.read_csv(fpath, sep=',', encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(fpath, sep=',', encoding='cp949')

            df.columns = df.columns.str.strip().str.lower()

            column_mapping = {
                'stck_shrn_iscd': 'stock_code',
                'code': 'stock_code',
                '종목코드': 'stock_code'
            }
            df = df.rename(columns=column_mapping)

            # stock_code 처리 (가능한 컬럼명으로 보완)
            if 'stock_code' not in df.columns:
                for possible_name in ['stck_shrn_iscd', '종목코드', 'code']:
                    if possible_name in df.columns:
                        df = df.rename(columns={possible_name: 'stock_code'})
                        break

            # 종목 코드 6자리로 포맷
            if 'stock_code' in df.columns:
                df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)

            df_list.append(df)

        if not df_list:
            raise FileNotFoundError("읽을 CSV가 없습니다.")

        # 파일 병합
        df = pd.concat(df_list, ignore_index=True)

        # 필수 컬럼 체크
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f" 필수 컬럼 누락: {missing_cols}")
            print(f"   실제 컬럼: {df.columns.tolist()}")
            raise KeyError(f"필수 컬럼이 없습니다: {missing_cols}")

        # timestamp 변환
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            print(" timestamp 변환 실패. 기본 형식으로 진행합니다.")

        df = df.sort_values(['stock_code', 'timestamp']).reset_index(drop=True)

        print(f" 전처리 완료")
        print(f"   종목 수: {df['stock_code'].nunique()}개")
        print(f"   데이터 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

        return df

    def create_technical_features(self, df):
        """기술적 지표 생성 (1분 단위 리샘플링 적용)"""
        print("\n 기술적 지표 계산 중 (1분봉 변환)...")

        result_dfs = []
        df = df.set_index('timestamp')

        for stock_code, group in df.groupby('stock_code'):
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            resampled = group[numeric_cols].resample('1min').last()
            resampled = resampled.ffill()
            resampled['stock_code'] = stock_code

            resampled['price_change_1'] = resampled['stck_prpr'].pct_change(1)
            resampled['price_change_5'] = resampled['stck_prpr'].pct_change(5)
            resampled['price_change_10'] = resampled['stck_prpr'].pct_change(10)

            resampled['ma_5'] = resampled['stck_prpr'].rolling(window=5, min_periods=1).mean()
            resampled['ma_10'] = resampled['stck_prpr'].rolling(window=10, min_periods=1).mean()
            resampled['ma_20'] = resampled['stck_prpr'].rolling(window=20, min_periods=1).mean()

            resampled['price_vs_ma5'] = (resampled['stck_prpr'] - resampled['ma_5']) / (resampled['ma_5'] + 1e-8)
            resampled['price_vs_ma20'] = (resampled['stck_prpr'] - resampled['ma_20']) / (resampled['ma_20'] + 1e-8)

            LOOK_AHEAD = 5
            THRESHOLD = 0.003
            resampled['future_price'] = resampled['stck_prpr'].shift(-LOOK_AHEAD)
            resampled['return'] = (resampled['future_price'] - resampled['stck_prpr']) / resampled['stck_prpr']
            resampled['target'] = (resampled['return'] > THRESHOLD).astype(int)

            resampled['tr_amount_change'] = resampled['acml_tr_pbmn'].pct_change(1)
            resampled['spread'] = (resampled['askp1'] - resampled['bidp1']) / (resampled['stck_prpr'] + 1e-8)
            resampled['spread_pct'] = resampled['spread'] * 100

            total_orders = resampled['total_askp_rsqn'] + resampled['total_bidp_rsqn'] + 1e-8
            resampled['buy_pressure'] = resampled['total_bidp_rsqn'] / total_orders

            total_cntg = resampled['seln_cntg_csnu'] + resampled['shnu_cntg_csnu'] + 1e-8
            resampled['buy_strength'] = resampled['shnu_cntg_csnu'] / total_cntg

            resampled['volatility_5'] = resampled['price_change_1'].rolling(window=5, min_periods=1).std()
            resampled['volatility_10'] = resampled['price_change_1'].rolling(window=10, min_periods=1).std()

            resampled['momentum_5'] = resampled['stck_prpr'] - resampled['stck_prpr'].shift(5)
            resampled['momentum_10'] = resampled['stck_prpr'] - resampled['stck_prpr'].shift(10)

            result_dfs.append(resampled.reset_index())

        final_df = pd.concat(result_dfs, ignore_index=True)
        print(f" 기술적 지표 추가 완료 (데이터 크기: {len(final_df)} rows)")

        return final_df

    def prepare_features(self):
        df = self.load_and_preprocess()
        df = self.create_technical_features(df)

        original_len = len(df)
        df = df.dropna()
        if len(df) == 0:
            print(" 유효한 데이터가 없습니다.")
            return None, None, None

        print(f"\n NaN 제거: {original_len:,} → {len(df):,} ({len(df) / original_len * 100:.1f}%)")

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

        X = df[feature_cols].copy()
        y = df['target'].copy()
        stock_codes = df['stock_code'].copy()

        print(f"\n 최종 데이터 준비 완료!")
        print(f"   샘플 수: {len(X):,}")
        print(f"   피쳐 수: {len(feature_cols)}")
        print(f"   상승(1): {(y == 1).sum():,}개 ({(y == 1).sum() / len(y) * 100:.1f}%)")
        print(f"   하락(0): {(y == 0).sum():,}개 ({(y == 0).sum() / len(y) * 100:.1f}%)")

        return X, y, stock_codes


if __name__ == "__main__":
    # 예: 폴더로 여러 날짜 CSV를 넣어 사용
    csv_input = "./data/csvs"  # 또는 "./data/*.csv" 같은 glob 패턴도 가능
    if os.path.exists(csv_input) or glob.glob(csv_input):
        loader = RealtimeFeatureLoader(csv_input)
        X, y, stock_codes = loader.prepare_features()
        if X is not None:
            print("\n[피쳐 샘플]")
            print(X.head())
            print(f"\n[타겟 분포]")
            print(y.value_counts())
    else:
        print("테스트용 파일/폴더가 없습니다.")