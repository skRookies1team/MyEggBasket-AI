import pandas as pd
import numpy as np
from datetime import datetime
import os


class RealtimeFeatureLoader:
    """
    실시간(또는 CSV 기반) 체결 데이터 로더 및 피처 생성기

    역할 요약:
    - CSV 파일(틱/분 단위 체결 데이터)을 로드하고 기본 전처리를 수행합니다.
    - 종목별로 기술적 지표(변동성, 이동평균, 모멘텀 등)를 계산합니다.
    - 학습용 레이블(target)을 생성합니다 (기본: 5틱 후 가격 상승 여부).
    - 모델 입력용 X, 타깃 y, 종목코드 리스트를 반환합니다.

    사용법 예:
        loader = RealtimeFeatureLoader('data/20251120.csv')
        X, y, codes = loader.prepare_features()

    주의:
    - 입력 CSV는 최소 필수 컬럼(`timestamp`, `stock_code`, `stck_prpr` 등)을 포함해야 합니다.
    - 파일 인코딩 문제로 utf-8/CP949 모두 지원하도록 처리합니다.
    """

    # 필수 컬럼 정의 (CSV 파일에 반드시 있어야 하는 컬럼들)
    REQUIRED_COLUMNS = [
        'timestamp', 'stock_code', 'stck_cntg_hour', 'stck_prpr',
        'prdy_vrss', 'prdy_ctrt', 'acml_tr_pbmn',
        'seln_cntg_csnu', 'shnu_cntg_csnu', 'askp1', 'bidp1',
        'total_askp_rsqn', 'total_bidp_rsqn'
    ]

    def __init__(self, csv_file_path):
        # CSV 경로 저장 및 존재 여부 확인
        self.csv_path = csv_file_path

        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f" CSV 파일을 찾을 수 없습니다: {csv_file_path}")

        print(f" CSV 파일 경로 확인 완료: {csv_file_path}")

    def load_and_preprocess(self):
        """CSV 로드 및 초기 전처리

        작업 내용:
        - 파일 인코딩(utf-8, cp949) 자동 처리
        - 컬럼명 소문자/공백 정리 및 표준화(mapping)
        - 종목코드 `stock_code` 칼럼으로 통일하고 6자리로 포맷(zfill)
        - 필수 컬럼 존재 여부 체크 (없으면 KeyError)
        - timestamp 컬럼을 datetime으로 변환하고 종목별로 시간 정렬
        """
        print("\n 체결 정보 CSV 로딩 중...")

        # CSV 로드 (인코딩 안정성 확보)
        try:
            df = pd.read_csv(self.csv_path, sep=',', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.csv_path, sep=',', encoding='cp949')

        # 컬럼명 정리 (공백/특수문자 제거, 소문자 변환)
        df.columns = df.columns.str.strip().str.lower()

        # 컬럼명 표준화 매핑: 다양한 입력 칼럼명을 'stock_code'로 통일
        column_mapping = {
            'stck_shrn_iscd': 'stock_code',
            'code': 'stock_code',
            '종목코드': 'stock_code'
        }

        df = df.rename(columns=column_mapping)

        # stock_code 컬럼 확보: 여러 후보 컬럼을 검사해서 통일
        if 'stock_code' not in df.columns:
            for possible_name in ['stck_shrn_iscd', '종목코드', 'code']:
                if possible_name in df.columns:
                    df = df.rename(columns={possible_name: 'stock_code'})
                    break

        # 종목 코드 정규화 (문자열, 6자리 포맷)
        df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)

        # 필수 컬럼 체크: 누락되면 명확한 에러로 처리
        missing_cols = []
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                missing_cols.append(col)

        if missing_cols:
            print(f" 필수 컬럼 누락: {missing_cols}")
            print(f"   실제 컬럼: {df.columns.tolist()}")
            raise KeyError(f"필수 컬럼이 없습니다: {missing_cols}")

        # timestamp를 datetime으로 변환 (변환 실패 시 경고)
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            print(" timestamp 변환 실패. 기본 형식으로 진행합니다.")

        # 정렬 (종목코드, 시간순) — 이후 롤링 계산을 위해 필요
        df = df.sort_values(['stock_code', 'timestamp']).reset_index(drop=True)

        print(f" 전처리 완료")
        print(f"   종목 수: {df['stock_code'].nunique()}개")
        print(f"   데이터 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

        return df

    def create_technical_features(self, df):
        """종목별 그룹 단위로 기술적 지표 계산

        수행 지표(예):
        - 최근 N틱 대비 가격 변화율 (1,5,10)
        - 이동평균(MA 5/10/20)
        - MA 대비 가격 위치
        - 거래대금 변화율
        - 호가 스프레드, 매수/매도 압력, 체결 강도
        - 변동성(최근 N틱 표준편차), 모멘텀
        - Label 생성: `target` = (5틱 후 가격 상승 여부)

        반환: 전처리 + 피처가 추가된 pandas DataFrame
        """
        print("\n 기술적 지표 계산 중...")

        result_dfs = []

        # 종목별로 그룹화하여 루프 처리 (메모리/속도는 데이터 크기에 따라 고려)
        for stock_code, group in df.groupby('stock_code'):
            group = group.copy()

            # 1) 가격 변화율: 최근 N틱 대비 비율
            group['price_change_1'] = group['stck_prpr'].pct_change(1)
            group['price_change_5'] = group['stck_prpr'].pct_change(5)
            group['price_change_10'] = group['stck_prpr'].pct_change(10)

            # 2) 이동평균
            group['ma_5'] = group['stck_prpr'].rolling(window=5, min_periods=1).mean()
            group['ma_10'] = group['stck_prpr'].rolling(window=10, min_periods=1).mean()
            group['ma_20'] = group['stck_prpr'].rolling(window=20, min_periods=1).mean()

            # 3) MA 대비 상대 가격 위치
            group['price_vs_ma5'] = (group['stck_prpr'] - group['ma_5']) / (group['ma_5'] + 1e-8)
            group['price_vs_ma20'] = (group['stck_prpr'] - group['ma_20']) / (group['ma_20'] + 1e-8)

            # 4) 거래대금 변화
            group['tr_amount_change'] = group['acml_tr_pbmn'].pct_change(1)

            # 5) 호가 기반 특성 (ask/bid 스프레드)
            group['spread'] = (group['askp1'] - group['bidp1']) / (group['stck_prpr'] + 1e-8)
            group['spread_pct'] = group['spread'] * 100

            # 6) 매수/매도 압력 — 호가 잔량 기반 비율
            total_orders = group['total_askp_rsqn'] + group['total_bidp_rsqn'] + 1e-8
            group['buy_pressure'] = group['total_bidp_rsqn'] / total_orders

            # 7) 체결 강도 (매수 체결 비율)
            total_cntg = group['seln_cntg_csnu'] + group['shnu_cntg_csnu'] + 1e-8
            group['buy_strength'] = group['shnu_cntg_csnu'] / total_cntg

            # 8) 변동성: 최근 N틱의 표준편차
            group['volatility_5'] = group['price_change_1'].rolling(window=5, min_periods=1).std()
            group['volatility_10'] = group['price_change_1'].rolling(window=10, min_periods=1).std()

            # 9) 모멘텀
            group['momentum_5'] = group['stck_prpr'] - group['stck_prpr'].shift(5)
            group['momentum_10'] = group['stck_prpr'] - group['stck_prpr'].shift(10)

            # 10) 타깃 생성: 기본은 5틱 뒤 상승 여부 (binary)
            group['future_price_5'] = group['stck_prpr'].shift(-5)
            group['target'] = (group['future_price_5'] > group['stck_prpr']).astype(int)

            result_dfs.append(group)

        final_df = pd.concat(result_dfs, ignore_index=True)
        print(f" 기술적 지표 추가 완료")

        return final_df

    def prepare_features(self):
        """최종 학습용 피처 생성 파이프라인

        단계:
        1. CSV 로드 및 전처리
        2. 기술적 지표 추가
        3. NaN 제거
        4. 피처 선택 및 X, y, stock_codes 반환
        """
        # 1. 로드 및 전처리
        df = self.load_and_preprocess()

        # 2. 기술적 지표 추가
        df = self.create_technical_features(df)

        # 3. NaN 제거: 계산 과정에서 생긴 결측치를 제거
        original_len = len(df)
        df = df.dropna()

        if len(df) == 0:
            print(" 유효한 데이터가 없습니다.")
            return None, None, None

        print(f"\n NaN 제거: {original_len:,} → {len(df):,} ({len(df)/original_len*100:.1f}%)")

        # 4. 피쳐 선택 (GCN 임베딩은 이후 병합 시 추가됨)
        feature_cols = [
            'prdy_ctrt',              # 전일대비율
            'price_change_1', 'price_change_5', 'price_change_10',
            'price_vs_ma5', 'price_vs_ma20',
            'tr_amount_change',       # 거래대금 변화율
            'spread', 'spread_pct',
            'buy_pressure',
            'buy_strength',           # 체결 강도
            'volatility_5', 'volatility_10',
            'momentum_5', 'momentum_10'
        ]

        X = df[feature_cols].copy()
        y = df['target'].copy()
        stock_codes = df['stock_code'].copy()

        print(f"\n 최종 데이터 준비 완료!")
        print(f"   샘플 수: {len(X):,}")
        print(f"   피쳐 수: {len(feature_cols)}")
        print(f"   상승(1): {(y==1).sum():,}개 ({(y==1).sum()/len(y)*100:.1f}%)")
        print(f"   하락(0): {(y==0).sum():,}개 ({(y==0).sum()/len(y)*100:.1f}%)")

        return X, y, stock_codes


# 실행 테스트
if __name__ == "__main__":
    csv_path = r"C:\Users\user\project\MyEggBasket-AI\20251120.csv"

    loader = RealtimeFeatureLoader(csv_path)
    X, y, stock_codes = loader.prepare_features()

    if X is not None:
        print("\n[피쳐 샘플]")
        print(X.head())
        print(f"\n[타겟 분포]")
        print(y.value_counts())
        print(f"\n[종목코드 샘플]")
        print(stock_codes.head())