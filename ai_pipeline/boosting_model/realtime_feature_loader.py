import pandas as pd
import numpy as np
from datetime import datetime
import os
import re


class RealtimeFeatureLoader:
    """
    실시간 체결 정보 CSV를 로드하고 머신러닝 피쳐로 변환
    (1분 단위 리샘플링 적용, 호가 정보 없는 데이터셋 대응)
    """

    def __init__(self, csv_file_path):
        self.csv_path = csv_file_path
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f" CSV 파일을 찾을 수 없습니다: {csv_file_path}")
        print(f" [Loader] 파일 처리 시작: {os.path.basename(csv_file_path)}")

    def _extract_code_from_filename(self):
        """파일명에서 종목코드 추출"""
        basename = os.path.basename(self.csv_path)
        match = re.search(r'(\d{6})', basename)
        return match.group(1) if match else '000000'

    def load_and_preprocess(self):
        """CSV 파일 로드 및 전처리"""
        try:
            df = pd.read_csv(self.csv_path, sep=',', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.csv_path, sep=',', encoding='cp949')

        df.columns = df.columns.str.strip().str.lower()

        # ---------------------------------------------------------
        # [Case A] OHLCV 포맷 (Date, Time, Close, Volume)
        # ---------------------------------------------------------
        if 'date' in df.columns and 'time' in df.columns and 'close' in df.columns:
            # 1. 기본 컬럼 매핑
            df['stock_code'] = self._extract_code_from_filename()

            # Timestamp 생성
            df['date'] = df['date'].astype(str)
            df['time'] = df['time'].astype(str).str.zfill(6)
            df['timestamp'] = pd.to_datetime(
                df['date'] + df['time'], format='%Y%m%d%H%M%S', errors='coerce'
            )

            df = df.rename(columns={'close': 'stck_prpr', 'volume': 'volume_qty'})

            # 2. 거래대금 추정 (거래량 * 현재가)
            if 'volume_qty' in df.columns:
                df['acml_tr_pbmn'] = df['stck_prpr'] * df['volume_qty']

            # 3. [New] 전일 대비율(prdy_ctrt) 직접 계산
            # 날짜별 종가(Last price)를 구해서 어제 종가를 찾음
            df['date_only'] = df['timestamp'].dt.date
            daily_close = df.groupby('date_only')['stck_prpr'].last().shift(1)  # 전일 종가

            # 원본 데이터에 전일 종가 매핑
            df = df.merge(daily_close.rename('prev_close'), left_on='date_only', right_index=True, how='left')

            # 등락률 계산: (현재가 - 전일종가) / 전일종가 * 100
            df['prdy_ctrt'] = ((df['stck_prpr'] - df['prev_close']) / df['prev_close'] * 100).fillna(0.0)

            # 불필요한 임시 컬럼 제거
            df = df.drop(columns=['date_only', 'prev_close'])

        # ---------------------------------------------------------
        # [Case B] 기존 포맷 (stck_prpr 포함)
        # ---------------------------------------------------------
        else:
            column_mapping = {
                'stck_shrn_iscd': 'stock_code', 'code': 'stock_code',
                'stck_prpr': 'stck_prpr', '현재가': 'stck_prpr',
                'acml_tr_pbmn': 'acml_tr_pbmn',
                'prdy_ctrt': 'prdy_ctrt'
            }
            df = df.rename(columns=column_mapping)
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                pass

        # 공통 후처리
        if 'stock_code' not in df.columns:
            df['stock_code'] = self._extract_code_from_filename()

        df = df.sort_values(['stock_code', 'timestamp']).reset_index(drop=True)
        return df

    def create_technical_features(self, df):
        """기술적 지표 생성 (1분 단위)"""
        if 'timestamp' not in df.columns: return pd.DataFrame()

        df = df.set_index('timestamp')
        result_dfs = []

        for stock_code, group in df.groupby('stock_code'):
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            resampled = group[numeric_cols].resample('1min').last().ffill()

            resampled['stock_code'] = stock_code

            # 1. 가격 변화율
            resampled['price_change_1'] = resampled['stck_prpr'].pct_change(1)
            resampled['price_change_5'] = resampled['stck_prpr'].pct_change(5)
            resampled['price_change_10'] = resampled['stck_prpr'].pct_change(10)

            # 2. 이동평균 & 이격도
            for w in [5, 20]:
                ma = resampled['stck_prpr'].rolling(window=w).mean()
                resampled[f'price_vs_ma{w}'] = (resampled['stck_prpr'] - ma) / (ma + 1e-8)

            # 3. 거래대금 변화율
            if 'acml_tr_pbmn' in resampled.columns:
                resampled['tr_amount_change'] = resampled['acml_tr_pbmn'].pct_change(1)
            else:
                resampled['tr_amount_change'] = 0.0

            # 4. 변동성 & 모멘텀
            resampled['volatility_5'] = resampled['price_change_1'].rolling(window=5).std()
            resampled['volatility_10'] = resampled['price_change_1'].rolling(window=10).std()
            resampled['momentum_5'] = resampled['stck_prpr'] - resampled['stck_prpr'].shift(5)
            resampled['momentum_10'] = resampled['stck_prpr'] - resampled['stck_prpr'].shift(10)

            # 5. 전일대비율
            if 'prdy_ctrt' not in resampled.columns:
                resampled['prdy_ctrt'] = 0.0

            # ✅ [핵심 수정] Target 생성 로직 완화
            LOOK_AHEAD = 5  # 5분 뒤 가격
            THRESHOLD = 0.001  # 0.1%로 낮춤 (기존 0.3%)

            future_price = resampled['stck_prpr'].shift(-LOOK_AHEAD)
            ret = (future_price - resampled['stck_prpr']) / (resampled['stck_prpr'] + 1e-8)

            # ✅ [추가] 타겟 생성 전 데이터 검증
            valid_ret = ret.dropna()
            if len(valid_ret) > 0:
                positive_count = (valid_ret > THRESHOLD).sum()
                print(
                    f"   [{stock_code}] Target 후보: {len(valid_ret)}개, 상승: {positive_count}개 ({positive_count / len(valid_ret) * 100:.1f}%)")

            resampled['target'] = (ret > THRESHOLD).astype(int)

            result_dfs.append(resampled.reset_index())

        if not result_dfs: return pd.DataFrame()

        final_df = pd.concat(result_dfs, ignore_index=True)

        # ✅ [최종 검증] Target 분포 출력
        total_targets = final_df['target'].dropna()
        if len(total_targets) > 0:
            up_count = (total_targets == 1).sum()
            print(
                f"\n [Target 생성 완료] 전체: {len(total_targets)}개, 상승(1): {up_count}개 ({up_count / len(total_targets) * 100:.2f}%)")

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
        # [수정] 'timestamp'를 반드시 포함해야 나중에 날짜별 뉴스를 붙일 수 있습니다.
        feature_cols = [
            'timestamp',  # <--- [중요] 여기 추가!
            'prdy_ctrt',
            'price_change_1', 'price_change_5', 'price_change_10',
            'price_vs_ma5', 'price_vs_ma20', 'price_vs_ma60',
            'tr_amount_change',
            'volatility_5', 'volatility_10'
        ]

        # 데이터프레임에 실제 존재하는 컬럼만 선택
        valid_cols = [c for c in feature_cols if c in df.columns]

        # 없는 컬럼 0.0 처리 (timestamp 제외)
        for c in feature_cols:
            if c not in df.columns and c != 'timestamp':
                df[c] = 0.0

        X = df[valid_cols].copy()  # valid_cols 사용
        y = df['target'].copy()
        stock_codes = df['stock_code'].copy()

        return X, y, stock_codes


if __name__ == "__main__":
    pass