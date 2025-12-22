import pandas as pd
import numpy as np
import os
import glob
import ta
import re


class FeatureExpander:
    def __init__(self, data_dir=None, max_days=30):
        """
        초기화 시 1년치 분봉 데이터를 로드하여
        1. 전일 종가 기준 등락률(prdy_ctrt)
        2. 분 단위 기술적 지표 (RSI, SMA 등)
        를 미리 계산해두고, 요청 시 병합(Merge)해줍니다.
        max_days: 메모리에 유지할 최근 일수 (OOM 방지용)
        """
        self.ta_features_df = None
        self.max_days = max_days

        # 데이터 폴더 경로 설정
        if data_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.abspath(os.path.join(current_dir, "../../data"))
        else:
            self.data_dir = data_dir

        self._load_and_process_history()

    def _load_and_process_history(self):
        print(f" [TA] 1년치 분봉 데이터 로드 및 정밀 지표 계산 중...")

        # 1. _1Year.csv 파일 찾기
        target_files = glob.glob(os.path.join(self.data_dir, "*_1Year.csv"))

        if not target_files:
            print(" [TA] '_1Year.csv' 패턴의 파일을 찾을 수 없습니다.")
            return

        results = []

        for fpath in target_files:
            try:
                # 파일명에서 종목코드 추출 (예: 000270_1Year.csv -> 000270)
                basename = os.path.basename(fpath)
                code_match = re.search(r'(\d+)', basename)
                stock_code = None
                if code_match:
                    stock_code = code_match.group(1).zfill(6) # 예: 270 -> 000270

                # CSV 로드
                df = pd.read_csv(fpath)
                df.columns = df.columns.str.strip().str.lower()

                # 컬럼 매핑 (Close -> close)
                rename_map = {'close': 'close', 'volume': 'volume', 'stck_prpr': 'close', 'acml_vol': 'volume'}
                df = df.rename(columns=rename_map)

                # [수정 2] 파일명에서 코드를 못 찾았을 경우 CSV 내부 컬럼 확인
                if stock_code is None:
                    if 'stock_code' in df.columns:
                        # 첫 번째 행의 값을 가져와서 6자리 문자열로 변환
                        stock_code = str(df['stock_code'].iloc[0]).split('.')[0].strip().zfill(6)
                    elif 'code' in df.columns:
                        stock_code = str(df['code'].iloc[0]).split('.')[0].strip().zfill(6)
                    else:
                        # 코드 식별 불가 시 스킵
                        continue

                # Timestamp 생성 (YYYYMMDD + HHMMSS)
                if 'date' in df.columns and 'time' in df.columns:
                    df['ts_str'] = df['date'].astype(str) + df['time'].astype(str).str.zfill(6)
                    df['timestamp'] = pd.to_datetime(df['ts_str'], format='%Y%m%d%H%M%S', errors='coerce')
                elif 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                if 'timestamp' not in df.columns or 'close' not in df.columns:
                    continue

                # 정렬 (과거 -> 현재)
                df = df.sort_values('timestamp').reset_index(drop=True)
                df['code'] = stock_code

                # -------------------------------------------------------
                # [1] 전일 종가 기반 등락률 (prdy_ctrt) 계산
                # -------------------------------------------------------
                # 날짜별 마지막 가격(종가)을 구함
                df['date_only'] = df['timestamp'].dt.date
                daily_close = df.groupby('date_only')['close'].last().shift(1)  # 전일 종가

                # 원본 데이터에 '전일 종가' 컬럼 매핑
                df = df.merge(daily_close.rename('prev_close'), left_on='date_only', right_index=True, how='left')

                # 등락률 계산: (현재가 - 전일종가) / 전일종가 * 100
                # 전일 데이터가 없는 첫 날은 0으로 처리
                df['hist_prdy_ctrt'] = ((df['close'] - df['prev_close']) / df['prev_close'] * 100).fillna(0.0)

                # -------------------------------------------------------
                # [2] 분 단위 기술적 지표 (Continuous Minute Indicators)
                # -------------------------------------------------------
                # 끊김 없이 연결된 시계열을 사용하여 09:00 장시작 직후에도 정확한 MA/RSI 계산 가능

                # (A) RSI (14분 기준)
                df['hist_RSI_14'] = ta.momentum.rsi(df['close'], window=14, fillna=True)

                # (B) 이동평균 (5분, 20분, 60분, 120분)
                for w in [5, 20, 60, 120]:
                    ma = ta.trend.sma_indicator(df['close'], window=w, fillna=True)
                    # 이격도 (현재가 / 이동평균)
                    df[f'hist_Disparity_{w}'] = df['close'] / (ma + 1e-9)

                # (C) 볼린저 밴드 (20분 기준)
                bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
                df['hist_BB_High'] = bb.bollinger_hband()
                df['hist_BB_Low'] = bb.bollinger_lband()
                # 밴드 위치 (Percent B): 1.0 넘으면 상단 돌파, 0.0 미만이면 하단 이탈
                df['hist_BB_PctB'] = bb.bollinger_pband().fillna(0.5)

                # (D) MACD (12, 26, 9)
                macd = ta.trend.MACD(df['close'])
                df['hist_MACD'] = macd.macd()
                df['hist_MACD_Signal'] = macd.macd_signal()
                df['hist_MACD_Diff'] = macd.macd_diff()  # 히스토그램

                # (E) 거래량 이동평균 비율 (지금 거래량이 20분 평균 대비 얼마나 터졌나?)
                vol_ma20 = ta.trend.sma_indicator(df['volume'], window=20, fillna=True)
                df['hist_Vol_Ratio'] = df['volume'] / (vol_ma20 + 1e-9)

                # 병합에 필요한 컬럼만 선택하여 저장
                cols_to_keep = [
                    'code', 'timestamp',
                    'hist_prdy_ctrt',
                    'hist_RSI_14',
                    'hist_Disparity_5', 'hist_Disparity_20', 'hist_Disparity_60',
                    'hist_BB_PctB', 'hist_MACD_Diff', 'hist_Vol_Ratio'
                ]

                cutoff_date = df['timestamp'].max() - pd.Timedelta(days=self.max_days)
                df_recent = df[df['timestamp'] >= cutoff_date]

                results.append(df_recent[cols_to_keep])

            except Exception as e:
                print(f" [Error] {os.path.basename(fpath)} 처리 중 오류: {e}")
                continue

        # 전체 데이터를 하나의 DataFrame으로 병합 (메모리에 상주)
        if results:
            self.ta_features_df = pd.concat(results)
            # 검색 속도를 위해 인덱스 설정 (선택사항, 데이터 크기에 따라 조정)
            # self.ta_features_df.set_index(['code', 'timestamp'], inplace=True)
            print(f" [TA] 정밀 지표 계산 완료! (총 {len(self.ta_features_df):,}개 분봉 데이터 확보)")
        else:
            print(" [TA] 계산된 지표가 없습니다.")

    def add_technical_indicators(self, df):
        """
        메인 파이프라인 데이터(df)에 '정확한 분 단위 지표'를 병합합니다.
        Key: 종목코드(code) AND 시간(timestamp)
        """
        if self.ta_features_df is None or self.ta_features_df.empty:
            return df

        # 병합 키 컬럼 준비
        target_code_col = None
        if 'stck_shrn_iscd' in df.columns:
            target_code_col = 'stck_shrn_iscd'
        elif 'code' in df.columns:
            target_code_col = 'code'
        elif 'stock_code' in df.columns:
            target_code_col = 'stock_code'

        if not target_code_col or 'timestamp' not in df.columns:
            return df

        # 타입 통일
        df[target_code_col] = df[target_code_col].astype(str).str.strip().str.zfill(6)

        # Timestamp 타입 통일 (혹시 모를 오류 방지)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # print(f" [TA] 과거 데이터 매핑 중... (Target: {len(df)} rows)")

        # 병합 (Left Join)
        # 파이프라인의 데이터(df)를 기준으로, 1년치 역사 데이터(ta_features_df)에서
        # 해당 '시간'과 '종목'에 맞는 지표를 정확히 가져옵니다.
        merged_df = pd.merge(
            df,
            self.ta_features_df,
            left_on=[target_code_col, 'timestamp'],
            right_on=['code', 'timestamp'],
            how='left'
        )

        # 중복된 'code' 컬럼 제거
        if 'code' in merged_df.columns and target_code_col != 'code':
            merged_df = merged_df.drop(columns=['code'])

        # 매핑되지 않은 데이터(과거 데이터에 없는 최신 데이터 등)는 0 처리
        # (실시간 수집 데이터가 1Year 파일보다 최신일 경우 발생 가능 -> 이 경우 실시간 로더의 지표가 대체함)
        fill_cols = [c for c in merged_df.columns if c.startswith('hist_')]
        merged_df[fill_cols] = merged_df[fill_cols].fillna(0)

        return merged_df


if __name__ == "__main__":
    # 테스트
    expander = FeatureExpander()