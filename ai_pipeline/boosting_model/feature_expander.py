import pandas as pd
import numpy as np
import os
import ta

class FeatureExpander:
    def __init__(self):
        """
        초기화 시 5일치 데이터를 미리 로드하여 '단기 기술적 지표'를 계산해둡니다.
        """
        self.ta_features_df = None
        self._load_and_process_history()

    def _load_and_process_history(self):
        # 1. 파일 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # stock_data_5days.csv 파일을 찾습니다.
        candidates = [
            os.path.join(current_dir, "../../data/stock_data_5days.csv"),
            os.path.join(current_dir, "../../stock_data_5days.csv"),
            os.path.join(current_dir, "stock_data_5days.csv")
        ]
        
        csv_path = None
        for path in candidates:
            if os.path.exists(path):
                csv_path = path
                break
        
        if not csv_path:
            print(" [TA] stock_data_5days.csv 파일을 찾을 수 없습니다. (지표 생성 건너뜀)")
            return

        print(f" [TA] 5일치 데이터 로드 및 단기 지표 계산 중...")
        
        try:
            # 2. 데이터 로드
            df_hist = pd.read_csv(csv_path)
            
            # 3. 컬럼명 매핑 (ta 라이브러리가 좋아하는 이름으로 변경)
            rename_map = {
                'stck_oprc': 'open',
                'stck_hgpr': 'high',
                'stck_lwpr': 'low',
                'stck_prpr': 'close',
                'acml_vol': 'volume',
                'date': 'date'
            }
            df_hist = df_hist.rename(columns=rename_map)

            # 4. 종목코드 포맷 통일 (5930 -> 005930)
            if 'stock_code' in df_hist.columns:
                df_hist['code'] = df_hist['stock_code'].astype(str).str.strip().str.zfill(6)
            else:
                print(" CSV에 'stock_code' 컬럼이 없습니다.")
                return

            # 5. 날짜순 정렬 (과거 -> 현재)
            df_hist = df_hist.sort_values(['code', 'date'])
            
            # -------------------------------------------------------
            # 🎯 [핵심] 5일 데이터 전용 지표 계산 로직
            # -------------------------------------------------------
            results = []
            grouped = df_hist.groupby('code')
            
            for code, group in grouped:
                if len(group) < 2: continue # 데이터 너무 적으면 패스
                
                g = group.copy()
                
                # (A) RSI (상대강도지수) - 3일 기준
                # 5일 데이터로 14일 RSI를 구하면 NaN이 뜨므로 window=3으로 설정
                g['RSI_3'] = ta.momentum.rsi(g['close'], window=3, fillna=True)
                
                # (B) 이동평균선 (SMA) - 3일, 5일
                g['SMA_3'] = ta.trend.sma_indicator(g['close'], window=3, fillna=True)
                g['SMA_5'] = ta.trend.sma_indicator(g['close'], window=5, fillna=True)
                
                # (C) 이격도 (Disparity)
                # 현재 주가가 5일 평균 대비 얼마나 높은가? (1.05면 5% 높음)
                # 분모가 0일 수 있으니 안전하게 처리
                g['Disparity_5'] = g['close'] / (g['SMA_5'] + 1e-9)
                
                # (D) 변동성 (ATR) - 3일 기준
                g['ATR_3'] = ta.volatility.average_true_range(
                    g['high'], g['low'], g['close'], window=3, fillna=True
                )
                
                # (E) 거래량 변화 (오늘 거래량 / 3일 평균 거래량)
                vol_sma3 = ta.trend.sma_indicator(g['volume'], window=3, fillna=True)
                g['Vol_Ratio'] = g['volume'] / (vol_sma3 + 1e-9)

                # 계산된 지표 중 '가장 최근 날짜(마지막 행)'만 가져옴
                # 왜냐하면 우리는 '오늘'의 예측을 위해 피처를 붙이는 것이니까요.
                latest_row = g.iloc[[-1]].copy()
                
                # 불필요한 원본 컬럼 제거 (선택사항)
                cols_to_keep = ['code', 'RSI_3', 'SMA_3', 'SMA_5', 'Disparity_5', 'ATR_3', 'Vol_Ratio']
                results.append(latest_row[cols_to_keep])

            # 6. 결과 하나로 합치기
            if results:
                self.ta_features_df = pd.concat(results)
                print(f" 단기 지표 계산 완료! ({len(self.ta_features_df)}개 종목 확보)")
            
        except Exception as e:
            print(f" TA 지표 생성 중 오류: {e}")

    def add_technical_indicators(self, df):
        """
        메인 파이프라인 데이터(df)에 계산해둔 지표를 붙입니다.
        """
        if self.ta_features_df is None or self.ta_features_df.empty:
            return df
        
        # 병합 키 컬럼 찾기
        target_col = None
        if 'stck_shrn_iscd' in df.columns: target_col = 'stck_shrn_iscd'
        elif 'code' in df.columns: target_col = 'code'
        
        if not target_col: return df

        # 타입 통일
        df[target_col] = df[target_col].astype(str).str.strip().str.zfill(6)
        
        print(f" [TA] 5일치 기반 단기 지표 병합 중...")

        # 병합 (Left Join)
        merged_df = pd.merge(
            df, 
            self.ta_features_df, 
            left_on=target_col, 
            right_on='code', 
            how='left'
        )
        
        # 중복된 'code' 컬럼 제거
        if 'code' in merged_df.columns and target_col != 'code':
            merged_df = merged_df.drop(columns=['code'])
            
        # 데이터가 없어서 NaN인 경우 0으로 채움
        merged_df = merged_df.fillna(0)
        
        print(f" 병합 완료. (총 컬럼: {len(merged_df.columns)})")
        return merged_df

if __name__ == "__main__":
    # 테스트 코드
    expander = FeatureExpander()
    # 5930(삼성전자) 테스트용 데이터
    dummy_df = pd.DataFrame({'stck_shrn_iscd': ['005930', '000660']})
    res = expander.add_technical_indicators(dummy_df)
    print(res)