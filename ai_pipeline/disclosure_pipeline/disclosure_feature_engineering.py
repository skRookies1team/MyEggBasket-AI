import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

class DisclosureFeatureEngineer:
    """
    integrated_financial_data.csv 전처리 및 ML 피처 변환 (개선 버전)
    """
    
    def __init__(self, disclosure_csv_path):
        self.disclosure_path = disclosure_csv_path
        self.df = None
        
    def load_disclosure_data(self):
        """공시 데이터 로드"""
        try:
            self.df = pd.read_csv(self.disclosure_path, encoding='utf-8-sig')
        except:
            try:
                self.df = pd.read_csv(self.disclosure_path, encoding='cp949')
            except Exception as e:
                print(f"❌ CSV 로드 실패: {e}")
                return None
        
        return self.df
    
    def clean_and_standardize(self):
        """데이터 정제 및 표준화 (개선)"""
        df = self.df.copy()
        
        # 1. 종목코드 6자리 포맷팅
        df['stock_code'] = df['stock_code'].astype(str).str.strip().str.zfill(6)
        
        # 2. 상장폐지 종목 제거
        df = df[df['stock_code'] != '000000']
        df = df[df['stock_code'] != 'nan']
        df = df[~df['stock_code'].isna()]
        
        # 🔥 [핵심 개선] None 문자열 및 특수값 일괄 처리
        # 'None', '-', '', 'nan', 'NaN', 'N/A' 등을 모두 NaN으로 변환
        null_values = ['None', '-', '', 'nan', 'NaN', 'N/A', 'null', '#N/A', '#VALUE!']
        
        for col in df.columns:
            if col not in ['stock_code', 'bsns_year', 'corp_name', 'report_name', 'corp_code', 'rcept_no', 'rcept_dt', 'reprt_code']:
                # 문자열 타입 null 값 치환
                df[col] = df[col].replace(null_values, np.nan)
                
                # 숫자 변환 시도
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.df = df
        return df
    
    def create_financial_features(self):
        """재무 데이터 기반 피처 생성"""
        df = self.df.copy()
        
        # =============================================
        # 1. 재무비율 계산 (핵심 지표)
        # =============================================
        
        # ROE (자기자본이익률)
        if 'fin_net_income' in df.columns and 'fin_total_equity' in df.columns:
            df['roe'] = df['fin_net_income'] / (df['fin_total_equity'] + 1e-10)
            df['roe'] = df['roe'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # ROA (총자산이익률)
        if 'fin_net_income' in df.columns and 'fin_total_assets' in df.columns:
            df['roa'] = df['fin_net_income'] / (df['fin_total_assets'] + 1e-10)
            df['roa'] = df['roa'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # 영업이익률
        if 'fin_op_income' in df.columns and 'fin_revenue' in df.columns:
            df['operating_margin'] = df['fin_op_income'] / (df['fin_revenue'] + 1e-10)
            df['operating_margin'] = df['operating_margin'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # 순이익률
        if 'fin_net_income' in df.columns and 'fin_revenue' in df.columns:
            df['net_margin'] = df['fin_net_income'] / (df['fin_revenue'] + 1e-10)
            df['net_margin'] = df['net_margin'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # 부채비율
        if 'fin_total_liabilities' in df.columns and 'fin_total_equity' in df.columns:
            df['debt_ratio'] = df['fin_total_liabilities'] / (df['fin_total_equity'] + 1e-10)
            df['debt_ratio'] = df['debt_ratio'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # 자기자본비율
        if 'fin_total_equity' in df.columns and 'fin_total_assets' in df.columns:
            df['equity_ratio'] = df['fin_total_equity'] / (df['fin_total_assets'] + 1e-10)
            df['equity_ratio'] = df['equity_ratio'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # =============================================
        # 2. 규모 관련 피처 (Log Scale)
        # =============================================
        
        if 'fin_total_assets' in df.columns:
            df['log_assets'] = np.log1p(df['fin_total_assets'].fillna(0))
        
        if 'fin_revenue' in df.columns:
            df['log_revenue'] = np.log1p(df['fin_revenue'].fillna(0))
        
        if 'emp_total_count' in df.columns:
            df['emp_total_count'] = pd.to_numeric(df['emp_total_count'], errors='coerce')
            df['log_employees'] = np.log1p(df['emp_total_count'].fillna(0))
        
        # =============================================
        # 3. 성장성 지표 (YoY)
        # =============================================
        
        if 'bsns_year' in df.columns:
            df = df.sort_values(['stock_code', 'bsns_year'])
            
            if 'fin_revenue' in df.columns:
                df['revenue_growth'] = df.groupby('stock_code')['fin_revenue'].pct_change(fill_method=None)
                df['revenue_growth'] = df['revenue_growth'].fillna(0).replace([np.inf, -np.inf], 0)
            
            if 'fin_op_income' in df.columns:
                df['op_income_growth'] = df.groupby('stock_code')['fin_op_income'].pct_change(fill_method=None)
                df['op_income_growth'] = df['op_income_growth'].fillna(0).replace([np.inf, -np.inf], 0)
            
            if 'fin_total_assets' in df.columns:
                df['asset_growth'] = df.groupby('stock_code')['fin_total_assets'].pct_change(fill_method=None)
                df['asset_growth'] = df['asset_growth'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # =============================================
        # 4. 기타 피처
        # =============================================
        
        if 'capital_change_count' in df.columns:
            df['capital_change_count'] = pd.to_numeric(df['capital_change_count'], errors='coerce').fillna(0)
        
        if 'treasury_stock_event' in df.columns:
            df['has_treasury_event'] = (df['treasury_stock_event'] == 'Y').astype(int)
        
        self.df = df
        return df
    
    def get_feature_columns(self):
        """ML 모델에 사용할 피처 컬럼 목록 반환"""
        feature_cols = [
            'roe', 'roa', 'operating_margin', 'net_margin',
            'debt_ratio', 'equity_ratio',
            'log_assets', 'log_revenue', 'log_employees',
            'revenue_growth', 'op_income_growth', 'asset_growth',
            'capital_change_count',
            'has_treasury_event'
        ]
        
        available_cols = []
        for col in feature_cols:
            if col not in self.df.columns:
                continue
            
            # 🔥 변별력 체크 완화 (95% -> 90%)
            zero_ratio = (self.df[col] == 0).sum() / len(self.df)
            if zero_ratio > 0.90:
                continue
            
            # 단일값 피처 제외
            if self.df[col].nunique() <= 1:
                continue
                
            available_cols.append(col)
        
        return available_cols
    
    def prepare_ml_features(self):
        """ML 모델 입력용 최종 데이터 준비"""
        # 1. 데이터 로드
        if self.df is None:
            self.load_disclosure_data()
        
        if self.df is None:
            return None, []
        
        # 2. 정제
        self.clean_and_standardize()
        
        # 3. 피처 생성
        self.create_financial_features()
        
        # 4. 피처 선택
        feature_cols = self.get_feature_columns()
        
        if not feature_cols:
            print("⚠️ 사용 가능한 피처가 없습니다!")
            return None, []
        
        # 5. 최종 데이터프레임
        final_df = self.df[['stock_code', 'bsns_year'] + feature_cols].copy()
        
        # 6. 결측치 처리 (0으로 채우기)
        final_df[feature_cols] = final_df[feature_cols].fillna(0)
        
        # 7. 이상치 제거 (극단값 클리핑)
        for col in feature_cols:
            if col not in ['has_treasury_event', 'capital_change_count']:
                lower = final_df[col].quantile(0.01)
                upper = final_df[col].quantile(0.99)
                final_df[col] = final_df[col].clip(lower, upper)
        
        return final_df, feature_cols


# ==========================================
# 테스트 실행
# ==========================================
if __name__ == "__main__":
    # 경로 설정
    csv_path = "integrated_financial_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
    else:
        engineer = DisclosureFeatureEngineer(csv_path)
        final_df, feature_cols = engineer.prepare_ml_features()
        
        print("\n[피처 컬럼 목록]")
        for col in feature_cols:
            print(f"  - {col}")
        
        print("\n[데이터 샘플]")
        print(final_df.head())
        
        # CSV 저장
        output_path = "disclosure_features.csv"
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 저장 완료: {output_path}")