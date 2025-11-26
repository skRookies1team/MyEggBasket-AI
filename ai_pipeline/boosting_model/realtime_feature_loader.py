import pandas as pd
import numpy as np
from datetime import datetime
import os

class RealtimeFeatureLoader:
    """
    실시간 체결 정보 CSV를 로드하고 머신러닝 피쳐로 변환
    """
    
    # 필수 컬럼 정의 (CSV 파일에 반드시 있어야 하는 컬럼들)
    REQUIRED_COLUMNS = [
        'timestamp', 'stock_code', 'stck_cntg_hour', 'stck_prpr',
        'prdy_vrss', 'prdy_ctrt', 'acml_tr_pbmn', 'acml_vol',
        'seln_cntg_csnu', 'shnu_cntg_csnu', 'askp1', 'bidp1',
        'total_askp_rsqn', 'total_bidp_rsqn'
    ]
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}")
        
        print(f"✅ CSV 파일 경로 확인 완료: {csv_path}")
    
    def load_and_preprocess(self):
        """CSV 파일 로드 및 전처리"""
        print("\n📊 체결 정보 CSV 로딩 중...")
        
        # CSV 로드
        df = pd.read_csv(self.csv_path, sep=',', encoding='utf-8')
        
        print(f"   원본 데이터: {len(df):,}개 행, {len(df.columns)}개 컬럼")
        
        # 컬럼명 정리 (공백/특수문자 제거, 소문자 변환)
        df.columns = df.columns.str.strip().str.lower()
        
        # 컬럼명 표준화 매핑
        column_mapping = {
            'stck_shrn_iscd': 'stock_code',
            # 실제 CSV의 컬럼명을 그대로 사용 (이미 표준)
        }
        
        df = df.rename(columns=column_mapping)
        
        # stock_code 컬럼 확인 및 처리
        if 'stock_code' not in df.columns:
            # 종목코드 컬럼 찾기
            for possible_name in ['stck_shrn_iscd', '종목코드', 'code']:
                if possible_name in df.columns:
                    df = df.rename(columns={possible_name: 'stock_code'})
                    break
        
        # 종목 코드 처리 (앞에 0 채우기 - 6자리로 통일)
        df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
        
        # 필수 컬럼 체크
        missing_cols = []
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"❌ 필수 컬럼 누락: {missing_cols}")
            print(f"   실제 컬럼: {df.columns.tolist()}")
            raise KeyError(f"필수 컬럼이 없습니다: {missing_cols}")
        
        # timestamp를 datetime으로 변환
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            print("⚠️ timestamp 변환 실패. 기본 형식으로 진행합니다.")
        
        # 정렬 (종목코드, 시간순)
        df = df.sort_values(['stock_code', 'timestamp']).reset_index(drop=True)
        
        print(f"✅ 전처리 완료")
        print(f"   종목 수: {df['stock_code'].nunique()}개")
        print(f"   데이터 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        
        return df
    
    def create_technical_features(self, df):
        """기술적 지표 생성"""
        print("\n📈 기술적 지표 계산 중...")
        
        result_dfs = []
        
        for stock_code, group in df.groupby('stock_code'):
            group = group.copy()
            
            # 1. 가격 변화율 (최근 N틱 대비)
            group['price_change_1'] = group['stck_prpr'].pct_change(1)
            group['price_change_5'] = group['stck_prpr'].pct_change(5)
            group['price_change_10'] = group['stck_prpr'].pct_change(10)
            
            # 2. 이동평균
            group['ma_5'] = group['stck_prpr'].rolling(window=5, min_periods=1).mean()
            group['ma_10'] = group['stck_prpr'].rolling(window=10, min_periods=1).mean()
            group['ma_20'] = group['stck_prpr'].rolling(window=20, min_periods=1).mean()
            
            # 3. 가격 위치 (MA 대비)
            group['price_vs_ma5'] = (group['stck_prpr'] - group['ma_5']) / (group['ma_5'] + 1e-8)
            group['price_vs_ma20'] = (group['stck_prpr'] - group['ma_20']) / (group['ma_20'] + 1e-8)
            
            # 4. 거래량 특징
            group['volume_ma_5'] = group['acml_vol'].rolling(window=5, min_periods=1).mean()
            group['volume_ratio'] = group['acml_vol'] / (group['volume_ma_5'] + 1e-8)
            
            # 5. 거래대금 특징
            group['tr_amount_change'] = group['acml_tr_pbmn'].pct_change(1)
            
            # 6. 호가 스프레드
            group['spread'] = (group['askp1'] - group['bidp1']) / (group['stck_prpr'] + 1e-8)
            group['spread_pct'] = group['spread'] * 100
            
            # 7. 매수/매도 압력
            total_orders = group['total_askp_rsqn'] + group['total_bidp_rsqn'] + 1e-8
            group['buy_pressure'] = group['total_bidp_rsqn'] / total_orders
            
            # 8. 체결 강도 (매수 vs 매도 체결량)
            total_cntg = group['seln_cntg_csnu'] + group['shnu_cntg_csnu'] + 1e-8
            group['buy_strength'] = group['shnu_cntg_csnu'] / total_cntg
            
            # 9. 변동성 (최근 N틱의 표준편차)
            group['volatility_5'] = group['price_change_1'].rolling(window=5, min_periods=1).std()
            group['volatility_10'] = group['price_change_1'].rolling(window=10, min_periods=1).std()
            
            # 10. 가격 모멘텀
            group['momentum_5'] = group['stck_prpr'] - group['stck_prpr'].shift(5)
            group['momentum_10'] = group['stck_prpr'] - group['stck_prpr'].shift(10)
            
            # 11. 타겟 생성 (다음 5틱 후 가격 상승 여부)
            group['future_price_5'] = group['stck_prpr'].shift(-5)
            group['target'] = (group['future_price_5'] > group['stck_prpr']).astype(int)
            
            result_dfs.append(group)
        
        final_df = pd.concat(result_dfs, ignore_index=True)
        print(f"✅ 기술적 지표 추가 완료")
        
        return final_df
    
    def prepare_features(self):
        """최종 피쳐 데이터 준비"""
        # 1. 로드 및 전처리
        df = self.load_and_preprocess()
        
        # 2. 기술적 지표 추가
        df = self.create_technical_features(df)
        
        # 3. NaN 제거
        original_len = len(df)
        df = df.dropna()
        
        if len(df) == 0:
            print("❌ 유효한 데이터가 없습니다.")
            return None, None, None
        
        print(f"\n🧹 NaN 제거: {original_len:,} → {len(df):,} ({len(df)/original_len*100:.1f}%)")
        
        # 4. 피쳐 선택 (GCN 임베딩은 나중에 merge)
        feature_cols = [
            'prdy_ctrt',              # 전일대비율
            'price_change_1', 'price_change_5', 'price_change_10',
            'price_vs_ma5', 'price_vs_ma20',
            'volume_ratio',
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
        
        print(f"\n✅ 최종 데이터 준비 완료!")
        print(f"   샘플 수: {len(X):,}")
        print(f"   피쳐 수: {len(feature_cols)}")
        print(f"   상승(1): {(y==1).sum():,}개 ({(y==1).sum()/len(y)*100:.1f}%)")
        print(f"   하락(0): {(y==0).sum():,}개 ({(y==0).sum()/len(y)*100:.1f}%)")
        
        return X, y, stock_codes


# 실행 테스트
if __name__ == "__main__":
    csv_path = r"C:\rookies4dev\final_project\MyEggBasket-AI\20251120.csv"
    
    loader = RealtimeFeatureLoader(csv_path)
    X, y, stock_codes = loader.prepare_features()
    
    if X is not None:
        print("\n[피쳐 샘플]")
        print(X.head())
        print(f"\n[타겟 분포]")
        print(y.value_counts())
        print(f"\n[종목코드 샘플]")
        print(stock_codes.head())