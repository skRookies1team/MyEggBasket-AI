import pandas as pd
import numpy as np
from datetime import datetime
import os

class RealtimeFeatureLoader:
    """
    실시간 체결 정보 CSV를 로드하고 머신러닝 피쳐로 변환
    
    CSV 컬럼:
    - timestamp: 체결 시간
    - stck_shrn_iscd: 종목코드
    - stck_prpr: 현재가
    - prdy_vrss: 전일대비
    - prdy_ctrt: 전일대비율
    - acml_vol: 누적 거래량
    - wght_avrg_prc: 가중평균가
    - askp1: 매도호가1
    - bidp1: 매수호가1
    - total_askp_rsqn: 총 매도호가 잔량
    - total_bidp_rsqn: 총 매수호가 잔량
    """
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}")
        
        print(f"✅ CSV 파일 경로 확인 완료: {csv_path}")
    
    def load_and_preprocess(self):
        """CSV 파일 로드 및 전처리"""
        print("\n📊 체결 정보 CSV 로딩 중...")
        
        # CSV 로드 (쉼표 구분자)
        df = pd.read_csv(self.csv_path, sep=',', encoding='utf-8')
        
        print(f"   원본 데이터: {len(df):,}개 행")
        
        # 컬럼명 정리 (공백 제거)
        df.columns = df.columns.str.strip()
        
        # 실제 컬럼명 출력 (디버깅용)
        print(f"   실제 컬럼명: {df.columns.tolist()[:5]}...")  # 처음 5개만
        
        # 종목코드 컬럼 찾기 (여러 가능성 체크)
        stock_col = None
        for possible_name in ['stck_shrn_iscd', 'stock_code', '종목코드', 'code']:
            if possible_name in df.columns:
                stock_col = possible_name
                break
        
        if stock_col is None:
            print(f"❌ 종목코드 컬럼을 찾을 수 없습니다. 전체 컬럼명:")
            print(df.columns.tolist())
            raise KeyError("종목코드 컬럼이 없습니다.")
        
        print(f"   종목 코드 컬럼: '{stock_col}'")
        print(f"   종목 수: {df[stock_col].nunique()}개")
        
        # 종목코드를 문자열로 변환 (앞에 0이 붙은 경우 처리)
        df[stock_col] = df[stock_col].astype(str).str.zfill(6)
        
        # 컬럼명 통일 (이후 코드에서 'stock_code'로 사용)
        if stock_col != 'stock_code':
            df = df.rename(columns={stock_col: 'stock_code'})
        
        # 필수 컬럼 확인 및 매핑
        required_cols = {
            'timestamp': ['timestamp', '시간', 'time'],
            'stck_prpr': ['stck_prpr', '현재가', 'price', 'current_price'],
            'prdy_vrss': ['prdy_vrss', '전일대비', 'change'],
            'prdy_ctrt': ['prdy_ctrt', '전일대비율', 'change_rate'],
            'acml_vol': ['acml_vol', '누적거래량', 'volume'],
            'wght_avrg_prc': ['wght_avrg_prc', '가중평균가', 'avg_price'],
            'askp1': ['askp1', '매도호가1', 'ask1'],
            'bidp1': ['bidp1', '매수호가1', 'bid1'],
            'total_askp_rsqn': ['total_askp_rsqn', '총매도호가잔량', 'total_ask'],
            'total_bidp_rsqn': ['total_bidp_rsqn', '총매수호가잔량', 'total_bid']
        }
        
        # 컬럼 매핑
        for standard_name, possible_names in required_cols.items():
            found = False
            for pname in possible_names:
                if pname in df.columns:
                    if pname != standard_name:
                        df = df.rename(columns={pname: standard_name})
                    found = True
                    break
            
            if not found:
                print(f"⚠️  '{standard_name}' 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼:")
                print(df.columns.tolist())
                raise KeyError(f"필수 컬럼 '{standard_name}'이 없습니다.")
        
        # timestamp를 datetime으로 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 정렬 (종목코드, 시간순)
        df = df.sort_values(['stock_code', 'timestamp']).reset_index(drop=True)
        
        print(f"✅ 전처리 완료")
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
            
            # 5. 호가 스프레드
            group['spread'] = (group['askp1'] - group['bidp1']) / (group['stck_prpr'] + 1e-8)
            
            # 6. 매수/매도 압력
            total_orders = group['total_askp_rsqn'] + group['total_bidp_rsqn'] + 1e-8
            group['buy_pressure'] = group['total_bidp_rsqn'] / total_orders
            
            # 7. 변동성 (최근 N틱의 표준편차)
            group['volatility_5'] = group['price_change_1'].rolling(window=5, min_periods=1).std()
            group['volatility_10'] = group['price_change_1'].rolling(window=10, min_periods=1).std()
            
            # 8. 가격 모멘텀 (현재가 - N틱 전 가격)
            group['momentum_5'] = group['stck_prpr'] - group['stck_prpr'].shift(5)
            group['momentum_10'] = group['stck_prpr'] - group['stck_prpr'].shift(10)
            
            # 9. 타겟 생성 (다음 N틱 후 가격 상승 여부)
            # 여기서는 5틱 후 가격이 현재보다 높으면 1, 낮으면 0
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
        df = df.dropna()
        
        if len(df) == 0:
            print("❌ 유효한 데이터가 없습니다.")
            return None, None, None
        
        # 4. 피쳐 선택 (GCN 임베딩은 나중에 merge)
        feature_cols = [
            'prdy_ctrt',           # 전일대비율
            'price_change_1', 'price_change_5', 'price_change_10',
            'price_vs_ma5', 'price_vs_ma20',
            'volume_ratio',
            'spread',
            'buy_pressure',
            'volatility_5', 'volatility_10',
            'momentum_5', 'momentum_10'
        ]
        
        X = df[feature_cols]
        y = df['target']
        stock_codes = df['stock_code']
        
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