import pandas as pd
import numpy as np
import torch
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

class FeatureEngineer:
    """
    주가 데이터와 GCN 임베딩을 결합하여 최종 학습 피처를 생성합니다.
    """
    
    def __init__(self):
        self.gcn_embeddings = None
        self.stock_mapping = None
    
    def load_gcn_embeddings(self):
        """저장된 GCN 임베딩 로드"""
        # 프로젝트 루트 경로에서 찾기
        current_dir = os.path.dirname(__file__)
        root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
        embedding_path = os.path.join(root_dir, "gcn_node_embeddings.pt")
        
        if not os.path.exists(embedding_path):
            print(f"❌ GCN 임베딩 파일이 없습니다: {embedding_path}")
            print(f"   먼저 다음 명령을 실행하세요:")
            print(f"   cd {root_dir}")
            print(f"   python ai_pipeline/pipeline_main.py")
            return None
        
        self.gcn_embeddings = torch.load(embedding_path, weights_only=True)
        print(f"✅ GCN 임베딩 로드 완료: {self.gcn_embeddings.shape}")
        return self.gcn_embeddings
    
    def load_stock_mapping(self):
        """종목 코드 → GCN 노드 인덱스 매핑 로드"""
        mapping_path = os.path.join(
            os.path.dirname(__file__), 
            "../graph_build/node_mapping.json"
        )
        
        if not os.path.exists(mapping_path):
            print(f"❌ 노드 매핑 파일이 없습니다: {mapping_path}")
            return None
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            idx_to_node = json.load(f)
        
        # 종목 코드만 추출 (6자리 숫자)
        stock_mapping = {}
        for idx, node_id in idx_to_node.items():
            if node_id.isdigit() and len(node_id) == 6:
                stock_mapping[node_id] = int(idx)
        
        self.stock_mapping = stock_mapping
        print(f"✅ 종목 매핑: {len(stock_mapping)}개 종목")
        return stock_mapping
    
    def create_dummy_stock_data(self, stock_codes, days=252):
        """
        더미 주가 데이터 생성 (실제 API 연동 전 테스트용)
        
        Parameters:
        - stock_codes: 종목 코드 리스트
        - days: 생성할 데이터 일수 (252일 = 1년 영업일)
        
        Returns:
        - DataFrame: 주가 피처가 포함된 데이터
        """
        print(f"📊 더미 주가 데이터 생성 중... ({len(stock_codes)}개 종목, {days}일)")
        
        all_data = []
        
        for stock_code in stock_codes:
            # 기본 가격 설정 (종목마다 다르게)
            base_price = np.random.randint(10000, 100000)
            
            # 일자별 데이터 생성
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            for i, date in enumerate(dates):
                # 가격 랜덤워크 (전날 대비 ±5% 변동)
                if i == 0:
                    close = base_price
                else:
                    change_pct = np.random.uniform(-0.05, 0.05)
                    close = all_data[-1]['Close'] * (1 + change_pct)
                
                open_price = close * np.random.uniform(0.98, 1.02)
                high = max(open_price, close) * np.random.uniform(1.0, 1.03)
                low = min(open_price, close) * np.random.uniform(0.97, 1.0)
                volume = np.random.randint(100000, 1000000)
                
                all_data.append({
                    'stock_code': stock_code,
                    'Date': date,
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close,
                    'Volume': volume
                })
        
        df = pd.DataFrame(all_data)
        print(f"✅ 더미 데이터 생성 완료: {len(df)}개 행")
        return df
    
    def add_technical_indicators(self, df):
        """기술적 지표 추가"""
        print("📈 기술적 지표 계산 중...")
        
        result_dfs = []
        
        for stock_code, group in df.groupby('stock_code'):
            group = group.sort_values('Date').copy()
            
            # 1. 수익률
            group['return_1d'] = group['Close'].pct_change(1)
            group['return_5d'] = group['Close'].pct_change(5)
            group['return_20d'] = group['Close'].pct_change(20)
            
            # 2. 이동평균
            group['ma_5'] = group['Close'].rolling(window=5).mean()
            group['ma_20'] = group['Close'].rolling(window=20).mean()
            group['ma_60'] = group['Close'].rolling(window=60).mean()
            
            # 3. 이동평균 간 거리
            group['ma_gap_5_20'] = (group['ma_5'] - group['ma_20']) / group['ma_20']
            group['ma_gap_20_60'] = (group['ma_20'] - group['ma_60']) / group['ma_60']
            
            # 4. RSI
            delta = group['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            group['rsi'] = 100 - (100 / (1 + rs))
            
            # 5. 거래량 비율
            group['volume_ma_5'] = group['Volume'].rolling(window=5).mean()
            group['volume_ratio'] = group['Volume'] / group['volume_ma_5']
            
            # 6. 변동성
            group['volatility_20'] = group['return_1d'].rolling(window=20).std()
            
            # 7. 타겟 변수 (다음 날 상승=1, 하락=0)
            group['target'] = (group['Close'].shift(-1) > group['Close']).astype(int)
            
            result_dfs.append(group)
        
        final_df = pd.concat(result_dfs, ignore_index=True)
        print(f"✅ 기술적 지표 추가 완료")
        return final_df
    
    def merge_gcn_embeddings(self, df):
        """GCN 임베딩을 주가 데이터에 병합"""
        print("🔗 GCN 임베딩 병합 중...")
        
        if self.gcn_embeddings is None:
            self.load_gcn_embeddings()
        
        if self.stock_mapping is None:
            self.load_stock_mapping()
        
        # GCN 임베딩이 로드되지 않았으면 에러 처리
        if self.gcn_embeddings is None:
            print("❌ GCN 임베딩을 로드할 수 없습니다.")
            print("   파이프라인을 먼저 실행하여 gcn_node_embeddings.pt를 생성하세요.")
            raise FileNotFoundError("gcn_node_embeddings.pt 파일이 필요합니다.")
        
        # GCN 임베딩 추가
        for i in range(self.gcn_embeddings.shape[1]):  # 16차원
            df[f'gcn_emb_{i}'] = 0.0
        
        for stock_code in df['stock_code'].unique():
            if stock_code in self.stock_mapping:
                node_idx = self.stock_mapping[stock_code]
                embedding = self.gcn_embeddings[node_idx].numpy()
                
                # 해당 종목의 모든 행에 임베딩 추가
                for i, val in enumerate(embedding):
                    df.loc[df['stock_code'] == stock_code, f'gcn_emb_{i}'] = val
            else:
                print(f"⚠️ {stock_code}는 GCN 그래프에 없습니다.")
        
        print(f"✅ GCN 임베딩 병합 완료")
        return df
    
    def create_final_features(self, stock_codes, use_dummy=True):
        """
        최종 학습 데이터 생성
        
        Returns:
        - X: 피처 DataFrame
        - y: 타겟 Series
        """
        print("\n" + "="*60)
        print("🏗️ 최종 피처 생성 시작")
        print("="*60)
        
        # 1. 주가 데이터 생성/로드
        if use_dummy:
            df = self.create_dummy_stock_data(stock_codes, days=252)
        else:
            # 실제 API로 데이터 받아오는 코드 (추후 구현)
            pass
        
        # 2. 기술적 지표 추가
        df = self.add_technical_indicators(df)
        
        # 3. GCN 임베딩 병합
        df = self.merge_gcn_embeddings(df)
        
        # 4. NaN 제거
        df = df.dropna()
        
        if len(df) == 0:
            print("❌ 유효한 데이터가 없습니다.")
            return None, None
        
        # 5. X, y 분리
        feature_cols = [col for col in df.columns 
                       if col not in ['target', 'stock_code', 'Date', 
                                     'Open', 'High', 'Low', 'Close', 'Volume']]
        
        X = df[feature_cols]
        y = df['target']
        
        print(f"\n✅ 최종 피처 생성 완료!")
        print(f"   총 샘플 수: {len(X):,}")
        print(f"   피처 개수: {len(feature_cols)}")
        print(f"   타겟 분포:")
        print(f"      하락(0): {(y==0).sum():,}개 ({(y==0).sum()/len(y)*100:.1f}%)")
        print(f"      상승(1): {(y==1).sum():,}개 ({(y==1).sum()/len(y)*100:.1f}%)")
        print("="*60)
        
        return X, y


# 실행 예시
if __name__ == "__main__":
    # GCN에서 나온 종목 코드 사용
    engineer = FeatureEngineer()
    engineer.load_stock_mapping()
    
    # 실제 GCN 그래프에 있는 종목만 사용
    stock_codes = list(engineer.stock_mapping.keys())[:20]  # 상위 20개만 테스트
    
    print(f"📌 사용할 종목: {stock_codes}")
    
    X, y = engineer.create_final_features(stock_codes, use_dummy=True)
    
    if X is not None:
        print("\n[피처 샘플]")
        print(X.head())
        print(f"\n[피처 컬럼]")
        print(X.columns.tolist())