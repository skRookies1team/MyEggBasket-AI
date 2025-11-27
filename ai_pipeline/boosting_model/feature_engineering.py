import pandas as pd
import numpy as np
import torch
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.boosting_model.realtime_feature_loader import RealtimeFeatureLoader

class FeatureEngineer:
    """
    체결 정보 피쳐와 GCN 임베딩을 결합하여 최종 학습 피처를 생성합니다.
    """
    
    def __init__(self, csv_path=None):
        self.gcn_embeddings = None
        self.stock_mapping = None
        self.csv_path = csv_path
    
    def load_gcn_embeddings(self):
        """저장된 GCN 임베딩 로드"""
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
    
    def merge_gcn_embeddings(self, X, stock_codes):
        """
        GCN 임베딩을 체결 정보 피쳐에 병합
        
        Parameters:
        - X: 기존 피쳐 DataFrame
        - stock_codes: 각 행에 해당하는 종목 코드 Series
        
        Returns:
        - DataFrame: GCN 임베딩이 추가된 피쳐
        """
        print("\n🔗 GCN 임베딩 병합 중...")
        
        if self.gcn_embeddings is None:
            self.load_gcn_embeddings()
        
        if self.stock_mapping is None:
            self.load_stock_mapping()
        
        if self.gcn_embeddings is None:
            raise FileNotFoundError("GCN 임베딩을 로드할 수 없습니다.")
        
        # GCN 임베딩 컬럼 초기화
        emb_dim = self.gcn_embeddings.shape[1]
        for i in range(emb_dim):
            X[f'gcn_emb_{i}'] = 0.0
        
        # 종목별로 GCN 임베딩 매핑
        matched = 0
        for idx, stock_code in enumerate(stock_codes):
            if stock_code in self.stock_mapping:
                node_idx = self.stock_mapping[stock_code]
                embedding = self.gcn_embeddings[node_idx].numpy()
                
                for i, val in enumerate(embedding):
                    X.iloc[idx, X.columns.get_loc(f'gcn_emb_{i}')] = val
                
                matched += 1
        
        print(f"✅ GCN 임베딩 병합 완료")
        print(f"   매칭된 종목: {matched}/{len(stock_codes)} ({matched/len(stock_codes)*100:.1f}%)")
        
        return X
    
    def create_final_features(self):
        """
        최종 학습 데이터 생성
        
        Returns:
        - X: 피처 DataFrame (체결정보 + GCN 임베딩)
        - y: 타겟 Series
        """
        print("\n" + "="*60)
        print("🏗️ 최종 피처 생성 시작")
        print("="*60)
        
        if self.csv_path is None:
            raise ValueError("CSV 파일 경로가 지정되지 않았습니다.")
        
        # 1. 체결 정보 로드 및 기술적 지표 생성
        loader = RealtimeFeatureLoader(self.csv_path)
        X, y, stock_codes = loader.prepare_features()
        
        if X is None:
            return None, None
        
        # 2. GCN 임베딩 병합
        X = self.merge_gcn_embeddings(X, stock_codes)
        
        # 3. 최종 정보 출력
        print(f"\n✅ 최종 피처 생성 완료!")
        print(f"   총 샘플 수: {len(X):,}")
        print(f"   피처 개수: {X.shape[1]}")
        print(f"   - 체결 정보 피쳐: {X.shape[1] - self.gcn_embeddings.shape[1]}")
        print(f"   - GCN 임베딩: {self.gcn_embeddings.shape[1]}")
        print(f"   타겟 분포:")
        print(f"      하락(0): {(y==0).sum():,}개 ({(y==0).sum()/len(y)*100:.1f}%)")
        print(f"      상승(1): {(y==1).sum():,}개 ({(y==1).sum()/len(y)*100:.1f}%)")
        print("="*60)
        
        return X, y


# 실행 예시
if __name__ == "__main__":
    csv_path = r"C:\Users\user\project\MyEggBasket-AI\20251120.csv"
    
    engineer = FeatureEngineer(csv_path=csv_path)
    X, y = engineer.create_final_features()
    
    if X is not None:
        print("\n[최종 피처 샘플]")
        print(X.head())
        print(f"\n[피처 컬럼]")
        print(X.columns.tolist())