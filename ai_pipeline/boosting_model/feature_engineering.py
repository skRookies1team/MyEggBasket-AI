import pandas as pd
import numpy as np
import torch
import json
import os
import sys

# 프로젝트 루트 경로
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
            return None
        
        # CPU/GPU 호환성 위해 map_location 추가
        self.gcn_embeddings = torch.load(embedding_path, map_location=torch.device('cpu'), weights_only=True)
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
            if node_id.isdigit():
                # node_id가 '005930' 형태라고 가정
                stock_mapping[node_id] = int(idx)
        
        self.stock_mapping = stock_mapping
        print(f"✅ 종목 매핑 정보 로드: {len(stock_mapping)}개 종목")
        return stock_mapping
    
    def merge_gcn_embeddings(self, X, stock_codes):
        """
        [성능 최적화됨] GCN 임베딩을 체결 정보 피쳐에 병합
        """
        print("\n🔗 GCN 임베딩 병합 중...")
        
        if self.gcn_embeddings is None:
            self.load_gcn_embeddings()
        
        if self.stock_mapping is None:
            self.load_stock_mapping()
        
        if self.gcn_embeddings is None:
            print("⚠️ GCN 임베딩이 없어 병합을 건너뜁니다.")
            return X
        
        # 1. 임베딩 차원 확인
        emb_dim = self.gcn_embeddings.shape[1]
        emb_cols = [f'gcn_emb_{i}' for i in range(emb_dim)]
        
        # 2. 고속 매핑을 위한 리스트 생성
        # for loop + iloc 대신 리스트 컴프리헨션 사용 (속도 대폭 향상)
        emb_list = []
        zero_emb = np.zeros(emb_dim) # 매칭 안 될 경우 0으로 채움
        matched_count = 0
        
        # numpy 변환 (속도 향상)
        gcn_emb_numpy = self.gcn_embeddings.numpy()
        
        for code in stock_codes:
            # 코드 포맷팅 (660 -> '000660')
            str_code = str(code).zfill(6)
            
            if str_code in self.stock_mapping:
                node_idx = self.stock_mapping[str_code]
                emb_list.append(gcn_emb_numpy[node_idx])
                matched_count += 1
            else:
                emb_list.append(zero_emb)

        # 3. 데이터프레임으로 변환 후 병합
        emb_df = pd.DataFrame(emb_list, columns=emb_cols, index=X.index)
        
        # 기존 X에 붙이기 (concat)
        X_final = pd.concat([X, emb_df], axis=1)
        
        print(f"✅ GCN 임베딩 병합 완료")
        print(f"   매칭된 종목: {matched_count}/{len(stock_codes)} ({(matched_count/len(stock_codes))*100:.1f}%)")
        
        return X_final
    
    def create_final_features(self):
        """
        최종 학습 데이터 생성
        [수정] 반환값에 stock_codes 추가 (총 3개 반환)
        """
        print("\n" + "="*60)
        print("🏗️ 최종 피처 생성 시작")
        print("="*60)
        
        if self.csv_path is None:
            raise ValueError("CSV 파일 경로가 지정되지 않았습니다.")
        
        # 1. 체결 정보 로드 (여기서 stock_codes를 받아옴)
        loader = RealtimeFeatureLoader(self.csv_path)
        
        # RealtimeFeatureLoader가 3개를 반환한다고 가정
        try:
            load_result = loader.prepare_features()
            if len(load_result) == 3:
                X, y, stock_codes = load_result
            else:
                # 만약 2개만 반환한다면 (예외 처리)
                X, y = load_result
                stock_codes = [] # 코드를 알 수 없음
        except Exception as e:
            print(f"❌ FeatureLoader 오류: {e}")
            return [], [], []
        
        if X is None or X.empty:
            return None, None, None
        
        # 2. GCN 임베딩 병합 (최적화된 함수 호출)
        X = self.merge_gcn_embeddings(X, stock_codes)
        
        # 3. 최종 정보 출력
        print(f"\n✅ 최종 피처 생성 완료!")
        print(f"   총 샘플 수: {len(X):,}")
        print(f"   피처 개수: {X.shape[1]}")
        
        if self.gcn_embeddings is not None:
             print(f"   - GCN 임베딩 포함됨 ({self.gcn_embeddings.shape[1]}차원)")
             
        if y is not None:
            print(f"   타겟 분포:")
            print(f"      하락(0): {(y==0).sum():,}개")
            print(f"      상승(1): {(y==1).sum():,}개")
        print("="*60)
        
        # [핵심 수정] 반드시 3개를 반환해야 함!
        return X, y, stock_codes

if __name__ == "__main__":
    csv_path = r"C:\Users\user\project\MyEggBasket-AI\20251120.csv"
    engineer = FeatureEngineer(csv_path=csv_path)
    
    # 3개 받기 테스트
    result = engineer.create_final_features()
    
    if result and result[0] is not None:
        X, y, codes = result
        print("\n[테스트 성공] 반환값 개수 확인 완료")
        print(f"X shape: {X.shape}, Codes length: {len(codes)}")