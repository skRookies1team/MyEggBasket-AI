import torch
import pandas as pd
import numpy as np
import os
import sys
import json

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# [중요] 모델 클래스 import
# 보통 run_gcn.py랑 같은 폴더나 model.py에 NewsStockGCN이 있을 겁니다.
# 만약 에러 나면 from ... import NewsStockGCN 부분을 사용자 환경에 맞게 수정해주세요!
try:
    from ai_pipeline.gcn_model.model import NewsStockGCN
except ImportError:
    # 혹시 model.py가 아니라면 run_gcn.py가 참조하는 곳을 찾아야 함
    # 임시로 여기에 클래스 정의가 필요할 수도 있음
    print("⚠️ NewsStockGCN 클래스를 찾을 수 없습니다. 경로를 확인해주세요.")
    pass

class GCNFeatureExtractor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("🧠 [GCN 추출기] 초기화 중...")

        # 1. 데이터 파일(.pt) 로드
        # run_gcn.py에 있던 경로 로직 그대로 차용
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pt_path = os.path.join(current_dir, "../../finance_graph_data.pt")
        
        if not os.path.exists(pt_path):
            print(f"❌ 데이터 파일이 없습니다: {pt_path}")
            self.data = None
            return

        # weights_only=False는 최신 토치 버전 경고 방지용
        try:
            self.data = torch.load(pt_path, weights_only=False)
        except:
            self.data = torch.load(pt_path) # 구버전 호환

        self.data = self.data.to(self.device)
        print(f"   ✅ 그래프 데이터 로드 완료 (노드 {self.data.num_nodes}개)")

        # 2. 모델 초기화 (run_gcn.py 설정 참고: in=3, hidden=16, out=16)
        self.model = NewsStockGCN(in_channels=3, hidden_channels=16, out_channels=16).to(self.device)
        
        # 3. 학습된 가중치 로드 (없으면 랜덤값 사용)
        if model_path is None:
            # 기본 경로: models 폴더 안의 베스트 모델
            model_path = os.path.join(current_dir, "models/gcn_best_model.pth")
            
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"   ✅ 학습된 모델 가중치 로드 완료")
        else:
            print(f"   ⚠️ 학습된 모델이 없어 '초기화 상태'로 진행합니다.")
            self.model.eval()


    def _load_json_mapping(self):
        """[비상용] JSON 파일에서 종목 매핑 로드"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            r"C:\Users\user\project\MyEggBasket-AI\node_mapping.json",
            os.path.join(current_dir, "../graph_build/node_mapping.json"),
            os.path.join(current_dir, "../../data/node_mapping.json"),
            os.path.join(current_dir, "../../node_mapping.json")
        ]
        
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        mapping = json.load(f) 
                        return {int(v): k for k, v in mapping.items()}
                except:
                    pass
        return None

    def get_embeddings(self):
        """
        모델을 실행하여 임베딩을 뽑고, 종목 코드와 매핑합니다.
        """
        if self.data is None: return {}

        # 1. 추론 (Inference)
        with torch.no_grad():
            # (노드 개수, 16) 크기의 텐서 출력
            embeddings = self.model(self.data)
            
        emb_np = embeddings.cpu().numpy()
        
        # 2. [핵심] 인덱스(0,1,2..)를 종목코드(005930..)로 변환
        # 보통 .pt 파일 안에 stock_to_idx 정보가 저장되어 있습니다.
        mapping = {}
        
        if hasattr(self.data, 'stock_to_idx'):
            # data.stock_to_idx 가 { '005930': 100, ... } 형태라고 가정
            stock_to_idx = self.data.stock_to_idx
            # 뒤집기: { 100: '005930' }
            idx_to_stock = {v: k for k, v in stock_to_idx.items()}
            
            for idx, vector in enumerate(emb_np):
                if idx in idx_to_stock:
                    code = idx_to_stock[idx]
                    mapping[code] = vector
        else:
            print("⚠️ 경고: 데이터 파일에 'stock_to_idx' 매핑 정보가 없습니다.")
            print("   -> 임베딩은 뽑았지만 어떤 종목인지 알 수 없어 병합이 불가능합니다.")
            print("   -> 그래프 생성 코드(create_graph.py)에서 data.stock_to_idx = ... 를 저장했는지 확인하세요.")
            
        return mapping

    def add_gcn_features(self, df, code_col='code'):
        """
        DataFrame에 GCN 피처(gcn_0 ~ gcn_15)를 붙여줍니다.
        """
        # 데이터프레임의 종목코드 컬럼 이름 확인 (stck_shrn_iscd 등)
        target_col = code_col
        if 'stck_shrn_iscd' in df.columns:
            target_col = 'stck_shrn_iscd'
        elif 'code' in df.columns:
            target_col = 'code'
            
        print(f"🧬 [{target_col}] 기준으로 GCN 피처 병합 시작...")
        
        emb_dict = self.get_embeddings()
        if not emb_dict:
            return df

        # 딕셔너리 -> 데이터프레임 변환
        emb_df = pd.DataFrame.from_dict(emb_dict, orient='index')
        emb_df.columns = [f'gcn_{i}' for i in range(emb_df.shape[1])]
        emb_df.index.name = target_col
        emb_df = emb_df.reset_index()
        
        # 원본과 병합 (Left Join)
        # 문자열/숫자 타입 불일치 방지
        df[target_col] = df[target_col].astype(str).str.strip()
        emb_df[target_col] = emb_df[target_col].astype(str).str.strip()
        
        merged_df = pd.merge(df, emb_df, on=target_col, how='left')
        
        # GCN 정보가 없는 종목은 0으로 채움
        gcn_cols = [c for c in merged_df.columns if c.startswith('gcn_')]
        merged_df[gcn_cols] = merged_df[gcn_cols].fillna(0)
        
        print(f"✅ 병합 완료! (총 {len(gcn_cols)}개 피처 추가됨)")
        return merged_df
    


    

if __name__ == "__main__":
    # 테스트
    extractor = GCNFeatureExtractor()
    # 더미 데이터로 테스트
    dummy_df = pd.DataFrame({'stck_shrn_iscd': ['005930', '000660']})
    res = extractor.add_gcn_features(dummy_df)
    print(res.head())