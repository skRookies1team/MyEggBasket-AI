import numpy as np
import pandas as pd
import os

class GraphFeatureLoader:
    def __init__(self, data_dir=None):
        if data_dir is None:
            # 기본 경로: 프로젝트 루트/data
            curr = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.abspath(os.path.join(curr, "../../data"))
        
        self.emb_path = os.path.join(data_dir, "gcn_embeddings.npy")
        self.node_path = os.path.join(data_dir, "gcn_node_list.csv")
        self.embedding_map = {}
        
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.emb_path) and os.path.exists(self.node_path):
            try:
                emb = np.load(self.emb_path)
                nodes = pd.read_csv(self.node_path, dtype=str)
                
                # 컬럼명 처리
                code_col = 'stock_code' if 'stock_code' in nodes.columns else nodes.columns[0]
                codes = nodes[code_col].str.strip().str.zfill(6).tolist()
                
                # 딕셔너리로 매핑 { '005930': [0.1, 0.2, ...] }
                for i, code in enumerate(codes):
                    if i < len(emb):
                        self.embedding_map[code] = emb[i]
                print(f" [Graph] GCN 임베딩 로드 완료: {len(self.embedding_map)}개 종목")
            except Exception as e:
                print(f" [Graph] 로드 실패: {e}")
        else:
            print(" [Graph] 임베딩 파일이 없습니다.")

    def get_embedding_features(self, stock_code):
        """
        해당 종목의 GCN 임베딩 벡터를 딕셔너리로 반환
        """
        stock_code = str(stock_code).zfill(6)
        embedding = self.embedding_map.get(stock_code, np.zeros(16)) # 없으면 0벡터
        
        return {f'gcn_emb_{i}': val for i, val in enumerate(embedding)}