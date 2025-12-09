import os
import sys
import torch
import pandas as pd
import numpy as np
import json

# 프로젝트 루트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from ai_pipeline.gcn_model.model import GCNEncoder


class GCNInference:
    def __init__(self, model_path=None, edge_path=None, mapping_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        # 1. 경로 설정 (기본값)
        if model_path is None:
            model_path = os.path.join(current_dir, "best_gcn_model.pth")
        if edge_path is None:
            edge_path = os.path.join(project_root, "ai_pipeline/graph_build/graph_edges.csv")
        if mapping_path is None:
            mapping_path = os.path.join(project_root, "ai_pipeline/graph_build/node_mapping.json")

        self.model_path = model_path
        self.edge_path = edge_path
        self.mapping_path = mapping_path

        # 2. 모델 및 데이터 로드
        self._load_resources()

    def _load_resources(self):
        # (1) 노드 매핑 로드
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, 'r', encoding='utf-8') as f:
                self.node_mapping = json.load(f)  # {'005930': 0, ...}
                self.inv_mapping = {v: k for k, v in self.node_mapping.items()}
        else:
            raise FileNotFoundError(f"노드 매핑 파일이 없습니다: {self.mapping_path}")

        # (2) 엣지 데이터 로드 및 텐서 변환
        if os.path.exists(self.edge_path):
            df_edges = pd.read_csv(self.edge_path, dtype=str)
            src_idx = [self.node_mapping[s] for s in df_edges['source'] if s in self.node_mapping]
            dst_idx = [self.node_mapping[t] for t in df_edges['target'] if t in self.node_mapping]

            # 주의: 양방향 그래프로 가정 (필요시 수정)
            self.edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long).to(self.device)
        else:
            raise FileNotFoundError(f"엣지 파일이 없습니다: {self.edge_path}")

        # (3) 모델 로드
        # 입력 차원은 학습 시 사용한 feature dimension과 같아야 함 (예: 16)
        # TODO: 실제 학습된 feature dimension에 맞춰 in_channels 수정 필요
        self.model = GCNEncoder(in_channels=16, out_channels=16).to(self.device)

        if os.path.exists(self.model_path):
            try:
                # weights_only=True 권장 (보안)
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()  # 평가 모드 설정 (Dropout 비활성화)
                print(" [GCN Inference] 학습된 모델 로드 완료")
            except Exception as e:
                print(f" [GCN Inference] 모델 로드 실패: {e}")
        else:
            print(" [GCN Inference] 경고: 학습된 모델 파일이 없습니다. 랜덤 가중치로 시작합니다.")

    def get_embeddings(self, feature_df):
        """
        feature_df: 종목코드가 포함된 현재 시점의 피처 DataFrame
        Returns: {종목코드: 임베딩벡터(numpy)} 딕셔너리
        """
        num_nodes = len(self.node_mapping)
        feature_dim = 16  # 학습시 설정한 차원

        # 1. 입력 텐서 생성 (Node Features)
        # 기본적으로 0으로 초기화
        x_features = np.zeros((num_nodes, feature_dim), dtype=np.float32)

        # DataFrame에 있는 값 채워넣기
        # 예: feature_df에 'sentiment_score', 'volatility' 등의 컬럼이 있다고 가정
        # 여기서는 단순화를 위해 임의의 피처를 사용하거나, ES에서 가져온 값을 매핑해야 함
        # *실제 구현시에는 feature_engineering.py 로직을 참고하여 값을 채워야 합니다*

        # (임시) 랜덤 노이즈 + 데이터가 있으면 매핑 (실전에서는 실제 피처 사용)
        x_features += np.random.randn(num_nodes, feature_dim) * 0.01

        x_tensor = torch.tensor(x_features, dtype=torch.float).to(self.device)

        # 2. 임베딩 추출 (Forward)
        with torch.no_grad():
            embeddings = self.model(x_tensor, self.edge_index)

        embeddings_np = embeddings.cpu().numpy()

        # 3. 딕셔너리로 변환
        result = {}
        for idx, vector in enumerate(embeddings_np):
            if idx in self.inv_mapping:
                code = self.inv_mapping[idx]
                result[code] = vector

        return result