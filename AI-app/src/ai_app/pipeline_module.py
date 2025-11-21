import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from transformers import pipeline as hf_pipeline

# ----------------------------------------------------------------------
# 1. NLP 모듈: 텍스트 감성 점수 추출
# ----------------------------------------------------------------------
class NLPExtractor:
    """Hugging Face 기반의 KR-FINBERT를 사용하여 감성 점수를 추출하는 모듈."""
    def __init__(self, model_name="snunlp/KR-FINBERT-SC"):
        print(f"✅ NLP 모듈 초기화: {model_name} 로드 중...")
        # 'text-classification' 파이프라인 로드
        self.classifier = hf_pipeline("text-classification", model=model_name)
        print("✅ NLP 모듈 로드 완료.")

    def extract_sentiment(self, texts: list[str]) -> pd.DataFrame:
        """
        텍스트 리스트를 입력받아 감성 점수를 반환합니다.

        반환: pd.DataFrame (label, score)
        """
        if not texts:
            return pd.DataFrame(columns=['label', 'score'])
        
        # FinBERT 추론 실행
        results = self.classifier(texts, truncation=True)
        
        scores = [{
            'sentiment_label': res['label'], 
            'sentiment_score': res['score'] * (1 if res['label'] == '긍정' else -1) # 긍정: +, 부정: -
        } for res in results]
        
        return pd.DataFrame(scores)

# ----------------------------------------------------------------------
# 2. GCN 모듈: Node Embedding 추출
# ----------------------------------------------------------------------
class GCNFeatureExtractor(torch.nn.Module):
    """
    GCN 모델 구조 정의. 이 모델의 최종 출력 (32차원 벡터)이
    'Node Embedding'으로 XGBoost/LightGBM으로 전달됩니다.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        print(f"✅ GCN 초기화: 입력={in_channels}, 임베딩 차원={out_channels}")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCNPipeline:
    """GCN 모델을 통해 초기 피처와 관계로부터 Node Embedding을 추출하는 파이프라인."""
    def __init__(self, in_dim: int, out_dim: int = 32, model_path: str = None):
        self.model = GCNFeatureExtractor(in_channels=in_dim, hidden_channels=64, out_channels=out_dim)
        
        # 학습된 가중치 로드 (실제 운영 환경)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
            except FileNotFoundError:
                print(f"경고: 학습된 GCN 가중치 파일({model_path})을 찾을 수 없습니다. 무작위 가중치로 실행됩니다.")

    def _create_graph_data(self, features_df: pd.DataFrame, relations_df: pd.DataFrame) -> Data:
        """DataFrame을 PyG Data 객체로 변환합니다."""
        
        # features_df의 인덱스를 노드 ID로 사용한다고 가정합니다.
        # 1. Node Features (초기 피처) 생성
        features_tensor = torch.tensor(features_df.values, dtype=torch.float)
        
        # 2. Edge Index 생성 (종목 관계)
        # DataFrame 인덱스(0부터 시작)를 사용해야 합니다.
        node_map = {ticker: i for i, ticker in enumerate(features_df.index)}
        
        source_nodes = relations_df['from_ticker'].map(node_map).values
        target_nodes = relations_df['to_ticker'].map(node_map).values
        
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        
        data = Data(x=features_tensor, edge_index=edge_index)
        print(f"✅ 그래프 데이터 생성 완료. 노드 수: {data.num_nodes}, 엣지 수: {data.num_edges}")
        return data

    def extract_embedding(self, initial_features: pd.DataFrame, relations_df: pd.DataFrame) -> pd.DataFrame:
        """
        GCN 추론을 통해 Node Embedding을 추출하고 DataFrame 형태로 반환합니다.
        """
        graph_data = self._create_graph_data(initial_features, relations_df)
        
        self.model.eval() # 추론 모드
        with torch.no_grad():
            # [노드 수, out_dim] 형태의 Node Embedding 텐서
            node_embeddings = self.model(graph_data.x, graph_data.edge_index).numpy()

        # 결과를 DataFrame으로 변환하여 반환
        embedding_df = pd.DataFrame(
            node_embeddings,
            index=initial_features.index,
            columns=[f'gcn_emb_{i}' for i in range(self.model.conv2.out_channels)]
        )
        print(f"✅ GCN Node Embedding 추출 완료. 형태: {embedding_df.shape}")
        return embedding_df

# ----------------------------------------------------------------------
# 3. XGBoost 모듈: 최종 학습 및 예측
# ----------------------------------------------------------------------
class XGBoostModel:
    """최종 피처를 입력받아 XGBoost를 학습/추론하는 모듈."""
    def __init__(self, params: dict):
        self.model = xgb.XGBClassifier(**params, random_state=42)

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, test_size=0.2):
        """데이터를 분할하고 모델을 학습 및 평가합니다."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,    
            random_state=42,  
            stratify=y 
        )
        
        print(f"✅ XGBoost 학습 데이터 크기: {len(X_train)}")
        self.model.fit(X_train, y_train) 
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"--- XGBoost 결과 ---")
        print(f"테스트 데이터 정확도 (Accuracy): {accuracy:.4f}")
        print(f"----------------------")
        return accuracy