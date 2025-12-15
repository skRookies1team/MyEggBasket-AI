import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from torch_geometric.data import Data

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

es = Elasticsearch(ES_HOST)

def load_graph_data():
    print(" 그래프 데이터(CSV/JSON) 로딩 중...")

    # 1. 파일 경로 정의
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    edge_path = os.path.join(project_root, "data", "graph_edges.csv")
    map_path = os.path.join(project_root, "node_mapping.json")

    # 만약 루트에 없으면 data 폴더도 확인
    if not os.path.exists(map_path):
        map_path = os.path.join(project_root, "data", "node_mapping.json")

    if not os.path.exists(edge_path) or not os.path.exists(map_path):
        print(f" 파일이 없습니다.\n 1. {edge_path}\n 2. {map_path}")
        return None

    # 2. 로딩
    edges_df = pd.read_csv(edge_path, dtype=str)
    with open(map_path, "r", encoding='utf-8') as f:
        idx_to_node = json.load(f)

    # JSON 키(문자열) -> 정수 변환
    idx_to_node_int = {int(k): v for k, v in idx_to_node.items()}

    return edges_df, idx_to_node_int


def get_node_features(idx_to_node):
    """
    각 노드(점)가 가질 능력치(Feature) 벡터 생성
    - 차원: 3 [Sentiment_Score, Is_News, Is_Stock]
    - 수정사항: 부스팅 모델에서 시계열로 처리할 변동성/트렌드는 여기서 제외함
    """
    print(" 노드 특징(Feature) 벡터 생성 중 (ES 조회 포함)...")

    num_nodes = len(idx_to_node)
    x = torch.zeros((num_nodes, 3), dtype=torch.float)

    for idx in range(num_nodes):
        node_id = str(idx_to_node[idx])

        # 1. 종목 코드인 경우 (숫자로만 구성됨)
        if node_id.isdigit():
            x[idx, 0] = 0.0  # 감성점수 없음
            x[idx, 1] = 0.0  # 뉴스 아님
            x[idx, 2] = 1.0  # 주식 맞음

        # 2. 뉴스 노드인 경우
        else:
            sentiment_val = 0.0
            if es:
                try:
                    res = es.get(index="news_articles", id=node_id)
                    source = res['_source']

                    # 감성 점수 추출
                    if 'sentiment_score' in source:
                        sentiment_val = float(source['sentiment_score'])
                    elif 'analysis_results' in source and source['analysis_results']:
                        sentiment_val = float(source['analysis_results'][0].get('sentiment_score', 0.0))

                except Exception:
                    sentiment_val = 0.0

            x[idx, 0] = sentiment_val
            x[idx, 1] = 1.0  # 뉴스 맞음
            x[idx, 2] = 0.0  # 주식 아님

    print(" 노드 특징 생성 완료.")
    return x


def create_pytorch_dataset():
    # 1. 데이터 로드
    data_pack = load_graph_data()
    if not data_pack: return

    edges_df, idx_to_node = data_pack

    # ------------------------------------------------------------------
    # [수정] 종목 매핑 정보 생성 (조건 완화 및 정규화)
    # ------------------------------------------------------------------
    print(" 엣지 데이터 인덱싱 변환 중...")
    stock_to_idx = {}

    for idx, node_id in idx_to_node.items():
        node_str = str(node_id).strip()

        # 길이가 6이 아니더라도 숫자면 주식 코드로 인정하고 0을 채워 저장
        # 예: "5930" -> "005930"
        if node_str.isdigit():
            norm_code = node_str.zfill(6)
            stock_to_idx[norm_code] = idx

    print(f" 종목 매핑 정보 생성 완료 ({len(stock_to_idx)}개 종목)")

    # ------------------------------------------------------------------
    # 엣지 인덱스 변환
    # ------------------------------------------------------------------
    node_to_idx = {v: k for k, v in idx_to_node.items()}

    edges_df['source_idx'] = edges_df['source'].astype(str).map(node_to_idx)
    edges_df['target_idx'] = edges_df['target'].astype(str).map(node_to_idx)

    # 매핑 안된 엣지 제거
    edges_df = edges_df.dropna(subset=['source_idx', 'target_idx'])

    source_tensor = torch.tensor(edges_df['source_idx'].values.astype(int), dtype=torch.long)
    target_tensor = torch.tensor(edges_df['target_idx'].values.astype(int), dtype=torch.long)
    edge_index = torch.stack([source_tensor, target_tensor], dim=0)

    # 3. 노드 특징 생성
    x = get_node_features(idx_to_node)

    # 4. PyG 데이터 객체 생성 및 저장
    data = Data(x=x, edge_index=edge_index)
    data.stock_to_idx = stock_to_idx  # [중요] 복구된 매핑 저장

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    save_path = os.path.join(project_root, "finance_graph_data.pt")

    torch.save(data, save_path)

    print("-" * 50)
    print(f" 데이터셋 변환 완료!")
    print(f" 저장 경로: {save_path}")
    print(f" 노드: {data.num_nodes}개 / 엣지: {data.num_edges}개")
    print("-" * 50)


if __name__ == "__main__":
    create_pytorch_dataset()