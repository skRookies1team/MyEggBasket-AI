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

es = Elasticsearch("http://localhost:9200")

def load_graph_data():
    print(" 그래프 데이터(CSV/JSON) 로딩 중...")
    
    # 1. 파일 경로 정의 (data 폴더 확인)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # build_edges.py가 저장한 경로와 일치시킴
    edge_path = os.path.join(project_root, "data", "graph_edges.csv")
    map_path = os.path.join(project_root, "node_mapping.json") # 루트에 있다고 가정

    # 만약 루트에 없으면 data 폴더도 확인
    if not os.path.exists(map_path):
        map_path = os.path.join(project_root, "data", "node_mapping.json")

    if not os.path.exists(edge_path) or not os.path.exists(map_path):
        print(f" 파일이 없습니다.\n 1. {edge_path}\n 2. {map_path}")
        return None

    # 2. 로딩
    edges_df = pd.read_csv(edge_path, dtype=str) # 코드가 '005930' 처럼 읽히도록 문자열 강제
    with open(map_path, "r", encoding='utf-8') as f:
        idx_to_node = json.load(f) # "0": "news_id...", "1": "005930"
    
    # JSON 키(문자열) -> 정수 변환: {0: "news_id...", 1: "005930"}
    idx_to_node_int = {int(k): v for k, v in idx_to_node.items()}
    
    return edges_df, idx_to_node_int

def get_node_features(idx_to_node):
    """
    각 노드(점)가 가질 능력치(Feature) 벡터를 만듭니다.
    """
    print(" 노드 특징(Feature) 벡터 생성 중 (ES 조회 포함)...")
    
    num_nodes = len(idx_to_node)
    x = torch.zeros((num_nodes, 3), dtype=torch.float)
    
    for idx in range(num_nodes):
        node_id = idx_to_node[idx]
        
        # 1. 종목 코드인 경우
        if node_id.isdigit() and len(node_id) == 6:
            x[idx, 0] = 0.0
            x[idx, 1] = 0.0
            x[idx, 2] = 1.0
            
        # 2. 뉴스 노드인 경우
        else:
            if es:
                try:
                    res = es.get(index="news_articles", id=node_id)
                    sentiments = res['_source'].get('sentiments', [0.0])
                    avg_sentiment = np.mean(sentiments) if sentiments else 0.0
                    
                    x[idx, 0] = avg_sentiment
                    x[idx, 1] = 1.0
                    x[idx, 2] = 0.0
                except:
                    x[idx, 0] = 0.0
                    x[idx, 1] = 1.0
                    x[idx, 2] = 0.0
            else:
                x[idx, 0] = 0.0
                x[idx, 1] = 1.0
                x[idx, 2] = 0.0

    print(" 노드 특징 생성 완료.")
    return x

def create_pytorch_dataset():
    # 1. 데이터 로드
    data_pack = load_graph_data()
    if not data_pack: return
    
    edges_df, idx_to_node = data_pack
    
    # ------------------------------------------------------------------
    # 🚑 [핵심 수정] 문자열(코드)을 정수(인덱스)로 변환 (Mapping)
    # ------------------------------------------------------------------
    print(" 엣지 데이터 인덱싱 변환 중 (String -> Integer)...")
    
    # 1. 역방향 매핑 생성 (종목코드 -> 인덱스)
    # 예: {"005930": 1, "news_123": 0}
    node_to_idx = {v: k for k, v in idx_to_node.items()}
    
    # 2. DataFrame의 문자열을 숫자로 치환
    # map 함수를 써서 '005930'을 1로 바꿉니다.
    edges_df['source_idx'] = edges_df['source'].astype(str).map(node_to_idx)
    edges_df['target_idx'] = edges_df['target'].astype(str).map(node_to_idx)
    
    # 3. 매핑 실패한(NaN) 행 제거 (혹시 모를 오류 방지)
    initial_len = len(edges_df)
    edges_df = edges_df.dropna(subset=['source_idx', 'target_idx'])
    
    if len(edges_df) < initial_len:
        print(f" 경고: {initial_len - len(edges_df)}개의 엣지가 매핑되지 않아 제거되었습니다.")
        
    # 4. 정수형으로 변환 (float -> int)
    source_tensor = torch.tensor(edges_df['source_idx'].values.astype(int), dtype=torch.long)
    target_tensor = torch.tensor(edges_df['target_idx'].values.astype(int), dtype=torch.long)
    
    # 5. Edge Index 생성
    edge_index = torch.stack([source_tensor, target_tensor], dim=0)
    # ------------------------------------------------------------------
    
    # 3. 노드 특징 텐서 생성
    x = get_node_features(idx_to_node)
    
    # 4. PyG 데이터 객체 생성
    data = Data(x=x, edge_index=edge_index)

    # 이름표(stock_to_idx) 저장 로직 (필수!)
    stock_to_idx = {}
    for idx, node_id in idx_to_node.items():
        if node_id.isdigit() and len(node_id) == 6:
            stock_to_idx[node_id] = idx
            
    data.stock_to_idx = stock_to_idx
    print(f" 종목 매핑 정보 저장 완료 ({len(stock_to_idx)}개 종목)")
    
    # 5. 저장
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