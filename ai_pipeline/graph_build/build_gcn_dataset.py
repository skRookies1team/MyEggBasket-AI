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
    print("🔄 그래프 데이터(CSV/JSON) 로딩 중...")
    
    # 1. 파일 경로 정의
    current_dir = os.path.dirname(__file__)
    edge_path = os.path.join(current_dir, "graph_edges.csv")
    map_path = os.path.join(current_dir, "node_mapping.json")

    if not os.path.exists(edge_path) or not os.path.exists(map_path):
        print(f"❌ 파일이 없습니다. build_edges.py를 먼저 실행하세요.")
        return None

    # 2. 로딩
    edges_df = pd.read_csv(edge_path)
    with open(map_path, "r", encoding='utf-8') as f:
        idx_to_node = json.load(f) # "0": "news_id...", "1": "005930"
    
    # JSON의 키(Key)는 문자열로 저장되므로, 정수형(int)으로 변환한 맵이 필요
    idx_to_node_int = {int(k): v for k, v in idx_to_node.items()}
    
    return edges_df, idx_to_node_int

def get_node_features(idx_to_node):
    """
    각 노드(점)가 가질 능력치(Feature) 벡터를 만듭니다.
    - 뉴스 노드: [감성점수평균, 1, 0]
    - 종목 노드: [0.0, 0, 1]
    """
    print("📊 노드 특징(Feature) 벡터 생성 중 (ES 조회 포함)...")
    
    num_nodes = len(idx_to_node)
    # Feature Dimension = 3 (감성점수, Is_News, Is_Stock)
    x = torch.zeros((num_nodes, 3), dtype=torch.float)
    
    for idx in range(num_nodes):
        node_id = idx_to_node[idx]
        
        # 1. 종목 코드인 경우 (숫자 6자리) -> 종목 노드
        if node_id.isdigit() and len(node_id) == 6:
            # [0, 0, 1] 설정
            x[idx, 0] = 0.0  # 종목 자체의 감성은 일단 0 (나중에 주가 등락률로 대체 가능)
            x[idx, 1] = 0.0  # 뉴스 아님
            x[idx, 2] = 1.0  # 종목 맞음
            
        # 2. 그 외 -> 뉴스 노드 (ES ID)
        else:
            # ES에서 감성 점수 가져오기
            try:
                res = es.get(index="news_articles", id=node_id)
                sentiments = res['_source'].get('sentiments', [0.0])
                
                # 감성 점수 평균 내기
                avg_sentiment = np.mean(sentiments) if sentiments else 0.0
                
                # [감성점수, 1, 0] 설정
                x[idx, 0] = avg_sentiment
                x[idx, 1] = 1.0 # 뉴스 맞음
                x[idx, 2] = 0.0 # 종목 아님
                
            except Exception:
                # 데이터를 못 찾았으면 기본값
                x[idx, 0] = 0.0
                x[idx, 1] = 1.0
                x[idx, 2] = 0.0

    print("✅ 노드 특징 생성 완료.")
    return x

def create_pytorch_dataset():
    # 1. 데이터 로드
    data_pack = load_graph_data()
    if not data_pack: return
    
    edges_df, idx_to_node = data_pack
    
    # 2. 엣지 텐서 생성 (Edge Index)
    # shape: [2, num_edges]
    # PyG는 엣지를 (출발점들, 도착점들) 두 줄로 쌓은 형태를 원합니다.
    source = torch.tensor(edges_df['source'].values, dtype=torch.long)
    target = torch.tensor(edges_df['target'].values, dtype=torch.long)
    edge_index = torch.stack([source, target], dim=0)
    
    # 3. 노드 특징 텐서 생성 (Node Features 'x')
    x = get_node_features(idx_to_node)
    
    # 4. PyG 데이터 객체 생성
    data = Data(x=x, edge_index=edge_index)

    stock_to_idx = {}
    for idx, node_id in idx_to_node.items():
        # 종목코드(6자리 숫자)만 골라냄
        if node_id.isdigit() and len(node_id) == 6:
            stock_to_idx[node_id] = idx
            
    data.stock_to_idx = stock_to_idx  # 👈 여기에 저장!
    print(f"🏷️ 종목 매핑 정보 저장 완료 ({len(stock_to_idx)}개 종목)")
    
    # 5. 저장
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    save_path = os.path.join(root_dir, "finance_graph_data.pt")
    
    torch.save(data, save_path)
    
    print("-" * 50)
    print(f"🎉 데이터셋 변환 완료!")
    print(f"💾 저장 경로: {os.path.abspath(save_path)}")
    print(f"📊 데이터 정보: {data}")
    print("-" * 50)
    print("이제 이 .pt 파일을 GCN 모델에 넣어서 학습할 수 있습니다.")

if __name__ == "__main__":
    create_pytorch_dataset()