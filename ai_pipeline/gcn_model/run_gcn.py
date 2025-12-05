import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from elasticsearch import Elasticsearch

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# ---------------------------------------------------------
# 1. GCN 모델 정의
# ---------------------------------------------------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # [수정] 16차원 출력을 위해 중간 레이어를 64에서 32로 조정 (선택사항, 64 유지 가능)
        self.conv1 = GCNConv(in_channels, 32) 
        self.conv2 = GCNConv(32, out_channels)
        self.dropout = nn.Dropout(p=0.3) 

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x 

# ---------------------------------------------------------
# 2. 데이터 로드 및 학습 실행
# ---------------------------------------------------------
def train_gcn():
    print(" 그래프 데이터 로딩 중...")
    
    # 엣지 데이터 로드
    edge_path = os.path.join(project_root, "graph_edges.csv")
    if not os.path.exists(edge_path):
        print(" 엣지 파일이 없습니다. build_edges.py를 먼저 실행하세요.")
        return

    df_edges = pd.read_csv(edge_path, dtype=str)
    all_nodes = sorted(list(set(df_edges['source']).union(set(df_edges['target']))))
    node_to_idx = {code: i for i, code in enumerate(all_nodes)}
    
    print(f" 데이터 준비 완료: 노드 {len(all_nodes)}개, 엣지 {len(df_edges)}개")

    # ES 연결
    es = Elasticsearch("http://localhost:9200")
    
    # 피처 차원 설정 (감성, 가중감성, 변동성 + 랜덤노이즈)
    feature_dim = 16 
    
    # 0이 아닌 아주 작은 랜덤 값으로 초기화 (Feature Collapse 방지)
    x_features = np.random.randn(len(all_nodes), feature_dim) * 0.01 
    
    print(" ES에서 피처 조회 및 매핑 중...")
    found_count = 0
    
    # [수정] 정확한 인덱스 이름 'stock_features_v1' 사용
    target_index = "stock_features_v1"

    for i, code in enumerate(all_nodes):
        try:
            # 해당 종목의 최신 데이터 1건 조회
            resp = es.search(
                index=target_index, 
                body={
                    "query": {"term": {"stock_code": code}}, 
                    "size": 1, 
                    "sort": [{"timestamp": "desc"}]
                }
            )
            
            if resp['hits']['hits']:
                hit = resp['hits']['hits'][0]['_source']
                
                # [수정] 감성 점수, Decay, 변동성 3가지 피처를 모두 활용
                # 데이터가 있으면 앞쪽 3개 차원에 덮어쓰기
                score = hit.get('sentiment_score', 0.0)
                decay = hit.get('sentiment_decay', 0.0)
                volatility = hit.get('sentiment_volatility', 0.0)
                
                x_features[i, 0] = score
                x_features[i, 1] = decay
                x_features[i, 2] = volatility
                
                found_count += 1
        except Exception as e:
            pass
            
    print(f" 피처 매핑 완료: {found_count}/{len(all_nodes)} 종목 데이터 존재")
    
    # 데이터가 너무 없으면 경고
    if found_count < 10:
        print(" [경고] 매핑된 데이터가 너무 적습니다! run_batch.py를 실행했는지 확인하세요.")

    # 텐서 변환
    x = torch.tensor(x_features, dtype=torch.float)
    
    src_idx = [node_to_idx[s] for s in df_edges['source']]
    dst_idx = [node_to_idx[t] for t in df_edges['target']]
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    # 모델 학습
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ---------------------------------------------------------
    # [핵심 수정] out_channels를 64에서 16으로 변경!
    # ---------------------------------------------------------
    model = GCNEncoder(in_channels=feature_dim, out_channels=16).to(device) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    x = x.to(device)
    edge_index = edge_index.to(device)

    print(" GAE 학습 시작 (Target Embedding Dim: 16)...")
    model.train()
    
    for epoch in range(201): 
        optimizer.zero_grad()
        
        # 1. 임베딩 생성
        z = model(x, edge_index)
        
        # 2. Positive Sample (실제 연결된 엣지)
        pos_src = z[edge_index[0]]
        pos_dst = z[edge_index[1]]
        # 내적값(유사도) 계산
        pos_scores = (pos_src * pos_dst).sum(dim=1)
        
        # 3. Negative Sample (랜덤으로 섞어서 가짜 엣지 생성)
        # edge_index[1]을 랜덤하게 섞음 -> 무작위 연결 생성
        neg_dst_indices = edge_index[1][torch.randperm(edge_index.size(1))]
        neg_dst = z[neg_dst_indices]
        neg_scores = (pos_src * neg_dst).sum(dim=1)
        
        # 4. Loss 계산 (BPR Loss 또는 Binary Cross Entropy 변형)
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
        
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} (Pos: {pos_loss:.4f}, Neg: {neg_loss:.4f})")

    # 결과 저장
    model.eval()
    with torch.no_grad():
        final_embeddings = model(x, edge_index)
        # 코사인 유사도 계산을 위해 정규화
        final_embeddings = F.normalize(final_embeddings, p=2, dim=1)
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join(save_dir, "gcn_embeddings.npy"), final_embeddings.cpu().numpy())
    
    df_nodes = pd.DataFrame(all_nodes, columns=['stock_code'])
    df_nodes.to_csv(os.path.join(save_dir, "gcn_node_list.csv"), index=False)
    
    print(" 임베딩 및 노드 리스트 저장 완료! (16차원)")

if __name__ == "__main__":
    train_gcn()