import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
import torch_geometric.transforms as T
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch

# 프로젝트 루트 경로 설정 (기존과 동일하게 유지)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# ==========================================
# 1. 데이터 로드 및 그래프 생성 클래스
# ==========================================
class GraphDataLoader:
    def __init__(self):
        # 데이터 경로를 프로젝트 구조에 맞게 설정
        self.edge_path = os.path.join(project_root, "data", "graph_edges.csv")
        self.es = Elasticsearch("http://localhost:9200")

    def load_data(self):
        print(" [GraphDataLoader] 데이터 로딩 시작...")
        
        # 1. 엣지 파일 로드
        if not os.path.exists(self.edge_path):
            raise FileNotFoundError(f"엣지 파일이 없습니다: {self.edge_path} -> 먼저 build_edges.py를 실행하세요.")
            
        df_edges = pd.read_csv(self.edge_path, dtype=str)
        
        # 2. 노드 매핑 (String Code -> Integer Index)
        all_nodes = sorted(list(set(df_edges['source']).union(set(df_edges['target']))))
        node_to_idx = {code: i for i, code in enumerate(all_nodes)}
        
        print(f"  - 노드 개수: {len(all_nodes)}개")
        print(f"  - 엣지 개수: {len(df_edges)}개")

        # 3. 노드 Feature (X) 생성 - ES에서 가져오기
        feature_dim = 16
        # 초기값: 아주 작은 랜덤 노이즈 (Feature Collapse 방지)
        x_features = np.random.randn(len(all_nodes), feature_dim) * 0.01
        
        print("  - ES에서 노드 피처(감성 점수 등) 조회 중...")
        found_count = 0
        target_index = "stock_features_v1"

        for i, code in enumerate(all_nodes):
            try:
                # 최신 데이터 1건 조회
                resp = self.es.search(
                    index=target_index,
                    body={
                        "query": {"term": {"stock_code": code}},
                        "size": 1,
                        "sort": [{"timestamp": "desc"}]
                    }
                )
                if resp['hits']['hits']:
                    hit = resp['hits']['hits'][0]['_source']
                    # ES 필드를 피처 벡터 앞부분에 할당
                    x_features[i, 0] = hit.get('sentiment_score', 0.0)
                    x_features[i, 1] = hit.get('sentiment_decay', 0.0)
                    x_features[i, 2] = hit.get('sentiment_volatility', 0.0)
                    found_count += 1
            except Exception:
                pass 
        
        print(f"  - 피처 매핑 완료: {found_count}/{len(all_nodes)} 종목")

        # 4. PyTorch 텐서 변환
        x = torch.tensor(x_features, dtype=torch.float)
        
        src_idx = [node_to_idx[s] for s in df_edges['source']]
        dst_idx = [node_to_idx[t] for t in df_edges['target']]
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        
        # Data 객체 생성
        data = Data(x=x, edge_index=edge_index)
        
        return data, all_nodes

# ==========================================
# 2. GCN Encoder 모델 정의
# ==========================================
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # Layer 1: 입력 -> 32
        self.conv1 = GCNConv(in_channels, 32)
        # Layer 2: 32 -> 출력(16)
        self.conv2 = GCNConv(32, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)

# ==========================================
# 3. 학습 및 실행 메인 함수 (기존 호환성 위해 이름 유지)
# ==========================================
def train_gcn():
    # 설정
    LEARNING_RATE = 0.01
    EPOCHS = 200
    HIDDEN_CHANNELS = 16  # 최종 임베딩 차원
    
    # 1. 데이터 준비
    try:
        loader = GraphDataLoader()
        data, all_nodes = loader.load_data()
    except Exception as e:
        print(f" [Error] 데이터 로드 실패: {e}")
        return

    # 학습/검증용 엣지 분할 (Link Prediction Task)
    transform = T.RandomLinkSplit(
        num_val=0.1, 
        num_test=0.1, 
        is_undirected=True, 
        add_negative_train_samples=False,
        split_labels=True  # <--- 이 옵션이 꼭 있어야 합니다!
    )
    train_data, val_data, test_data = transform(data)

    # 2. 모델 초기화 (GAE)
    in_channels = data.num_features
    model = GAE(GCNEncoder(in_channels, HIDDEN_CHANNELS))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    x = train_data.x.to(device)
    edge_index = train_data.edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. 학습 루프
    print("\n--- GAE 학습 시작 (Run GCN) ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        z = model.encode(x, edge_index)
        
        # Reconstruction Loss
        loss = model.recon_loss(z, train_data.edge_index)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                z = model.encode(x, edge_index)
                auc, ap = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')

    print("--- 학습 완료 ---")

    # 4. 결과 저장
    model.eval()
    with torch.no_grad():
        final_z = model.encode(data.x.to(device), data.edge_index.to(device))
        final_z = F.normalize(final_z, p=2, dim=1)
        final_z_np = final_z.cpu().numpy()
    
    # 저장 경로: project_root/data/
    save_dir = os.path.join(project_root, "data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    emb_path = os.path.join(save_dir, "gcn_embeddings.npy")
    np.save(emb_path, final_z_np)
    
    node_path = os.path.join(save_dir, "gcn_node_list.csv")
    df_nodes = pd.DataFrame(all_nodes, columns=['stock_code'])
    df_nodes.to_csv(node_path, index=False)
    
    print(f"\n[저장 완료]")
    print(f" - 임베딩: {emb_path}")
    print(f" - 노드맵: {node_path}")

if __name__ == "__main__":
    train_gcn()