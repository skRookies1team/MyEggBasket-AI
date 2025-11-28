## 포트폴리오 외 추천 

import sys
import os
import torch
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

class ValueChainAnalyzer:
    def __init__(self):
        print("🔗 [GCN] 밸류체인 분석기 초기화 중...")
        
        # 1. 파일 경로 설정
        # 현재 위치: ai_pipeline/gcn_model/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
        
        # GCN 임베딩 (루트에 저장됨)
        self.emb_path = os.path.join(root_dir, "gcn_node_embeddings.pt")
        # 노드 매핑 (graph_build 폴더에 저장됨)
        self.map_path = os.path.join(current_dir, "../graph_build/node_mapping.json")
        # 엣지 파일 (설명용)
        self.edge_path = os.path.join(current_dir, "../graph_build/graph_edges.csv")

        self.embeddings = None
        self.node_map = None
        self.idx_to_code = None
        self.code_to_idx = None

        # 2. 데이터 로드
        self._load_data()

    def _load_data(self):
        """GCN 결과 파일 로드"""
        if not os.path.exists(self.emb_path) or not os.path.exists(self.map_path):
            print("❌ GCN 데이터 파일이 없습니다. 먼저 스케줄러를 실행해 그래프를 구축하세요.")
            return

        try:
            # 임베딩 벡터 로드 (PyTorch -> Numpy)
            # weights_only=False 옵션은 PyTorch 버전 이슈 대응
            self.embeddings = torch.load(self.emb_path, weights_only=False).numpy()
            
            # 매핑 정보 로드
            with open(self.map_path, "r", encoding='utf-8') as f:
                self.node_map = json.load(f) # "0": "news_id...", "1": "005930"
            
            # 인덱스 -> 코드 역매핑 생성 (검색용)
            # key가 문자열로 되어있으므로 int로 변환
            self.idx_to_code = {int(k): v for k, v in self.node_map.items()}
            
            # 코드 -> 인덱스 매핑 생성 (입력용)
            self.code_to_idx = {v: int(k) for k, v in self.node_map.items()}
            
            print("✅ GCN 데이터 로드 완료")
            print(f"   - 분석 대상 노드: {len(self.embeddings)}개")

        except Exception as e:
            print(f"❌ 데이터 로딩 실패: {e}")

    def find_similar_stocks(self, target_code, top_n=5):
        """
        특정 종목(target_code)과 가장 관계가 깊은(유사한) 종목을 찾습니다.
        """
        if self.embeddings is None:
            return []

        # 1. 입력 종목이 그래프에 있는지 확인
        if target_code not in self.code_to_idx:
            # print(f"⚠️ '{target_code}' 종목은 최근 뉴스 데이터에 없어서 분석 불가합니다.")
            return []

        # 2. 타겟 벡터 추출
        target_idx = self.code_to_idx[target_code]
        target_vector = self.embeddings[target_idx].reshape(1, -1) # (1, 16)

        # 3. 코사인 유사도 계산 (타겟 vs 전체)
        # 결과는 -1 ~ 1 사이의 값 (1에 가까울수록 강력한 관계)
        similarities = cosine_similarity(target_vector, self.embeddings)[0]

        # 4. 정렬 (높은 순서대로)
        # argsort는 낮은 순이라 [::-1]로 뒤집음
        sorted_indices = similarities.argsort()[::-1]

        results = []
        for idx in sorted_indices:
            # 자기 자신은 제외
            if idx == target_idx:
                continue

            code_or_id = self.idx_to_code[idx]

            # 5. 필터링: 뉴스 노드는 제외하고 '종목 코드(6자리 숫자)'만 남김
            if code_or_id.isdigit() and len(code_or_id) == 6:
                similarity_score = similarities[idx]
                
                # 유사도가 너무 낮은 건 굳이 추천 안 함 (예: 0.1 미만)
                if similarity_score < 0.1:
                    break

                results.append({
                    "code": code_or_id,
                    "score": round(float(similarity_score), 4) # 소수점 4자리
                })

                if len(results) >= top_n:
                    break
        
        return results

    def explain_relation(self, stock_code_A, stock_code_B):
        """
        두 종목(A, B)이 공유하는 뉴스(연결고리)가 있는지 확인합니다.
        그래프 엣지 파일(graph_edges.csv)을 직접 뒤집니다.
        """
        
        # 엣지 파일 존재 확인
        if not os.path.exists(self.edge_path):
            print("❌ 엣지 파일(graph_edges.csv)이 없습니다.")
            return

        # 매핑 정보 확인
        if stock_code_A not in self.code_to_idx or stock_code_B not in self.code_to_idx:
            print("⚠️ 종목 코드가 매핑 정보에 없습니다.")
            return

        # [수정 완료] self.stock_mapping -> self.code_to_idx 로 변경
        idx_A = self.code_to_idx[stock_code_A]
        idx_B = self.code_to_idx[stock_code_B]

        try:
            df_edges = pd.read_csv(self.edge_path)
            
            # A랑 연결된 뉴스 ID 찾기 (Target이 종목인 경우 Source는 뉴스)
            news_connected_to_A = set(df_edges[df_edges['target'] == idx_A]['source'].tolist())
            # B랑 연결된 뉴스 ID 찾기
            news_connected_to_B = set(df_edges[df_edges['target'] == idx_B]['source'].tolist())

            # 교집합 (공통 뉴스)
            common_news = news_connected_to_A.intersection(news_connected_to_B)
            
            print(f"\n🕵️‍♂️ [관계 분석] {stock_code_A} ↔ {stock_code_B}")
            print(f"   - A 관련 뉴스: {len(news_connected_to_A)}개")
            print(f"   - B 관련 뉴스: {len(news_connected_to_B)}개")
            print(f"   - 🔗 공통 뉴스(연결고리): {len(common_news)}개")
            
            if len(common_news) > 0:
                print("   👉 결론: 같은 뉴스에 등장하여 직접적으로 연결됨 (유사도 높음)")
            else:
                print("   👉 결론: 직접 연결은 없으나 2-Hop(공통 이웃) 등을 통해 간접 연결됨")
                
        except Exception as e:
            print(f"❌ 관계 분석 중 에러: {e}")

# ==========================================
# 테스트 실행 코드 (이 파일을 직접 실행할 때만 작동)
# ==========================================
if __name__ == "__main__":
    analyzer = ValueChainAnalyzer()
    
    # 1. 삼성전자와 가장 친한 종목 찾기
    target = "005930" # 삼성전자
    print(f"\n🔍 [{target}] 유사 종목 분석")
    similar_stocks = analyzer.find_similar_stocks(target, top_n=3)
    
    for item in similar_stocks:
        print(f"   - {item['code']} (유사도: {item['score']})")

    # 2. [검증] 왜 점수가 높은지 까보기 (1위 종목이랑 비교)
    if similar_stocks:
        top_friend = similar_stocks[0]['code']
        analyzer.explain_relation(target, top_friend)