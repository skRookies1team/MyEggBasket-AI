import sys
import os
import torch
import json
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

        self.embeddings = None
        self.node_map = None
        self.idx_to_code = None

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
            print(f"⚠️ '{target_code}' 종목은 최근 뉴스 데이터에 없어서 분석 불가합니다.")
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

# ==========================================
# 테스트 실행 코드 (이 파일을 직접 실행할 때만 작동)
# ==========================================
if __name__ == "__main__":
    analyzer = ValueChainAnalyzer()
    
    # 테스트할 종목 코드 (예: 삼성전자)
    # 실제 그래프에 있는 종목이어야 결과가 나옵니다.
    test_stock = "005930" 
    
    print(f"\n🔍 [{test_stock}] 종목의 GCN 밸류체인 분석 결과")
    related_stocks = analyzer.find_similar_stocks(test_stock, top_n=5)
    
    if related_stocks:
        for i, item in enumerate(related_stocks):
            print(f"   {i+1}위: {item['code']} (유사도: {item['score']})")
    else:
        print("   -> 연관된 종목을 찾을 수 없습니다. (뉴스 데이터 부족 등)")