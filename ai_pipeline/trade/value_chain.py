import pandas as pd
import os
import sys

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

class ValueChainLoader:
    def __init__(self, edge_path=None):
        if edge_path is None:
            # 기본 경로: data/graph_edges.csv
            edge_path = os.path.join(project_root, "data", "graph_edges.csv")
        
        self.relations = {}
        self._load_edges(edge_path)

    def _load_edges(self, path):
        if not os.path.exists(path):
            print(f" [ValueChain] 엣지 파일 없음: {path}")
            return

        try:
            df = pd.read_csv(path, dtype=str)
            count = 0
            # source -> target 저장 (단방향 or 양방향 정책 결정)
            # 여기서는 '연관된 종목'을 모두 찾기 위해 양방향으로 매핑
            for _, row in df.iterrows():
                src = row['source'].strip().zfill(6)
                tgt = row['target'].strip().zfill(6)
                
                if src not in self.relations: self.relations[src] = set()
                if tgt not in self.relations[tgt]: self.relations[tgt] = set()

                self.relations[src].add(tgt)
                self.relations[tgt].add(src) 
                count += 1
            
            print(f" [ValueChain] {len(self.relations)}개 종목의 관계망 로드 완료 (Edge: {count}개)")
        except Exception as e:
            print(f" [ValueChain] 로드 중 에러: {e}")

    def get_related_stocks(self, stock_code):
        """특정 종목과 연결된 종목 코드 리스트 반환"""
        stock_code = str(stock_code).zfill(6)
        if stock_code in self.relations:
            return list(self.relations[stock_code])
        return []