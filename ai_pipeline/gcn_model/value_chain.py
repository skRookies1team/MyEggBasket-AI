import sys
import os
import re
import pandas as pd

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

class ValueChainAnalyzer:
    """
    [CSV 기반] 기업 밸류체인 데이터를 활용한 연관 종목 분석기
    (최적화: 미리 관계 맵을 메모리에 로드하여 검색 속도 향상)
    """
    def __init__(self):
        print(" [ValueChain] CSV 밸류체인 데이터 로딩 및 인덱싱 중...")
        
        possible_paths = [
            os.path.join(current_dir, "value_chain_result.csv"),
            os.path.join(current_dir, "../../data/value_chain_result.csv"),
            os.path.join(current_dir, "../../value_chain_result.csv")
        ]
        
        self.csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.csv_path = path
                break
        
        # { '005930': {'name': '삼성전자', 'related': [ {'code':..., 'name':..., 'reason':...} ] } }
        self.chain_map = {} 
        self.code_to_name = {}

        self._load_and_preprocess()

    def _load_and_preprocess(self):
        """CSV 로드 후, 검색하기 좋게 미리 딕셔너리로 변환"""
        if not self.csv_path:
            print(" [Warning] 밸류체인 파일이 없습니다.")
            return

        try:
            try:
                df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            except:
                df = pd.read_csv(self.csv_path, encoding='cp949')
            
            df = df.fillna("")
            target_col = '기업_코드포함'

            if target_col not in df.columns:
                print(f" '{target_col}' 컬럼이 없습니다.")
                return

            # 데이터프레임을 순회하며 관계 맵 생성
            for idx, row in df.iterrows():
                category = self._get_category_name(row)
                companies_str = str(row.get(target_col, ''))
                
                # [(이름, 코드), (이름, 코드)...] 추출
                items = re.findall(r'([^,(\s]+)\s*\(([\d]{6})\)', companies_str)
                
                # 리스트 내 모든 종목 간의 관계 형성 (Clique)
                for my_name, my_code in items:
                    my_name = my_name.strip()
                    self.code_to_name[my_code] = my_name # 이름 등록
                    
                    if my_code not in self.chain_map:
                        self.chain_map[my_code] = []

                    # 나(my_code)를 제외한 나머지 종목들을 내 연관 리스트에 추가
                    for other_name, other_code in items:
                        other_name = other_name.strip()
                        if my_code == other_code:
                            continue
                        
                        # 중복 방지 로직 (이미 같은 사유로 등록된 경우 제외)
                        existing = [x for x in self.chain_map[my_code] if x['code'] == other_code]
                        if not existing:
                            self.chain_map[my_code].append({
                                "code": other_code,
                                "name": other_name,
                                "reason": category,
                                "score": 0.95
                            })

            print(f" 밸류체인 인덱싱 완료 ({len(self.chain_map)}개 종목)")

        except Exception as e:
            print(f" 데이터 로딩 실패: {e}")

    def _get_category_name(self, row):
        parts = []
        for col in ['섹터', '소분류1', '소분류2', '소분류3']:
            if col in row and str(row[col]).strip():
                parts.append(str(row[col]))
        return " > ".join(parts) if parts else "연관 테마"

    def get_stock_name(self, code):
        return self.code_to_name.get(code, code)

    def find_similar_stocks(self, target_code, top_n=5):
        """
        O(1) 속도로 연관 종목 반환
        """
        if target_code not in self.chain_map:
            return []
        
        recommendations = self.chain_map[target_code]
        return recommendations[:top_n]

if __name__ == "__main__":
    analyzer = ValueChainAnalyzer()
    print(analyzer.find_similar_stocks("005930")) # 삼성전자 테스트