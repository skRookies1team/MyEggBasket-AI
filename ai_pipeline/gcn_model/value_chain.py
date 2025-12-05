import sys
import os
import torch
import json
import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

class ValueChainAnalyzer:
    """
    [CSV 기반] 기업 밸류체인 데이터를 활용한 연관 종목 분석기
    """
    def __init__(self):
        print(" [ValueChain] CSV 밸류체인 데이터 로딩 중...")
        
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
        
        self.df = None
        self.stock_to_rows = {} 
        self.code_to_name = {} # [수정 3] 코드->이름 매핑용 딕셔너리 추가
        
        # 데이터 로드
        self._load_data()

    def _load_data(self):
        """CSV 파일 로드 및 전처리"""
        if not os.path.exists(self.csv_path):
            print(f" 밸류체인 데이터 파일이 없습니다: {self.csv_path}")
            print("   -> 먼저 map_stock_codes.py를 실행해서 파일을 생성해주세요.")
            return

        try:
            try:
                self.df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            except:
                self.df = pd.read_csv(self.csv_path, encoding='cp949')
            
            self.df = self.df.fillna("")
            
            # 종목별 인덱싱
            self.stock_to_rows = {}
            target_col = '기업_코드포함'
            
            if target_col not in self.df.columns:
                print(f" '{target_col}' 컬럼이 없습니다.")
                return
            
            for idx, row in self.df.iterrows():
                companies_str = str(row.get(target_col, ''))
                
                # 정규식: "삼성전자 (005930)" 형태 추출
                # items = [('삼성전자', '005930'), ('하이닉스', '000660')...]
                items = re.findall(r'([^,(\s]+)\s*\(([\d]{6})\)', companies_str)
                
                for name, code in items:
                    name = name.strip()
                    # 행 인덱스 저장
                    if code not in self.stock_to_rows:
                        self.stock_to_rows[code] = []
                    self.stock_to_rows[code].append(idx)
                    
                    # 이름 매핑 저장
                    self.code_to_name[code] = name

            print(f" 밸류체인 데이터 로드 완료 ({len(self.stock_to_rows)}개 종목)")

        except Exception as e:
            print(f" 데이터 로딩 실패: {e}")

    def get_stock_name(self, code):
        """  종목 코드를 입력하면 이름을 반환 """
        return self.code_to_name.get(code, code) # 없으면 코드 그대로 반환
            

    def _get_category_name(self, row):
        """추천 사유(테마명) 생성"""
        parts = []
        for col in ['섹터', '소분류1', '소분류2', '소분류3']:
            if col in row and row[col]:
                parts.append(str(row[col]))
        return " > ".join(parts) if parts else "연관 테마"
    

    def find_similar_stocks(self, target_code, top_n=5):
        """특정 종목과 같은 밸류체인에 있는 종목 추천"""
        if self.df is None or target_code not in self.stock_to_rows:
            return []

        row_indices = self.stock_to_rows[target_code]
        recommendations = []
        seen_codes = {target_code} 

        for idx in row_indices:
            row = self.df.iloc[idx]
            category = self._get_category_name(row)
            companies_str = str(row.get('기업_코드포함', ''))
            
            # "이름 (코드)" 추출
            items = re.findall(r'([^,(\s]+)\s*\(([\d]{6})\)', companies_str)
            
            for name, code in items:
                name = name.strip()
                if code not in seen_codes:
                    recommendations.append({
                        "code": code,
                        "name": name,          # <-- [중요] 이름 필드 포함!
                        "score": 0.95,
                        "reason": category     # <-- [중요] 이유 필드 포함!
                    })
                    seen_codes.add(code)

        return recommendations[:top_n]
    

    '''

    def explain_relation(self, stock_code_A, stock_code_B):
        """두 종목 관계 설명"""
        if self.df is None: return

        rows_A = set(self.stock_to_rows.get(stock_code_A, []))
        rows_B = set(self.stock_to_rows.get(stock_code_B, []))
        common_rows = rows_A.intersection(rows_B)
        
        print(f"\n [관계 분석] {stock_code_A} ↔ {stock_code_B}")
        if common_rows:
            print("    같은 밸류체인 그룹에 속해 있습니다:")
            for idx in common_rows:
                print(f"      - [{self._get_category_name(self.df.iloc[idx])}]")
        else:
            print("    직접적인 밸류체인 공유 관계가 없습니다.")

'''

# 테스트 실행
if __name__ == "__main__":
    analyzer = ValueChainAnalyzer()
    print(analyzer.find_similar_stocks("005930"))
    