import sys
import os
import pandas as pd
import re

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

class ValueChainAnalyzer:
    """
    [CSV 기반] 기업 밸류체인 데이터를 활용한 연관 종목 분석기
    (GCN 대신 정확한 엑셀 데이터를 사용하여 100% 신뢰할 수 있는 관계를 추천합니다)
    """
    def __init__(self):
        print("🔗 [ValueChain] 밸류체인 데이터 로딩 중...")
        
        # 1. 파일 경로 설정 (스마트 탐색)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        
        # 우선순위: data 폴더 -> 루트 폴더
        self.csv_path = os.path.join(project_root, "data", "value_chain_result.csv")
        
        if not os.path.exists(self.csv_path):
             self.csv_path = os.path.join(project_root, "value_chain_result.csv")

        self.df = None
        self.stock_to_rows = {} # { '005930': [0, 5, 10] } (종목코드가 포함된 행 인덱스들)
        
        # 2. 데이터 로드
        self._load_data()

    def _load_data(self):
        """CSV 파일 로드 및 전처리"""
        if not os.path.exists(self.csv_path):
            print(f"❌ 밸류체인 데이터 파일이 없습니다: {self.csv_path}")
            print("   -> 먼저 map_stock_codes.py를 실행해서 파일을 생성해주세요.")
            return

        try:
            # 인코딩 자동 감지 (utf-8-sig 또는 cp949)
            try:
                self.df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            except:
                self.df = pd.read_csv(self.csv_path, encoding='cp949')
            
            # 결측치 처리 (빈 칸은 빈 문자열로)
            self.df = self.df.fillna("")
            
            # [전처리] 종목별 인덱싱 (속도 최적화)
            # 어떤 종목이 엑셀의 몇 번째 줄(어떤 섹터)에 있는지 미리 저장해둠
            self.stock_to_rows = {}
            
            # '기업_코드포함' 컬럼 사용 (예: "삼성전자 (005930), SK하이닉스 (000660)")
            target_col = '기업_코드포함'
            if target_col not in self.df.columns:
                print(f"❌ '{target_col}' 컬럼이 없습니다. map_stock_codes.py를 먼저 실행하세요.")
                return

            for idx, row in self.df.iterrows():
                companies_str = str(row.get(target_col, ''))
                
                # 정규식으로 (숫자6자리) 코드만 추출
                matches = re.findall(r'\(([\d]{6})\)', companies_str)
                
                for code in matches:
                    if code not in self.stock_to_rows:
                        self.stock_to_rows[code] = []
                    # 이 종목이 등장한 행 번호(idx)를 저장
                    self.stock_to_rows[code].append(idx)
            
            print(f"✅ 밸류체인 데이터 로드 완료")
            print(f"   - 등록된 밸류체인 그룹(행): {len(self.df)}개")
            print(f"   - 등록된 종목 수: {len(self.stock_to_rows)}개")

        except Exception as e:
            print(f"❌ 데이터 로딩 실패: {e}")

    def _get_category_name(self, row):
        """행 데이터에서 섹터/분류 이름을 조합하여 반환 (추천 사유가 됨)"""
        parts = []
        # 엑셀 컬럼명에 맞춰서 계층 구조 생성
        for col in ['섹터', '소분류1', '소분류2', '소분류3']:
            if col in row and row[col]:
                parts.append(str(row[col]))
        return " > ".join(parts) if parts else "연관 테마"

    def find_similar_stocks(self, target_code, top_n=5):
        """
        특정 종목(target_code)과 같은 밸류체인(행)에 있는 종목들을 추천
        """
        if self.df is None or target_code not in self.stock_to_rows:
            # print(f"⚠️ '{target_code}' 종목은 밸류체인 데이터에 없습니다.")
            return []

        # 1. 해당 종목이 포함된 모든 행(Row) 인덱스 가져오기
        row_indices = self.stock_to_rows[target_code]
        
        recommendations = []
        seen_codes = {target_code} # 이미 추가한 종목 + 자기 자신은 제외

        # 2. 각 행을 순회하며 같은 줄에 있는 친구들 찾기
        for idx in row_indices:
            row = self.df.iloc[idx]
            category = self._get_category_name(row)
            
            # 같은 줄에 있는 기업들 파싱
            companies_str = str(row.get('기업_코드포함', ''))
            
            # 정규식: "이름 (코드)" 패턴 찾기
            # 예: "LG전자 (066570)" -> name="LG전자", code="066570"
            # 쉼표, 공백 등으로 복잡하게 섞여 있어도 괄호 패턴으로 정확히 찾음
            items = re.findall(r'([^,(\s]+)\s*\(([\d]{6})\)', companies_str)
            
            for name, code in items:
                name = name.strip()
                
                if code not in seen_codes:
                    recommendations.append({
                        "code": code,
                        "name": name,
                        "score": 0.95, # 팩트 기반이므로 신뢰도 고정값 (높음)
                        "reason": category # 예: "반도체 > 장비 > 전공정"
                    })
                    seen_codes.add(code)

        # 3. 결과 반환 (최대 N개)
        return recommendations[:top_n]

    def explain_relation(self, stock_code_A, stock_code_B):
        """
        두 종목이 어떤 관계로 연결되었는지 설명 (CSV 기반)
        """
        if self.df is None: return

        # 두 종목이 모두 포함된 행 찾기
        rows_A = set(self.stock_to_rows.get(stock_code_A, []))
        rows_B = set(self.stock_to_rows.get(stock_code_B, []))
        
        common_rows = rows_A.intersection(rows_B)
        
        print(f"\n🕵️‍♂️ [관계 분석] {stock_code_A} ↔ {stock_code_B}")
        
        if common_rows:
            print("   👉 두 종목은 다음 밸류체인에 함께 속해 있습니다:")
            for idx in common_rows:
                row = self.df.iloc[idx]
                category = self._get_category_name(row)
                print(f"      - [{category}]")
        else:
            print("   👉 직접적인 밸류체인 공유 관계가 없습니다.")

# ==========================================
# 테스트 실행 코드 (이 파일을 직접 실행할 때만 작동)
# ==========================================
if __name__ == "__main__":
    analyzer = ValueChainAnalyzer()
    
    # 테스트할 종목 코드 (엑셀에 있는 코드여야 함)
    # 예: LG전자 (066570)
    target = "066570" 
    
    print(f"\n🔍 [{target}] 밸류체인 추천 결과")
    recs = analyzer.find_similar_stocks(target, top_n=10)
    
    if recs:
        for r in recs:
            print(f"   👉 {r['name']} ({r['code']})")
            print(f"      이유: {r['reason']}")
    else:
        print("   (해당 종목은 밸류체인 파일에 없습니다)")

    # 관계 설명 테스트 (결과가 있다면)
    if len(recs) > 0:
        friend = recs[0]['code']
        analyzer.explain_relation(target, friend)


'''

### 🧐 주요 변경점 및 특징

1.  **GCN 의존성 제거:** 더 이상 `gcn_node_embeddings.pt` 파일을 찾지 않습니다. 오직 `value_chain_result.csv`만 있으면 됩니다.
2.  **정규식 파싱 강화:** `r'([^,(\s]+)\s*\(([\d]{6})\)'` 패턴을 사용하여, **"LG전자 (066570)"** 같은 형식에서 이름과 코드를 정확하게 분리해냅니다.
3.  **이유(Reason) 자동 생성:** 엑셀의 `섹터 > 소분류1 > 소분류2` 컬럼을 합쳐서 **"어떤 이유로 추천되었는지"** 명확하게 보여줍니다. (예: "가전 > 영상가전")

### 🚀 사용 방법

1.  이 코드로 파일을 덮어쓴 후,
2.  아래 명령어로 바로 테스트해 보세요. (LG전자가 엑셀에 있다면 관련 기업들이 뜰 겁니다.)

bash
python ai_pipeline/gcn_model/value_chain.py

'''