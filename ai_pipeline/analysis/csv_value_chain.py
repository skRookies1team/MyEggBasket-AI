import pandas as pd
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

class CSVValueChainAnalyzer:
    """
    기업 보고서 기반 CSV 데이터를 활용한 확실한 밸류체인 분석기
    """
    
    def __init__(self, csv_file_name="value_chain_data.csv"):
        print("🔗 [CSV] 밸류체인 데이터 로딩 중...")
        
        # 1. CSV 파일 경로 설정 (프로젝트 루트/data 폴더 안에 있다고 가정)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.abspath(os.path.join(current_dir, "../../data", csv_file_name))
        
        self.df = None
        self._load_data()

    def _load_data(self):
        """CSV 파일 로드 및 전처리"""
        if not os.path.exists(self.data_path):
            print(f"❌ 밸류체인 CSV 파일이 없습니다: {self.data_path}")
            print("   -> 'data' 폴더를 만들고 CSV 파일을 넣어주세요.")
            return

        try:
            # 문자열(String)로 읽어야 앞자리 '0'이 안 사라짐 (005930)
            self.df = pd.read_csv(self.data_path, dtype=str)
            
            # 컬럼명 공백 제거 및 소문자화 (실수 방지)
            self.df.columns = self.df.columns.str.strip().str.lower()
            
            # 필수 컬럼 확인 (사용자가 가진 CSV에 맞춰서 수정 가능)
            # 예: 'source' -> 'code', 'target' -> 'related_code'
            required_cols = ['code', 'related_code']
            
            if not all(col in self.df.columns for col in required_cols):
                print(f"⚠️ CSV 컬럼 이름이 다릅니다. 현재 컬럼: {self.df.columns.tolist()}")
                print("   -> 코드 내 'required_cols' 부분을 수정하거나 CSV 헤더를 'code', 'related_code'로 바꿔주세요.")
                return

            print(f"✅ 밸류체인 데이터 로드 완료: {len(self.df):,}개 관계 정보")

        except Exception as e:
            print(f"❌ CSV 로딩 실패: {e}")

    def find_similar_stocks(self, target_code, top_n=5):
        """
        특정 종목(target_code)과 연관된 종목들을 CSV에서 찾아서 반환
        (기존 GCN 함수와 이름/형식을 동일하게 맞춰서 교체가 쉽습니다)
        """
        if self.df is None:
            return []

        # 1. 기준 종목으로 필터링
        # (code 컬럼이 target_code인 행 찾기)
        related_rows = self.df[self.df['code'] == target_code].copy()
        
        if related_rows.empty:
            # 반대 방향(내가 related_code 쪽에 있는 경우)도 찾을지 결정
            # 쌍방향 관계라면 아래 주석 해제
            # related_rows = self.df[self.df['related_code'] == target_code].copy()
            # if related_rows.empty:
            return []

        # 2. 점수(score)가 있다면 정렬, 없으면 그냥 가져옴
        if 'score' in related_rows.columns:
            # 점수 컬럼을 실수형으로 변환 후 정렬
            related_rows['score'] = related_rows['score'].astype(float)
            related_rows = related_rows.sort_values('score', ascending=False)
        
        # 3. 결과 포맷팅 (기존 코드와 호환되게)
        results = []
        for _, row in related_rows.head(top_n).iterrows():
            
            # 관계 설명(relation) 컬럼이 있으면 가져오기
            reason = row['relation'] if 'relation' in row else "연관 종목"
            score = float(row['score']) if 'score' in row else 1.0
            
            results.append({
                "code": row['related_code'],
                "name": row.get('related_name', ''), # 이름 있으면 넣고
                "score": score,
                "reason": reason # 이게 핵심! (왜 추천했는지 근거)
            })
            
        return results

    def get_relation_reason(self, code_a, code_b):
        """
        두 종목이 왜 연결됐는지 이유 반환 (설명력 강화)
        """
        if self.df is None: return "데이터 없음"
        
        # A -> B 검색
        row = self.df[(self.df['code'] == code_a) & (self.df['related_code'] == code_b)]
        if not row.empty:
            return row.iloc[0].get('relation', '연관 종목')
            
        # B -> A 검색 (역방향)
        row = self.df[(self.df['code'] == code_b) & (self.df['related_code'] == code_a)]
        if not row.empty:
            return row.iloc[0].get('relation', '연관 종목')
            
        return None

'''
# ==========================================
# 테스트 실행
# ==========================================
if __name__ == "__main__":
    # 1. 테스트용 더미 파일 만들기 (파일 없으면 자동 생성)
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    csv_path = os.path.join(data_dir, "value_chain_data.csv")
    
    # 테스트 파일이 없을 때만 생성
    if not os.path.exists(csv_path):
        print("🧪 테스트용 CSV 파일을 생성합니다...")
        dummy_data = """code,related_code,related_name,relation,score
005930,000660,SK하이닉스,반도체 경쟁사,0.95
005930,005935,삼성전자우,우선주,0.99
005380,000270,기아,현대차그룹 계열사,0.98
"""
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(dummy_data)

    # 2. 분석기 실행
    analyzer = CSVValueChainAnalyzer()
    
    target = "005930" # 삼성전자
    print(f"\n🔍 [{target}] 밸류체인 분석 결과 (CSV 기반)")
    
    recs = analyzer.find_similar_stocks(target)
    for r in recs:
        print(f"   👉 {r['code']} ({r['name']}) - 이유: {r['reason']} (강도: {r['score']})")
```

---

### 3단계: 통합하기 (어떻게 바꾸면 되나요?)

이제 기존에 `PortfolioAnalyzer`나 `pipeline_main.py`에서 **GCN 쓰던 부분을 이 CSV 분석기로 갈아끼우면** 됩니다.

#### 예시: `ai_pipeline/portfolio/analyzer.py` 수정

```python
# 기존 import
# from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer
# ...

# [수정] GCN 분석기 대신 -> CSV 분석기 import
from ai_pipeline.analysis.csv_value_chain import CSVValueChainAnalyzer  # <--- NEW!

class PortfolioAnalyzer:
    def __init__(self, csv_path):
        # ... (기존 코드) ...
        
        # [수정] 밸류체인 분석기 교체
        # self.vc_analyzer = ValueChainAnalyzer()  <-- (X) 기존 GCN
        self.vc_analyzer = CSVValueChainAnalyzer() # <-- (O) 신규 CSV

    # ... (중략) ...

    def analyze_value_chain(self, target_code, top_n=5):
        """
        CSV 기반으로 연관 종목 추천
        """
        # 기존 로직 싹 지우고, 그냥 CSV 분석기 함수 한 줄 호출하면 끝!
        return self.vc_analyzer.find_similar_stocks(target_code, top_n)

        '''