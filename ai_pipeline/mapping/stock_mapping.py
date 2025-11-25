import FinanceDataReader as fdr
import re

class StockMapper:
    def __init__(self):
        print("📊 주식 종목 리스트 로딩 중... (KOSPI/KOSDAQ)")
        self.stock_dict = self._load_stock_data()
        print(f"✅ 로딩 완료: 총 {len(self.stock_dict)}개 종목 감시 중")

    def _load_stock_data(self):
        """
        KOSPI, KOSDAQ 전 종목 데이터를 가져와서
        { '삼성전자': '005930', 'SK하이닉스': '000660', ... } 형태의 딕셔너리로 만듦
        """
        try:
            # 1. KOSPI & KOSDAQ 종목 리스트 다운로드
            df_kospi = fdr.StockListing('KOSPI')
            df_kosdaq = fdr.StockListing('KOSDAQ')

            # 2. 데이터 합치기
            # Name(종목명), Code(종목코드) 컬럼만 사용
            df_stocks = df_kospi[['Code', 'Name']]._append(df_kosdaq[['Code', 'Name']])
            
            # 3. 딕셔너리로 변환 (Key: 종목명, Value: 종목코드)
            stock_map = dict(zip(df_stocks['Name'], df_stocks['Code']))

            # -------------------------------------------------------
            # [필수] 검색 품질을 높이기 위한 예외 처리 (Stopwords)
            # 너무 흔한 단어라 기사에 자주 나오는데, 실제로는 종목명인 경우 제외
            # 예: "대상", "가정", "신원", "국동" 등 문맥상 헷갈리는 것들
            # -------------------------------------------------------
            ignore_list = {"대상", "가정", "신원", "서울", "국동", "보락", "대동"} 
            
            for word in ignore_list:
                if word in stock_map:
                    del stock_map[word]

            return stock_map

        except Exception as e:
            print(f"❌ 주식 데이터 로딩 실패: {e}")
            return {}

    def extract_related_stocks(self, text):
        """
        뉴스 텍스트를 받아서 등장하는 종목 코드 리스트를 반환
        입력: "삼성전자와 SK하이닉스가 상승했다."
        출력: ['005930', '000660']
        """
        found_stocks = set()
        
        # 텍스트가 없으면 빈 리스트 반환
        if not text:
            return []

        # [알고리즘] 모든 종목을 하나씩 검색하면 느리므로
        # 일단 단순하게 루프를 돌리되, 나중에 속도 이슈 생기면 Aho-Corasick 등으로 고도화 가능
        # 현재 수준(2500개)에서는 단순 루프도 충분히 빠름.
        
        for name, code in self.stock_dict.items():
            # 1. 일단 텍스트 안에 종목명이 있는지 1차 확인 (속도 최적화)
            if name in text:
                
                # 2. [정밀 검사] 진짜 단독 단어인지 확인 (Regex)
                # (?<![가-힣a-zA-Z0-9]) : 내 앞에 한글,영어,숫자가 없어야 한다!
                # 예: "실효성" -> 앞에 '실'(한글)이 있으므로 탈락
                # 예: " 효성" -> 앞에 공백이므로 통과
                # 예: "(효성)" -> 앞에 특수문자이므로 통과
                
                pattern = r'(?<![가-힣a-zA-Z0-9])' + re.escape(name)
                
                # 정규식 조건에 맞는 게 하나라도 있으면 인정
                if re.search(pattern, text):
                    found_stocks.add(code)
                
        return list(found_stocks)
# 전역에서 한 번만 로딩해서 쓰도록 인스턴스 생성 (Singleton처럼 활용)
mapper = StockMapper()

def get_stock_mentions(text):
    """외부에서 호출하기 편하게 만든 래퍼 함수"""
    return mapper.extract_related_stocks(text)