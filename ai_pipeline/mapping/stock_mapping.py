import FinanceDataReader as fdr
import re

class StockMapper:
    def __init__(self):
        print("📊 주식 종목 리스트 로딩 중... (KOSPI/KOSDAQ)")
        self.stock_dict = self._load_stock_data()
        
        # [핵심 1] 긴 이름부터 먼저 검색하기 위해 길이 역순으로 정렬된 키 리스트 생성
        # 예: ['SK하이닉스', 'SK스퀘어', ..., 'SK'] 순서
        self.sorted_stock_names = sorted(self.stock_dict.keys(), key=len, reverse=True)
        
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

        temp_text = text

        for name in self.sorted_stock_names:
            
            # 일단 in 연산자로 빠르게 1차 필터링 (속도 최적화)
            if name not in temp_text:
                continue

            pattern = r'(?<![가-힣a-zA-Z0-9])' + re.escape(name) + r'(?![가-힣a-zA-Z0-9])'

            if re.search(pattern, temp_text):
                code = self.stock_dict[name]
                found_stocks.add(code)
                
                # [핵심 4] 찾은 단어를 지워버림 (마스킹)
                # 예: "SK하이닉스 상승" -> "SK하이닉스" 발견 -> "@@@@@ 상승" 으로 변경
                # 나중에 "SK"가 검색할 때 "SK" 글자가 없으므로 중복 안 잡힘!
                # 길이를 유지하기 위해 같은 길이의 특수문자로 치환
                temp_text = re.sub(pattern, '#' * len(name), temp_text)
                
        return list(found_stocks)
        
# 전역에서 한 번만 로딩해서 쓰도록 인스턴스 생성 (Singleton처럼 활용)
mapper = StockMapper()

def get_stock_mentions(text):
    """외부에서 호출하기 편하게 만든 래퍼 함수"""
    return mapper.extract_related_stocks(text)