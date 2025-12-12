import sys
import os

# 분석에 사용할 키워드 DB 정의
KEYWORD_DB = {
    "경제/금융": [
        "금리 인하", "피벗", "인플레이션", "가계 부채", "환율", "강달러", "스태그플레이션",
        "반도체", "HBM", "AI 반도체", "2차전지", "배터리", "공매도", "밸류업", "기업 가치 제고",
        "비트코인", "가상자산", "ETF", "IPO", "전세 사기", "재건축", "재개발", "PF 위기", "미분양"
    ],
    "정치/사회": [
        "총선", "정계 개편", "특검법", "거부권", "당정 갈등", "지지율",
        "의대 증원", "의료 파업", "의료 대란", "저출산", "인구 소멸", "고령화",
        "묻지마 범죄", "교권 침해", "마약", "연금 개혁", "노동 개혁", "늘봄 학교", "기후 동행 카드"
    ],
    "IT/과학": [
        "인공지능", "생성형 AI", "ChatGPT", "Gemini", "온디바이스 AI", "AI 규제", "딥페이크",
        "데이터 센터", "자율주행", "UAM", "로봇", "협동 로봇", "양자 컴퓨터",
        "망 사용료", "플랫폼 규제", "5G", "6G", "우주항공청", "누리호", "스페이스X"
    ],
    "국제/외교": [
        "우크라이나 전쟁", "이스라엘", "하마스", "미중 갈등", "대만 해협", "양안 관계",
        "미국 대선", "트럼프", "해리스", "엔저", "오염수", "북한 도발", "정찰 위성",
        "공급망", "탈중국", "희토류", "IRA", "인플레이션 감축법"
    ],
    "문화/라이프": [
        "K-컬처", "OTT", "넷플릭스", "천만 영화", "K-팝", "하이브",
        "기후 위기", "탄소 중립", "폭염", "이상 기후", "엘니뇨",
        "숏폼", "팝업 스토어", "탕후루", "제로 슈거"
    ]
}


class TrendKeywordExtractor:
    @staticmethod
    def extract_top_keyword(text):
        """
        본문에서 키워드 등장 횟수를 세어 가장 많이 언급된 (카테고리, 키워드)를 반환합니다.
        """
        best_category = None
        best_keyword = None
        max_count = 0

        for category, keywords in KEYWORD_DB.items():
            for kw in keywords:
                # 텍스트 내 등장 횟수 카운트
                count = text.count(kw)
                if count > max_count:
                    max_count = count
                    best_keyword = kw
                    best_category = category

        # 하나도 발견되지 않았으면 None 반환
        if max_count == 0:
            return None, None

        return best_category, best_keyword

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from ai_pipeline.news_source.finance_news_list import fetch_finance_news_list
from ai_pipeline.news_source.news_article_crawler import extract_real_article_url, fetch_article_text
from ai_pipeline.news_etl.es_uploader import save_news_to_es, exists_in_es
from ai_pipeline.nlp.news_analyzer import NewsAnalyzer
from ai_pipeline.gcn_model.value_chain import ValueChainAnalyzer

def run_finance_news_etl():
    print(" ETL 시작됨 (문장단위 정밀분석 + 밸류체인)")

    news_urls = fetch_finance_news_list(max_pages=3)
    
    if not news_urls:
        print(" 수집된 뉴스가 없습니다.")
        return
    
    # 분석기들 초기화
    news_analyzer = NewsAnalyzer()
    vc_analyzer = ValueChainAnalyzer()
    
    saved_count = 0
    skipped_dup = 0

    for idx, finance_url in enumerate(news_urls):
        real_url = extract_real_article_url(finance_url)
        
        # 1. 중복 체크
        if exists_in_es(real_url):
            skipped_dup += 1
            continue

        print(f"[{idx+1}]   분석 중: {real_url}")

        # 2. 제목, 본문, 날짜 수집 (수정된 크롤러는 3개를 반환함)
        if not crawled_data:
            print("     본문 없음/짧음 -> Pass")
            continue

        # [수정] 반환값 3개 언패킹
        title, article_text, p_date = crawled_data
        
        if len(article_text) < 50:
             print("     본문 너무 짧음 -> Pass")
             continue

        # ---------------------------------------------------------
        #  [Core 1] 문장 단위 정밀 감성 분석
        # ---------------------------------------------------------
        # results: { '005930': {'sentiment_score': 0.9, ...} }
        # details: [ {'sentence': '...', 'ticker': '...', 'sentiment': 0.9}, ... ]
        analysis_results, sentence_details = news_analyzer.analyze_article(article_text)

        if not analysis_results:
            print("    [Pass] 종목 언급 없음 (일반 뉴스)")
            continue

        related_stocks = list(analysis_results.keys())

        # ---------------------------------------------------------
        #  [Core 2] 밸류체인 연관 종목 추출
        # ---------------------------------------------------------
        value_chain_info = []
        for stock_code in related_stocks:
            # 각 종목별로 연관된 친구들을 찾음 (CSV 기반)
            recs = vc_analyzer.find_similar_stocks(stock_code, top_n=3)
            if recs:
                for r in recs:
                    value_chain_info.append({
                        "source_code": stock_code,
                        "related_code": r['code'],
                        "related_name": r['name'],
                        "reason": r['reason']
                    })

        top_category, top_keyword = TrendKeywordExtractor.extract_top_keyword(article_text)
        if top_keyword:
            print(f"    [Trend] {top_category} | {top_keyword}")

        # ---------------------------------------------------------
        #  [Core 3] 최종 저장
        # ---------------------------------------------------------
        print(f"    발견: {related_stocks} | 문장수: {len(sentence_details)} | VC연관: {len(value_chain_info)}개")
        
        # 변수명 매칭 (url -> real_url, text -> article_text, published_date 추가)
        save_news_to_es(
            url=real_url,
            title=title,
            text=article_text,
            published_date=p_date,
            related_stocks=related_stocks,
            analysis_results=analysis_results,
            sentence_details=sentence_details,
            value_chain_info=value_chain_info,
            category=top_category,
            keyword=top_keyword
        )
        saved_count += 1

    print("\n" + "="*40)
    print(f" ETL 완료")
    print(f"   - 새로 저장됨: {saved_count}건")
    print(f"   - 중복 건너뜀: {skipped_dup}건")
    print("="*40)

if __name__ == "__main__":
    run_finance_news_etl()