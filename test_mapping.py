import sys
import os

# 경로 설정
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ai_pipeline.mapping.stock_mapping import get_stock_mentions

def test_mapping():
    print("\n🔍 뉴스-종목 매핑 테스트 시작\n")

    # 예제 1: 명확한 종목
    news1 = "삼성전자가 오늘 8만전자를 회복했고, SK하이닉스도 동반 상승했다."
    stocks1 = get_stock_mentions(news1)
    print(f"📰 뉴스: {news1}")
    print(f"👉 추출된 종목 코드: {stocks1}") # 예상: 삼성전자, 하이닉스 코드
    print("-" * 50)

    # 예제 2: 종목이 없는 경우
    news2 = "오늘 코스피 지수는 전반적으로 하락 마감했다. 환율은 상승했다."
    stocks2 = get_stock_mentions(news2)
    print(f"📰 뉴스: {news2}")
    print(f"👉 추출된 종목 코드: {stocks2}") # 예상: 빈 리스트 []
    print("-" * 50)

    # 예제 3: 여러 종목 (2차전지)
    news3 = "에코프로와 에코프로비엠, 그리고 POSCO홀딩스가 강세를 보였다."
    stocks3 = get_stock_mentions(news3)
    print(f"📰 뉴스: {news3}")
    print(f"👉 추출된 종목 코드: {stocks3}")
    print("-" * 50)

if __name__ == "__main__":
    test_mapping()