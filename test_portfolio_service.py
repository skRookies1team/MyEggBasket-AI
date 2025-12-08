import os
import sys
import pandas as pd

# 모듈 경로 설정
from ai_pipeline.portfolio.analyzer import PortfolioAnalyzer
from ai_pipeline.portfolio.rebalancer import PortfolioRebalancer

def run_simulation():
    print("="*60)
    print(" 주식 포트폴리오 관리 서비스 시뮬레이션")
    print("="*60)

    # 1. 데이터 준비 (오늘자 주가 데이터)
    # [주의] 실제 파일명으로 변경 필요
    csv_file = "20251120.csv"
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), csv_file))
    
    if not os.path.exists(csv_path):
        print(f" 데이터 파일이 없습니다: {csv_file}")
        return

    # 2. 분석기 초기화
    analyzer = PortfolioAnalyzer(csv_path)
    rebalancer = PortfolioRebalancer()

    # ---------------------------------------------------------
    # [시나리오] 사용자 정보 입력
    # 보유 종목: 삼성전자(005930), 카카오(035720)
    # ---------------------------------------------------------
    my_portfolio = {
        '005930': 5000000,  # 삼성전자 500만원
        '035720': 2000000   # 카카오 200만원
    }
    print(f"\n 사용자 포트폴리오: {list(my_portfolio.keys())}")
    print("-" * 60)

    # ---------------------------------------------------------
    # 기능 1: 보유 종목 진단 (밸류체인 & AI 점수)
    # ---------------------------------------------------------
    print("\n [보유 종목 진단]")
    
    # AI 점수 확인
    my_scores = analyzer.get_ai_scores(filter_codes=my_portfolio.keys())
    if my_scores is not None:
        print("    내 종목 AI 점수:")
        print(my_scores.to_string(index=False))
    
    # 밸류체인 추천
    print("\n    보유 종목 연관 추천 (Value Chain):")
    for code in my_portfolio.keys():
        recs = analyzer.analyze_value_chain(code, top_n=2)
        if recs:
            print(f"       [{code}] 관련: {[r['name'] for r in recs]}")
            print(f"         (이유: {recs[0]['reason']})")

    # ---------------------------------------------------------
    # 기능 2: 투자 성향별 신규 추천 (AI Filtering)
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print(" [신규 종목 추천] 투자 성향을 선택하세요.")
    
    # 시나리오 A: 공격형 투자자
    print("\n    [A] 공격형(Aggressive) 추천 리스트:")
    agg_recs = analyzer.recommend_by_style('aggressive', top_n=3, exclude_codes=my_portfolio.keys())
    for r in agg_recs:
        print(f"      Code: {r['code']} | 점수: {r['ai_score']} | 이유: {r['reason']}")

    # 시나리오 B: 줍줍형 투자자
    print("\n    [B] 줍줍형(Reversal) 추천 리스트:")
    rev_recs = analyzer.recommend_by_style('reversal', top_n=3, exclude_codes=my_portfolio.keys())
    for r in rev_recs:
        print(f"      Code: {r['code']} | 점수: {r['ai_score']} | 이유: {r['reason']}")

    # ---------------------------------------------------------
    # 기능 3: AI 리밸런싱 (공격형 추천 종목 중 1위 종목을 편입한다고 가정)
    # ---------------------------------------------------------
    if not agg_recs:
        print("\n 추천 종목이 없어서 리밸런싱을 건너뜁니다.")
        return

    top_pick = agg_recs[0]['code']
    top_score = agg_recs[0]['ai_score']
    
    print("\n" + "-" * 60)
    print(f" [AI 리밸런싱] 추천 1위 '{top_pick}'을 포트폴리오에 추가합니다.")
    
    # 리밸런싱을 위해 데이터프레임 구성
    # 내 종목 + 추천 종목
    target_codes = list(my_portfolio.keys()) + [top_pick]
    
    # 전체 AI 점수 가져오기
    all_scores = analyzer.get_ai_scores(filter_codes=target_codes)
    
    # 리밸런싱 계산기 돌리기
    print("    비중 계산 중...")
    result_df = rebalancer.run_ai_rebalancing(my_portfolio, all_scores)
    
    if result_df is not None:
        print("\n    [최종 주문서]")
        print(result_df.to_string(index=False))
        print("\n 시뮬레이션 완료!")

if __name__ == "__main__":
    run_simulation()