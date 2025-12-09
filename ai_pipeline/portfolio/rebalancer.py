import pandas as pd
import numpy as np


class PortfolioEngine:
    def __init__(self, max_turnover=0.2):
        self.max_turnover = max_turnover  # 한 번에 바꿀 수 있는 최대 비중 (20%)

    def calculate_weights(self, prediction_df, current_portfolio):
        """
        prediction_df: ['stock_code', 'ai_score'] (score: 0.0 ~ 1.0)
        current_portfolio: {'stock_code': current_weight}
        """
        target_weights = {}

        # 1. Score 기반 기본 비중 산출 (Softmax 또는 비례 배분)
        # 예: 0.6 이상인 종목만 매수 대상
        candidates = prediction_df[prediction_df['ai_score'] > 0.6].copy()

        if candidates.empty:
            return {}  # 매수할 게 없음 (현금 보유)

        # 점수 비례 배분 (Simple Logic)
        total_score = candidates['ai_score'].sum()
        candidates['target_weight'] = candidates['ai_score'] / total_score

        # 2. 리스크 관리 및 제약 조건 적용 (Rebalancing)
        for code, row in candidates.iterrows():
            target_w = row['target_weight']
            current_w = current_portfolio.get(code, 0.0)

            # 급격한 변동 제한 (Turnover Constraint)
            # 목표가 10%인데 현재 0%라면, 이번엔 5%까지만 매수 등
            diff = target_w - current_w
            if abs(diff) > self.max_turnover:
                target_w = current_w + (np.sign(diff) * self.max_turnover)

            target_weights[code] = target_w

        return target_weights


class PortfolioRebalancer:
    """
    AI 점수 기반 포트폴리오 리밸런싱 엔진
    """

    def __init__(self, risk_aversion='neutral'):
        # risk_aversion: 'aggressive', 'neutral', 'conservative'
        self.risk_aversion = risk_aversion

    def run_ai_rebalancing(self, current_holdings, ai_scores_df, total_budget=None):
        """
        current_holdings: {'005930': 5000000, ...} (현재 보유 평가금액)
        ai_scores_df: DataFrame ['code', 'ai_score']
        total_budget: 리밸런싱 후 운용할 총 자산 (None이면 현재 자산 총액 유지)
        """
        print(f"\n [Portfolio] AI 점수 기반 리밸런싱 시작 (모드: {self.risk_aversion})")

        if ai_scores_df is None or ai_scores_df.empty:
            print(" [Error] AI 예측 점수가 없습니다.")
            return pd.DataFrame()

        # 1. 자산 총액 계산
        current_total = sum(current_holdings.values())
        if total_budget is None:
            total_budget = current_total

        # 보유 중이지만 AI 점수가 없는 종목 처리 (기본 점수 부여)
        merged_df = ai_scores_df.copy()
        for code in current_holdings.keys():
            if code not in merged_df['code'].values:
                # 점수 정보가 없으면 중립(50점) 혹은 매도 유도(30점) 처리
                new_row = pd.DataFrame({'code': [code], 'ai_score': [40.0], 'opinion': ['데이터없음']})
                merged_df = pd.concat([merged_df, new_row], ignore_index=True)

        # 2. Score to Weight 변환 (비중 산출 로직)
        # 점수가 임계치(예: 60점) 이상인 종목들만 매수 대상으로 선정
        buy_candidates = merged_df[merged_df['ai_score'] >= 60].copy()

        if buy_candidates.empty:
            print(" [Warning] 매수 추천 종목(60점 이상)이 없습니다. 전량 현금화 또는 관망을 추천합니다.")
            return pd.DataFrame()

        # 점수 제곱 등을 통해 상위 종목에 더 많은 비중 부여 (Softmax 유사 효과)
        buy_candidates['weight_score'] = np.power(buy_candidates['ai_score'], 2)
        total_weight_score = buy_candidates['weight_score'].sum()

        buy_candidates['target_ratio'] = buy_candidates['weight_score'] / total_weight_score

        # 3. 최종 주문 생성
        rebalancing_plan = []

        # (1) 매수/보유 대상 처리
        for _, row in buy_candidates.iterrows():
            code = row['code']
            target_ratio = row['target_ratio']
            target_amt = total_budget * target_ratio
            current_amt = current_holdings.get(code, 0)
            diff = target_amt - current_amt

            rebalancing_plan.append({
                'code': code,
                'ai_score': row['ai_score'],
                'current_amt': int(current_amt),
                'target_ratio': round(target_ratio, 4),
                'target_amt': int(target_amt),
                'diff': int(diff),
                'action': '매수' if diff > 0 else '비중축소'
            })

        # (2) 매도 대상 처리 (점수가 낮아서 buy_candidates에 못 든 보유 종목)
        buy_codes = set(buy_candidates['code'].values)
        for code, amt in current_holdings.items():
            if code not in buy_codes:
                rebalancing_plan.append({
                    'code': code,
                    'ai_score': 0.0,  # 점수 낮음
                    'current_amt': int(amt),
                    'target_ratio': 0.0,
                    'target_amt': 0,
                    'diff': -int(amt),
                    'action': '전량매도'
                })

        df_plan = pd.DataFrame(rebalancing_plan)

        # 보기 좋게 정렬
        df_plan = df_plan.sort_values(by='target_amt', ascending=False)
        return df_plan[['code', 'ai_score', 'action', 'diff', 'target_amt', 'current_amt', 'target_ratio']]



# ==========================================
# 테스트 실행 코드 (사용 예시)
# ==========================================
if __name__ == "__main__":
    rebalancer = PortfolioRebalancer()
    
    # [상황 설정] 내 계좌 상황
    my_portfolio = {
        '005930': 5000000, # 삼성전자
        '000660': 4000000, # SK하이닉스
        '005380': 1000000  # 현대차
    }
    
    # [가상 데이터] AI가 분석한 점수라고 가정
    # 실제로는 analyzer.get_ai_scores() 결과를 넣으면 됩니다.
    fake_ai_scores = pd.DataFrame({
        'code': ['005930', '000660', '005380'],
        'ai_score': [80.0, 40.0, 90.0] 
        # 현대차(90) > 삼성전자(80) > 하이닉스(40)
    })
    
    # 실행
    df = rebalancer.run_ai_rebalancing(my_portfolio, fake_ai_scores)
    
    # 출력
    print(df.to_string(index=False))
    print("-" * 60)
    print(" 해석:")
    for _, row in df.iterrows():
        action = row['매매제안']
        if action == "BUY":
            print(f"   [{row['종목코드']}] 비중이 부족합니다. {abs(row['차액']):,.0f}원 만큼 더 사세요 (매수)")
        elif action == "SELL":
            print(f"   [{row['종목코드']}] 비중이 너무 높습니다. {abs(row['차액']):,.0f}원 만큼 파세요 (매도)")
        else:
            print(f"   [{row['종목코드']}] 목표 비중과 일치합니다. (유지)")
            