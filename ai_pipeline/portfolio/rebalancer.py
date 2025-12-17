import pandas as pd
import numpy as np


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

        # ------------------------------------------------------------------
        # [핵심 수정 1] 데이터 무결성 확보 (공백 제거 및 6자리 통일)
        # ------------------------------------------------------------------
        # 보유 종목 코드 정제
        cleaned_holdings = {str(k).strip().zfill(6): v for k, v in current_holdings.items()}
        current_total = sum(cleaned_holdings.values())

        # 예측 데이터 코드 정제
        ai_scores_df = ai_scores_df.copy()
        ai_scores_df['code'] = ai_scores_df['code'].astype(str).str.strip().str.zfill(6)

        if total_budget is None:
            total_budget = current_total

        # ------------------------------------------------------------------
        # 1. 데이터 병합 (보유 종목 누락 방지)
        # ------------------------------------------------------------------
        # 보유 중이지만 AI 점수가 없는 종목 -> 기본 점수(40점) 부여하여 급격한 매도 방지
        merged_df = ai_scores_df.copy()
        held_codes = set(cleaned_holdings.keys())

        # 예측 결과에 없는 보유 종목 찾기
        prediction_codes = set(merged_df['code'].values)
        missing_holdings = held_codes - prediction_codes

        if missing_holdings:
            print(f" [Info] 점수 미확인 보유종목 기본값 처리: {missing_holdings}")
            missing_data = []
            for code in missing_holdings:
                missing_data.append({'code': code, 'ai_score': 40.0, 'opinion': '데이터없음'})
            merged_df = pd.concat([merged_df, pd.DataFrame(missing_data)], ignore_index=True)

        # ------------------------------------------------------------------
        # [핵심 수정 2] 이중 필터링 로직 (Dual Threshold)
        # - 신규 종목: 60점 이상이어야 매수 (깐깐함)
        # - 보유 종목: 40점 이상이면 유지 (관대함 - 비중 축소로 유도)
        # ------------------------------------------------------------------

        # 조건 1: 신규 진입 (보유X AND 점수 >= 60)
        cond_new_buy = (~merged_df['code'].isin(held_codes)) & (merged_df['ai_score'] >= 60)

        # 조건 2: 보유 유지 (보유O AND 점수 >= 40) -> 삼성전자(45점)가 여기서 살아남음!
        cond_keep = (merged_df['code'].isin(held_codes)) & (merged_df['ai_score'] >= 40)

        # 최종 후보군 선정
        buy_candidates = merged_df[cond_new_buy | cond_keep].copy()

        if buy_candidates.empty:
            print(" [Warning] 유효한 투자 대상이 없습니다. 전량 현금화 또는 관망을 추천합니다.")
            return pd.DataFrame()

        # ------------------------------------------------------------------
        # 2. 비중 산출 (Score^2 가중치 방식)
        # ------------------------------------------------------------------
        # 점수 제곱을 통해 고득점 종목에 비중을 몰아줌 (Softmax 효과)
        # 45점(삼성전자) vs 95점(주도주) -> 2025 vs 9025 -> 약 4.5배 비중 차이 발생
        buy_candidates['weight_score'] = np.power(buy_candidates['ai_score'], 2)
        total_weight_score = buy_candidates['weight_score'].sum()

        buy_candidates['target_ratio'] = buy_candidates['weight_score'] / total_weight_score

        # ------------------------------------------------------------------
        # 3. 최종 주문 생성
        # ------------------------------------------------------------------
        rebalancing_plan = []

        # (1) 매수/보유 대상 처리
        for _, row in buy_candidates.iterrows():
            code = row['code']
            target_ratio = row['target_ratio']
            target_amt = int(total_budget * target_ratio)
            current_amt = int(cleaned_holdings.get(code, 0))
            diff = target_amt - current_amt

            # 액션 결정
            if diff > 0:
                action = '매수'
            elif diff < 0:
                action = '비중축소'  # 0점이 아니므로 '전량매도'가 아닌 '축소'가 됨
            else:
                action = '유지'

            rebalancing_plan.append({
                'code': code,
                'ai_score': row['ai_score'],
                'current_amt': current_amt,
                'target_ratio': round(target_ratio, 4),
                'target_amt': target_amt,
                'diff': int(diff),
                'action': action
            })

        # (2) 전량 매도 대상 처리 (점수가 너무 낮아(40점 미만) 탈락한 보유 종목)
        surviving_codes = set(buy_candidates['code'].values)
        for code, amt in cleaned_holdings.items():
            if code not in surviving_codes:
                rebalancing_plan.append({
                    'code': code,
                    'ai_score': 0.0,  # 기준 미달
                    'current_amt': int(amt),
                    'target_ratio': 0.0,
                    'target_amt': 0,
                    'diff': -int(amt),
                    'action': '전량매도'
                })

        df_plan = pd.DataFrame(rebalancing_plan)

        # 금액 기준 내림차순 정렬
        if not df_plan.empty:
            df_plan = df_plan.sort_values(by='target_amt', ascending=False)

        return df_plan[['code', 'ai_score', 'action', 'diff', 'target_amt', 'current_amt', 'target_ratio']]


# ==========================================
# 테스트 실행 코드 (버그 수정됨)
# ==========================================
if __name__ == "__main__":
    rebalancer = PortfolioRebalancer()

    # [상황 설정]
    my_portfolio = {
        '005930': 15000000,  # 삼성전자 (보유중)
        '000660': 5000000,  # SK하이닉스 (보유중)
    }

    # [가상 데이터]
    # 삼성전자: 45점 (기존 로직이면 전량매도였으나, 이제는 비중축소로 살아남아야 함)
    # 하이닉스: 95점 (강력 매수)
    # 카카오: 70점 (신규 진입)
    fake_ai_scores = pd.DataFrame({
        'code': ['005930', '000660', '035720'],
        'ai_score': [45.0, 95.0, 70.0],
        'opinion': ['중립', '강력매수', '매수']
    })

    # 실행
    df = rebalancer.run_ai_rebalancing(my_portfolio, fake_ai_scores)

    # 출력 확인
    print("\n [테스트 결과 확인]")
    print(df.to_string(index=False))

    print("-" * 60)
    # 005930이 '전량매도'가 아니라 '비중축소'로 뜨는지 확인
    for _, row in df.iterrows():
        print(f" [{row['code']}] {row['ai_score']}점 -> {row['action']} (목표금액: {row['target_amt']:,}원)")