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
        """
        print(f"\n [Portfolio] AI 점수 기반 리밸런싱 시작 (모드: {self.risk_aversion})")

        if ai_scores_df is None or ai_scores_df.empty:
            print(" [Error] AI 예측 점수가 없습니다.")
            return pd.DataFrame()

        # ------------------------------------------------------------------
        # 1. 데이터 정제
        # ------------------------------------------------------------------
        cleaned_holdings = {str(k).strip().zfill(6): v for k, v in current_holdings.items()}
        current_total = sum(cleaned_holdings.values())

        ai_scores_df = ai_scores_df.copy()
        ai_scores_df['code'] = ai_scores_df['code'].astype(str).str.strip().str.zfill(6)

        if total_budget is None:
            total_budget = current_total

        # ------------------------------------------------------------------
        # 2. 보유 종목 데이터 보정 (누락 방지)
        # ------------------------------------------------------------------
        merged_df = ai_scores_df.copy()
        held_codes = set(cleaned_holdings.keys())
        prediction_codes = set(merged_df['code'].values)

        missing_holdings = held_codes - prediction_codes
        if missing_holdings:
            print(f" [Info] 점수 미확인 보유종목 기본값 처리: {missing_holdings}")
            missing_data = []
            for code in missing_holdings:
                missing_data.append({'code': code, 'ai_score': 40.0, 'opinion': '데이터없음'})
            merged_df = pd.concat([merged_df, pd.DataFrame(missing_data)], ignore_index=True)

        # ------------------------------------------------------------------
        # 3. 이중 필터링 (신규 진입장벽은 높게, 보유 유지는 낮게)
        # ------------------------------------------------------------------
        cond_new_buy = (~merged_df['code'].isin(held_codes)) & (merged_df['ai_score'] >= 60)
        cond_keep = (merged_df['code'].isin(held_codes)) & (merged_df['ai_score'] >= 40)

        buy_candidates = merged_df[cond_new_buy | cond_keep].copy()

        if buy_candidates.empty:
            print(" [Warning] 유효한 투자 대상이 없습니다.")
            return pd.DataFrame()

        # ------------------------------------------------------------------
        # 4. 비중 산출 (Score^2 가중치)
        # ------------------------------------------------------------------
        buy_candidates['weight_score'] = np.power(buy_candidates['ai_score'], 2)
        total_weight_score = buy_candidates['weight_score'].sum()
        buy_candidates['target_ratio'] = buy_candidates['weight_score'] / total_weight_score

        # ------------------------------------------------------------------
        # 5. 최종 주문 생성 (유지 구간 적용)
        # ------------------------------------------------------------------
        rebalancing_plan = []

        # [핵심] 유지 구간 설정 (전체 자산의 2% 미만 변동은 무시)
        THRESHOLD_RATIO = 0.02
        threshold_amt = total_budget * THRESHOLD_RATIO

        for _, row in buy_candidates.iterrows():
            code = row['code']
            target_ratio = row['target_ratio']
            target_amt = int(total_budget * target_ratio)
            current_amt = int(cleaned_holdings.get(code, 0))
            diff = target_amt - current_amt

            # [핵심 수정] 변동폭이 임계값보다 작으면 '유지'
            if abs(diff) < threshold_amt:
                action = '유지'
                # 유지는 굳이 사고팔지 않으므로 목표금액을 현재금액으로 맞춤 (선택사항)
                # target_amt = current_amt
                # diff = 0
            elif diff > 0:
                action = '매수'
            else:
                action = '비중축소'

            rebalancing_plan.append({
                'code': code,
                'ai_score': row['ai_score'],
                'current_amt': current_amt,
                'target_ratio': round(target_ratio, 4),
                'target_amt': target_amt,
                'diff': int(diff),
                'action': action
            })

        # 탈락한 보유 종목 처리 (전량 매도)
        surviving_codes = set(buy_candidates['code'].values)
        for code, amt in cleaned_holdings.items():
            if code not in surviving_codes:
                rebalancing_plan.append({
                    'code': code,
                    'ai_score': 0.0,
                    'current_amt': int(amt),
                    'target_ratio': 0.0,
                    'target_amt': 0,
                    'diff': -int(amt),
                    'action': '전량매도'
                })

        df_plan = pd.DataFrame(rebalancing_plan)

        if not df_plan.empty:
            df_plan = df_plan.sort_values(by='target_amt', ascending=False)

        return df_plan[['code', 'ai_score', 'action', 'diff', 'target_amt', 'current_amt', 'target_ratio']]


if __name__ == "__main__":
    pass
