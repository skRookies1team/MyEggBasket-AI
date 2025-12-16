import pandas as pd
import os
import sys

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 이전에 만든 ValueChainAnalyzer 활용
try:
    from ai_pipeline.gcn_model.value_chain import ValueChainAnalyzer
except ImportError:
    print(" [Error] ValueChainAnalyzer를 찾을 수 없습니다.")
    ValueChainAnalyzer = None


class ValueChainStrategy:
    def __init__(self):
        print(" [Strategy] 밸류체인 전략 분석기 초기화...")
        if ValueChainAnalyzer:
            self.vc_analyzer = ValueChainAnalyzer()
        else:
            self.vc_analyzer = None

    def analyze_predictions(self, prediction_df):
        """
        Logic:
        1. AI Score 80점 이상인 대장주 식별
        2. 대장주의 밸류체인(공급/고객사) 조회
        3. 연관 종목도 60점 이상이면 '동반 매수' 추천
        """
        if self.vc_analyzer is None or prediction_df is None or prediction_df.empty:
            return pd.DataFrame()

        # 조회 최적화
        score_map = prediction_df.set_index('code')['ai_score'].to_dict()

        # 1. 메인 트리거 종목 (강력 매수 후보)
        strong_buy_df = prediction_df[prediction_df['ai_score'] >= 80]

        final_recommendations = []
        checked_pairs = set()

        print(f" [Strategy] 강력 시그널 종목 {len(strong_buy_df)}개 기반 밸류체인 분석 중...")

        for _, row in strong_buy_df.iterrows():
            main_code = row['code']
            main_score = row['ai_score']
            main_name = self.vc_analyzer.get_stock_name(main_code)

            # 2. 밸류체인 연관 종목 검색 (Top 10)
            related_stocks = self.vc_analyzer.find_similar_stocks(main_code, top_n=10)

            for rel in related_stocks:
                rel_code = rel['code']
                rel_name = rel['name']
                rel_reason = rel['reason']  # 예: "반도체 장비 > 세정"

                # 중복 방지
                pair_key = tuple(sorted([main_code, rel_code]))
                if pair_key in checked_pairs: continue

                # 3. 연관 종목 점수 확인
                rel_score = score_map.get(rel_code, 0)

                # [전략] 대장주(80+) & 연관주(60+) 동반 상승 시그널
                if rel_score >= 60:
                    rationale = (
                        f"주도주 '{main_name}'({main_score}점)의 상승세에 힘입어, "
                        f"밸류체인으로 연결된 '{rel_name}'({rel_reason})도 "
                        f"AI 점수 {rel_score}점으로 동반 상승이 유력함."
                    )

                    rec_item = {
                        "Target_Code": rel_code,
                        "Target_Stock": rel_name,
                        "Target_Score": rel_score,
                        "Main_Stock": main_name,
                        "Main_Score": main_score,
                        "Relation": rel_reason,
                        "Rationale": rationale
                    }
                    final_recommendations.append(rec_item)
                    checked_pairs.add(pair_key)

        # 점수 합산 순 정렬
        final_recommendations.sort(key=lambda x: x['Main_Score'] + x['Target_Score'], reverse=True)

        return pd.DataFrame(final_recommendations)