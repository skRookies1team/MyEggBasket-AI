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
        # print(" [Strategy] 밸류체인 전략 분석기 초기화...")
        if ValueChainAnalyzer:
            self.vc_analyzer = ValueChainAnalyzer()
        else:
            self.vc_analyzer = None

    def analyze_predictions(self, prediction_df):
        """
        Logic:
        1. 메인 종목(점수 높은 것) 선정
        2. 밸류체인 조회 및 '후보군 출력' (사용자 요청 사항)
        3. 연관 종목 점수 확인 후 필터링
        """
        if self.vc_analyzer is None or prediction_df is None or prediction_df.empty:
            return pd.DataFrame()

        # 1. 빠른 조회를 위해 딕셔너리 변환 {code: score}
        score_map = prediction_df.set_index('code')['ai_score'].to_dict()
        
        # 기준 설정 (테스트를 위해 낮게 유지)
        MAIN_THRESHOLD = 50.0  
        TARGET_THRESHOLD = 40.0
        
        strong_buy_df = prediction_df[prediction_df['ai_score'] >= MAIN_THRESHOLD]
        
        final_recommendations = []
        checked_pairs = set()

        print(f"\n [Strategy Analysis] 분석 대상(메인) 종목: {len(strong_buy_df)}개")

        for _, row in strong_buy_df.iterrows():
            main_code = row['code']
            main_score = row['ai_score']
            main_name = self.vc_analyzer.get_stock_name(main_code) 

            # -------------------------------------------------------
            # [Step 1] 밸류체인 연관 종목 조회 및 나열
            # -------------------------------------------------------
            related_stocks = self.vc_analyzer.find_similar_stocks(main_code, top_n=20)
            
            if not related_stocks:
                # 연관 종목이 없는 경우 스킵
                continue

            print(f"\n  주도주 분석: {main_name} ({main_code}) | AI 점수: {main_score}점")
            
            # 연관 종목 이름들만 모아서 출력 (사용자 요청 1)
            rel_names = [r['name'] for r in related_stocks]
            print(f"    ㄴ  발견된 밸류체인 종목 ({len(rel_names)}개): {', '.join(rel_names)}")

            # -------------------------------------------------------
            # [Step 2] 각 연관 종목의 점수 확인 및 필터링
            # -------------------------------------------------------
            pass_count = 0
            for rel in related_stocks:
                rel_code = rel['code']
                rel_name = rel['name']
                rel_reason = rel['reason']
                
                # 점수 조회 (없으면 0점)
                rel_score = score_map.get(rel_code, 0)
                
                # (디버깅용 출력) 각 연관 종목의 점수 상황 보여주기
                # print(f"       - {rel_name}: {rel_score}점", end="")

                pair_key = tuple(sorted([main_code, rel_code]))
                if pair_key in checked_pairs:
                    # print(" (중복 생략)")
                    continue
                
                # 기준 통과 여부 확인
                if rel_score >= TARGET_THRESHOLD:
                    # print(" ->  [Pass]")
                    pass_count += 1
                    
                    rationale = (
                        f"주도주 '{main_name}'(예측확률 {main_score}%)의 상승 시그널 포착. "
                        f"이에 따라 밸류체인상 연관된 '{rel_name}'({rel_reason}) 또한 "
                        f"AI 점수 {rel_score}%로 동반 상승이 유력함."
                    )
                    
                    rec_item = {
                        "Main_Stock": main_name,
                        "Main_Score": main_score,
                        "Target_Stock": rel_name,
                        "Target_Score": rel_score,
                        "Relation": rel_reason,
                        "Rationale": rationale
                    }
                    final_recommendations.append(rec_item)
                    checked_pairs.add(pair_key)
                else:
                    # print(f" ->  [Fail] 기준({TARGET_THRESHOLD}점) 미달")
                    pass

            if pass_count == 0:
                print(f"    ㄴ  동반 상승 추천 실패: 연관 종목들의 점수가 모두 기준({TARGET_THRESHOLD}점) 미만입니다.")

        if not final_recommendations:
            return pd.DataFrame()

        # 점수 합산 순 정렬
        final_recommendations.sort(key=lambda x: x['Main_Score'] + x['Target_Score'], reverse=True)
        
        return pd.DataFrame(final_recommendations)