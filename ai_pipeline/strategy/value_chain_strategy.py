import pandas as pd
import os
import sys

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 이전에 만든 ValueChainAnalyzer 활용
try:
    from ai_pipeline.gcn_model.value_chain import ValueChainAnalyzer
except ImportError:
    print(" [Error] ValueChainAnalyzer를 찾을 수 없습니다. ai_pipeline/gcn_model/value_chain.py 확인 필요")
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
        예측 결과(prediction_df)를 바탕으로 밸류체인 기반 동반 상승 추천을 생성합니다.
        
        Logic:
        1. AI Score가 80점 이상인 '대장주/메인종목'을 찾는다.
        2. 그 종목의 밸류체인(공급사, 고객사 등)을 조회한다.
        3. 연관된 종목의 AI Score도 60점 이상(매수권)인지 확인한다.
        4. 둘 다 좋다면 '동반 매수 추천' 리포트를 생성한다.
        """
        if self.vc_analyzer is None or prediction_df is None or prediction_df.empty:
            return []

        # 1. 빠른 조회를 위해 예측 결과를 딕셔너리로 변환 {code: score}
        score_map = prediction_df.set_index('code')['ai_score'].to_dict()
        
        # 2. 메인 트리거 종목 선정 (80점 이상인 강력 매수 종목)
        strong_buy_df = prediction_df[prediction_df['ai_score'] >= 80]
        
        final_recommendations = []
        checked_pairs = set() # 중복 추천 방지

        print(f" [Strategy] 강력 매수 시그널 종목 {len(strong_buy_df)}개 분석 중...")

        for _, row in strong_buy_df.iterrows():
            main_code = row['code']
            main_score = row['ai_score']
            # 이름 가져오기 (Analyzer 혹은 임시)
            main_name = self.vc_analyzer.get_stock_name(main_code) 

            # 3. 밸류체인 연관 종목 검색
            # (top_n=10 정도로 넉넉하게 가져옴)
            related_stocks = self.vc_analyzer.find_similar_stocks(main_code, top_n=10)
            
            for rel in related_stocks:
                rel_code = rel['code']
                rel_name = rel['name']
                rel_reason = rel['reason'] # 예: "반도체 장비 > 세정"
                
                # 중복 방지 (A->B 추천했으면 B->A는 생략하거나 포함 가능)
                pair_key = tuple(sorted([main_code, rel_code]))
                if pair_key in checked_pairs:
                    continue
                
                # 4. 연관 종목의 점수 확인
                rel_score = score_map.get(rel_code, 0)
                
                # [전략 조건]
                # 메인 종목은 이미 80점 이상.
                # 연관 종목은 60점 이상이면(매수 의견) 동반 추천.
                if rel_score >= 60:
                    
                    # 근거 텍스트 생성 (사용자가 원했던 그 포맷!)
                    rationale = (
                        f"주도주 '{main_name}'(예측확률 {main_score}%)의 강력한 상승 시그널 포착. "
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

        # 점수 합산(평균)이 높은 순으로 정렬
        final_recommendations.sort(key=lambda x: x['Main_Score'] + x['Target_Score'], reverse=True)
        
        return pd.DataFrame(final_recommendations)

if __name__ == "__main__":
    # 테스트용 더미 데이터
    strategy = ValueChainStrategy()
    
    # 가상의 예측 결과
    dummy_pred = pd.DataFrame({
        'code': ['005930', '000660', '035420', '012345'], # 삼성전자, 하이닉스 등
        'ai_score': [85.5, 72.0, 40.0, 65.0]
    })
    
    # 012345가 삼성전자의 밸류체인에 있다고 가정하고 테스트됨 (실제 데이터 있으면 작동)
    res = strategy.analyze_predictions(dummy_pred)
    if not res.empty:
        print(res['Rationale'].iloc[0])