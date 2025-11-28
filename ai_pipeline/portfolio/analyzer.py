## 투자성향 알고리즘 추가 예정 ( 안정형/ 공격형/ 불기둥형/.... ) 
### 포트폴리오 외 추천 

import sys
import os
import torch
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer
from ai_pipeline.boosting_model.train import StackingEnsemble

class PortfolioAnalyzer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.engineer = FeatureEngineer(csv_path)
        
        # 모델 로드
        self.model = StackingEnsemble()
        model_dir = os.path.join(os.path.dirname(__file__), "../boosting_model/models")
        
        if os.path.exists(os.path.join(model_dir, 'meta_model.pkl')):
            self.model.load_model(model_dir)
        else:
            print("⚠️ 학습된 모델이 없습니다.")

    def _get_market_data(self):
        """
        AI 점수뿐만 아니라 변동성, 모멘텀 등 '성향 분석'에 필요한 지표도 같이 가져옴
        """
        # 1. 피처 생성 (X: 데이터, y: 타겟, codes: 종목코드)
        X, _, stock_codes = self.engineer.create_final_features()
        
        if X is None: return None

        # 2. AI 상승 확률 예측
        probs = self.model.predict(X)
        
        # 3. 분석용 데이터프레임 생성
        # 팀원이 만든 피처 이름(volatility_10, momentum_5)을 그대로 활용
        df = pd.DataFrame({
            'code': stock_codes,
            'ai_score': np.round(probs * 100, 2),
            'volatility': X['volatility_10'],  # 변동성 (위험도)
            'momentum': X['momentum_5'],       # 최근 추세 (상승세)
            'price_change': X['price_change_5'] # 최근 등락률
        })
        
        return df

    def recommend_by_style(self, style='balanced', top_n=5, exclude_codes=[]):
        """
        [핵심 기능] 투자 성향별 종목 추천 알고리즘
        style: 'conservative'(안정), 'aggressive'(공격), 'momentum'(추세), 'reversal'(저점매수)
        """
        print(f"\n🔍 [{style.upper()}] 성향 맞춤 추천 분석 중...")
        
        df = self._get_market_data()
        if df is None: return []

        # 이미 가진 종목은 제외 (exclude_codes)
        df = df[~df['code'].isin(exclude_codes)]

        # ----------------------------------------------------
        # 🎯 성향별 필터링 로직 (여기가 핵심!)
        # ----------------------------------------------------
        
        # 공통 조건: 일단 AI가 오른다고 한 종목이어야 함 (60점 이상)
        candidates = df[df['ai_score'] >= 60].copy()
        
        if style == 'conservative': # 🛡️ 안정형
            # 변동성이 낮고, AI 확신도가 높은 종목
            # volatility 하위 50%만 선택
            threshold = candidates['volatility'].quantile(0.5)
            recs = candidates[candidates['volatility'] <= threshold]
            # 정렬 기준: AI 점수 높은 순
            recs = recs.sort_values('ai_score', ascending=False)

        elif style == 'aggressive': # 🚀 공격형
            # 변동성이 높더라도 AI 점수가 압도적으로 높은 종목
            # 정렬 기준: AI 점수 최우선
            recs = candidates.sort_values('ai_score', ascending=False)

        elif style == 'momentum': # 🔥 불기둥형 (추세추종)
            # 최근 5일간 상승세(Momentum > 0)이고 AI도 좋게 본 종목
            recs = candidates[candidates['momentum'] > 0]
            # 정렬 기준: 모멘텀(상승세) 강한 순서
            recs = recs.sort_values('momentum', ascending=False)

        elif style == 'reversal': # 💎 줍줍형 (저점매수)
            # 최근 가격은 떨어졌는데(Price Change < 0), AI 점수는 높은 종목 (반등 예상)
            recs = candidates[candidates['price_change'] < 0]
            # 정렬 기준: AI 점수 높은 순 (확신도)
            recs = recs.sort_values('ai_score', ascending=False)

        else: # 기본 (Balanced)
            recs = candidates.sort_values('ai_score', ascending=False)

        # Top N 추출
        final_recs = recs.head(top_n)
        
        # 결과 포맷팅
        result_list = []
        for _, row in final_recs.iterrows():
            result_list.append({
                'code': row['code'],
                'ai_score': row['ai_score'],
                'reason': f"변동성 {row['volatility']:.4f} / 모멘텀 {row['momentum']:.1f}"
            })
            
        return result_list

    def analyze_portfolio(self, current_portfolio):
        """
        [종합 진단] 내 포트폴리오 + 신규 추천
        current_portfolio: ['005930', '000660'] (보유 종목 리스트)
        """
        print("\n" + "="*50)
        print("📊 AI 포트폴리오 종합 진단")
        print("="*50)

        # 1. 기존 보유 종목 밸류체인 추천 (GCN)
        print("\n1️⃣ 보유 종목 기반 연관 추천 (Value Chain)")
        for stock in current_portfolio:
            related = self.analyze_value_chain(stock, top_n=2) # 아래 구현된 함수 호출
            if related:
                print(f"   👉 [{stock}] 보유자라면? -> {[r['code'] for r in related]} 추천")

        # 2. 성향별 신규 추천 (Boosting + Filtering)
        styles = ['conservative', 'aggressive', 'momentum', 'reversal']
        
        print("\n2️⃣ 투자 성향별 신규 추천 (AI Filtering)")
        for style in styles:
            recs = self.recommend_by_style(style, top_n=3, exclude_codes=current_portfolio)
            print(f"   [{style}] 추천: {[r['code'] for r in recs]}")

    def analyze_value_chain(self, target_code, top_n=5):
        """(기존 GCN 밸류체인 로직 유지)"""
        embeddings = self.engineer.load_gcn_embeddings()
        mapping = self.engineer.load_stock_mapping()
        
        if embeddings is None or mapping is None or target_code not in mapping:
            return []

        target_idx = mapping[target_code]
        target_vec = embeddings[target_idx].unsqueeze(0).numpy()
        all_vecs = embeddings.numpy()
        
        sim_scores = cosine_similarity(target_vec, all_vecs)[0]
        sorted_indices = sim_scores.argsort()[::-1]
        
        recommendations = []
        idx_to_code = {v: k for k, v in mapping.items()}
        
        for idx in sorted_indices:
            code = idx_to_code.get(idx)
            if idx != target_idx and code and code.isdigit() and len(code) == 6:
                recommendations.append({'code': code, 'similarity': round(sim_scores[idx], 4)})
                if len(recommendations) >= top_n: break
                    
        return recommendations

# 테스트 실행
if __name__ == "__main__":
    csv_path = r"C:\rookies4dev\final_project\MyEggBasket-AI\20251120.csv"
    analyzer = PortfolioAnalyzer(csv_path)
    
    # 내 포트폴리오 가상 데이터
    my_portfolio = ['005930', '035420'] # 삼성전자, NAVER
    
    analyzer.analyze_portfolio(my_portfolio)


'''

---

### 📝 코드 사용법

1.  이 파일은 **웹 서버(FastAPI/Django)**에서 사용자의 입력을 받아 실행되는 부분입니다.
2.  사용자가 "나는 **안정형**이야"라고 선택하면?
    * `analyzer.recommend_by_style('conservative')`를 호출해서 결과를 보여주면 됩니다.
3.  사용자가 "나는 **불기둥(급등주)**만 원해"라고 하면?
    * `analyzer.recommend_by_style('momentum')`을 호출합니다.

### 🖼️ 기대 결과 (Output)

```text
📊 AI 포트폴리오 종합 진단
==================================================

1️⃣ 보유 종목 기반 연관 추천 (Value Chain)
   👉 [005930] 삼성전자 보유자라면? -> ['000660', '005380'] 추천 (하이닉스, 현대차)

2️⃣ 투자 성향별 신규 추천 (AI Filtering)
   [conservative] 추천: ['035720', '003550', '030200'] (카카오, LG, KT) -> 변동성 낮음
   [aggressive] 추천: ['000660', '006400', '011070'] (하이닉스, 삼성SDI..) -> 점수 매우 높음
   [momentum] 추천: ['086520', '247540'] (에코프로, 에코프로비엠) -> 최근 급등 중
   [reversal] 추천: ['051910'] (LG화학) -> 많이 떨어졌는데 AI가 반등 예측

   '''