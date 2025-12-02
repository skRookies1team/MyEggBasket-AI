## 투자성향 알고리즘 추가 예정 ( 안정형/ 공격형/ 불기둥형/.... ) 
### 포트폴리오 외 추천 

import sys
import os
import numpy as np
import pandas as pd

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer
from ai_pipeline.boosting_model.train import StackingEnsemble
from ai_pipeline.gcn_model.value_chain import ValueChainAnalyzer

try:
    from value_chain import ValueChainAnalyzer
except ImportError:
    class ValueChainAnalyzer:
        def find_similar_stocks(self, code, top_n=5): return []
        def get_stock_name(self, code): return code

# 가상의 모듈 
# from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer
# from ai_pipeline.boosting_model.train import StackingEnsemble

class PortfolioAnalyzer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.vc_analyzer = ValueChainAnalyzer()
        
        # [핵심 1] 캐시 메모리 초기화 (중복 연산 방지용)
        self.cached_data = None 
        
        # 모델 로드
        self.model = None
        self.use_mock = False
        
        try:
            self.engineer = FeatureEngineer(csv_path)
            self.model = StackingEnsemble()
            model_dir = os.path.join(os.path.dirname(__file__), "../../boosting_model/models")
            model_path = os.path.join(model_dir, 'meta_model.pkl')
            
            if os.path.exists(model_path):
                self.model.load_model(model_dir)
                print(f"✅ 모델 로드 완료: {model_dir}")
            else:
                raise FileNotFoundError("모델 파일 없음")
                
        except Exception as e:
            print(f"⚠️ [System] 모델 로드 실패 ({e}) -> 가상 모드 전환")
            self.model = self._create_mock_model()
            self.use_mock = True


    def _create_mock_model(self):
        class MockModel:
            def predict_proba(self, X):
                n = len(X)
                probs = np.random.uniform(0.3, 0.95, size=(n, 2))
                return probs
            def predict(self, X):
                return np.random.uniform(0.3, 0.95, size=len(X))
        return MockModel()
    


    def _get_market_data(self):
        """
        AI 점수 계산 및 데이터 캐싱 처리
        """
        # [핵심 2] 이미 계산한 데이터가 있으면 바로 반환 (GCN 재연산 방지)
        if self.cached_data is not None:
            return self.cached_data

        X = None
        stock_codes = []
        
        # 1. 피처 생성 (GCN 등 무거운 작업)
        if not self.use_mock:
            try:
                features_ret = self.engineer.create_final_features()
                
                if len(features_ret) == 3:
                    X, _, stock_codes = features_ret
                elif len(features_ret) == 2:
                    X, stock_codes = features_ret
                else:
                    X = None
            except Exception as e:
                print(f"⚠️ 피처 생성 오류: {e}")
                X = None

        # 2. 데이터 없을 시 가상 데이터
        if X is None or self.model is None:
            stock_codes = ['005930', '000660', '035420', '035720', '005380']
            X = pd.DataFrame(np.random.randn(len(stock_codes), 5), 
                           columns=['volatility_10', 'momentum_5', 'price_change_5', 'stck_prpr', 'dummy'])
            up_probs = np.random.uniform(0.4, 0.95, size=len(stock_codes))
        else:
            # 3. AI 예측
            try:
                probs = self.model.predict_proba(X)
                if hasattr(probs, 'ndim') and probs.ndim == 2 and probs.shape[1] >= 2:
                    up_probs = probs[:, 1]
                else:
                    up_probs = probs.flatten()
            except:
                up_probs = self.model.predict(X)

        # 4. 종목명 매핑
        formatted_codes = []
        for code in stock_codes:
            # 1. 문자로 변환 ('660')
            str_code = str(code).strip()
            # 2. 6자리가 될 때까지 앞에 '0' 채우기
            full_code = str_code.zfill(6) 
            formatted_codes.append(full_code)
        
        stock_codes = formatted_codes

        names = []
        try:
            for code in stock_codes:
                # 이제 code가 '000660' 형식이므로 이름 매핑이 정상 작동함
                name = self.vc_analyzer.get_stock_name(code)
                names.append(name)
        except:
            names = stock_codes


        # 5. 결과 DataFrame 생성
        df = pd.DataFrame({
            'code': stock_codes,
            'name': names,
            'ai_score': np.round(up_probs * 100, 1),
            'volatility': X.get('volatility_10', [0]*len(X)), 
            'momentum': X.get('momentum_5', [0]*len(X)),      
            'price_change': X.get('price_change_5', [0]*len(X)), 
            'current_price': X.get('stck_prpr', [0]*len(X))
        })
        
        # [핵심 3] 결과 캐싱 (다음에 또 쓰기 위해 저장)
        self.cached_data = df
        return df


    def get_ai_scores(self, filter_codes=None):
        """
        [기능 1] AI 상승 확률(Score) 계산
        """
        df = self._get_market_data()
        
        # [수정] df가 None이거나 비어있으면 빈 결과를 반환 (에러 방지)
        if df is None or df.empty:
            return pd.DataFrame(columns=['code', 'ai_score'])

        if filter_codes:
            # 존재하는 종목만 필터링
            filtered_df = df[df['code'].isin(filter_codes)].copy()
            # 없는 것 찾아내기
            existing_codes = set(filtered_df['code'])
            missing_codes = set(filter_codes) - existing_codes

            if missing_codes:
                print(f"   ⚠️ CSV 데이터 미포함 종목: {missing_codes} (점수 예측 불가)")
                # 누락 종목은 점수 0점 또는 NaN으로 추가 표시
                missing_data = []
                for code in missing_codes:
                    name = self.vc_analyzer.get_stock_name(code)
                    missing_data.append({
                        'code': code, 
                        'name': name, 
                        'ai_score': 0.0, # 데이터 없으므로 0점 처리
                        'note': '데이터없음'
                    })

                if missing_data:
                    missing_df = pd.DataFrame(missing_data)
                    filtered_df = pd.concat([filtered_df, missing_df], ignore_index=True)
            
            return filtered_df[['code', 'name', 'ai_score']].sort_values('ai_score', ascending=False)
            
        return df[['code', 'name', 'ai_score']].sort_values('ai_score', ascending=False)
            
  
    def analyze_value_chain(self, target_code, top_n=5):
        """
        [기능 2] CSV 밸류체인 데이터를 이용한 연관 종목 추천
        """
        # CSV 기반 분석기 호출 (이전 단계에서 만든 코드 사용)
        recommendations = self.vc_analyzer.find_similar_stocks(target_code, top_n)
        
        if not recommendations:
            return []
            
        # 결과 반환 (Boosting 점수도 같이 보여주면 좋음 - 여기선 일단 밸류체인만)
        return recommendations


    def recommend_by_style(self, style='balanced', top_n=5, exclude_codes=[]):
        """
        [기능 3] 투자 성향별 종목 추천 알고리즘
        style: 'conservative'(안정), 'aggressive'(공격), 'momentum'(추세), 'reversal'(저점매수)
        """
        print(f"\n🔍 [{style.upper()}] 성향 맞춤 추천 분석 중...")
        
        df = self._get_market_data()
        if df is None or df.empty: return []

        # 이미 가진 종목은 제외 (exclude_codes)
        if exclude_codes:
            df = df[~df['code'].isin(exclude_codes)]

        # ----------------------------------------------------
        # 🎯 성향별 필터링 로직 (여기가 핵심!)
        # ----------------------------------------------------
        
        # 공통 조건: 일단 AI가 오른다고 한 종목이어야 함 (60점 이상)
        candidates = df[df['ai_score'] >= 60].copy()
        if candidates.empty: candidates = df.copy()

        # ----------------------------------------------------
        # 🎯 성향별 필터링 로직
        # ----------------------------------------------------
        if style == 'conservative': # 🛡️ 안정형
            # 변동성 하위 50% & AI 점수순 정렬
            threshold = candidates['volatility'].quantile(0.5)
            recs = candidates[candidates['volatility'] <= threshold]
            recs = recs.sort_values('ai_score', ascending=False)
            reason_template = "저변동성 안정적 흐름"

        elif style == 'aggressive': # 🚀 공격형
            # 변동성 무시, AI 점수 최우선
            recs = candidates.sort_values('ai_score', ascending=False)
            reason_template = "AI 강력 매수 시그널"

        elif style == 'momentum': # 🔥 불기둥형
            # 상승세(Momentum > 0)인 종목 중 모멘텀 강한 순
            recs = candidates[candidates['momentum'] > 0]
            recs = recs.sort_values('momentum', ascending=False)
            reason_template = "강력한 상승 추세 진입"

        elif style == 'reversal': # 💎 줍줍형
            # 최근 가격 하락(Change < 0)했으나 AI 점수 높은 것
            recs = candidates[candidates['price_change'] < 0]
            recs = recs.sort_values('ai_score', ascending=False)
            reason_template = "낙폭 과대 및 반등 기대"

        else: # 기본 (Balanced)
            recs = candidates.sort_values('ai_score', ascending=False)
            reason_template = "AI 추천"


        result_list = []
        for _, row in recs.head(top_n).iterrows():
            result_list.append({
                'code': row['code'],
                'ai_score': row['ai_score'],
                'style': style,
                'reason': f"{reason_template} (AI점수: {row['ai_score']}점)",
                'detail': f"변동성 {row['volatility']:.4f} / 모멘텀 {row['momentum']:.1f}"
            })
            
        return result_list
    


# ==========================================
# 테스트 실행 코드
# ==========================================
if __name__ == "__main__":
    # [수정] data 폴더 내의 '가장 최신 파일'로 경로 변경
    # 예: 20251120.csv -> 20251127.csv (가지고 계신 파일 중 가장 최근 날짜)
    csv_path = r"C:\Users\user\project\MyEggBasket-AI\data\20251127.csv"
    
    if os.path.exists(csv_path):
        # 파일 하나만 넘겨서 그 날짜 기준으로 분석
        analyzer = PortfolioAnalyzer(csv_path)
        
        # [상황] 사용자가 '삼성전자'를 가지고 있음
        my_portfolio = ['005930'] 
        
        print(f"\n💼 보유 종목: {my_portfolio}")

        # 1. 보유 종목 기반 추천 (밸류체인)
        print("\n1️⃣ [밸류체인] 보유 종목과 연관된 추천")
        for stock in my_portfolio:
            recs = analyzer.analyze_value_chain(stock)
            # recs가 None이거나 비어있을 수 있으므로 체크
            if recs:
                for r in recs:
                    print(f"   👉 {r['name']} ({r['code']}) - {r['reason']}")
            else:
                print("   ⚠️ 연관 종목을 찾을 수 없습니다.")

        # 2. 투자 성향별 추천 (신규 발굴)
        print("\n2️⃣ [투자성향] 새로운 종목 발굴")
        
        # 안정형 추천
        conservative = analyzer.recommend_by_style('conservative', top_n=3, exclude_codes=my_portfolio)
        if conservative:
            print(f"   🛡️ 안정형 추천: {[r['code'] for r in conservative]}")
        else:
            print("   🛡️ 안정형 추천 종목 없음")
        
        # 줍줍형 추천
        reversal = analyzer.recommend_by_style('reversal', top_n=3, exclude_codes=my_portfolio)
        if reversal:
            print(f"   💎 줍줍형 추천: {[r['code'] for r in reversal]}")
        else:
            print("   💎 줍줍형 추천 종목 없음")

    else:
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
