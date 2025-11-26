import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.boosting_model.train import StackingEnsemble
from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer

class StockPredictor:
    """학습된 스태킹 모델로 종목 예측"""
    
    def __init__(self, model_dir='models'):
        self.model = StackingEnsemble()
        
        if os.path.exists(model_dir):
            self.model.load_model(model_dir)
        else:
            print(f"⚠️ 모델 파일이 없습니다: {model_dir}")
            print("   먼저 train.py를 실행하세요.")
        
        self.engineer = FeatureEngineer()
    
    def predict_stocks(self, stock_codes):
        """
        종목별 다음 날 상승/하락 예측
        
        Returns:
        - DataFrame: 종목별 예측 결과
        """
        print(f"\n📊 {len(stock_codes)}개 종목 예측 중...")
        
        # 피처 생성
        X, _ = self.engineer.create_final_features(stock_codes, use_dummy=True)
        
        if X is None:
            print("❌ 피처 생성 실패")
            return None
        
        # 각 종목의 최신 데이터만 사용 (마지막 행)
        # 실제로는 오늘 날짜 데이터를 필터링해야 함
        latest_data = []
        stock_code_list = []
        
        # 더미 데이터에서는 종목별로 마지막 데이터 추출
        for code in stock_codes:
            # 실제 구현에서는 날짜로 필터링
            latest_data.append(X.iloc[-1])  # 임시: 마지막 행 사용
            stock_code_list.append(code)
        
        if not latest_data:
            print("❌ 예측할 데이터가 없습니다.")
            return None
        
        X_latest = pd.DataFrame(latest_data)
        
        # 예측
        predictions_proba = self.model.predict(X_latest)
        predictions_class = self.model.predict_class(X_latest)
        
        # 결과 정리
        results = pd.DataFrame({
            '종목코드': stock_code_list,
            '예측확률': predictions_proba,
            '예측': ['상승 ↑' if p == 1 else '하락 ↓' for p in predictions_class],
            '신뢰도': [f"{prob*100:.1f}%" if pred == 1 else f"{(1-prob)*100:.1f}%" 
                     for prob, pred in zip(predictions_proba, predictions_class)]
        })
        
        return results
    
    def predict_top_stocks(self, stock_codes, top_n=5):
        """상승 확률 TOP N 추천"""
        results = self.predict_stocks(stock_codes)
        
        if results is None:
            return None
        
        results_sorted = results.sort_values('예측확률', ascending=False)
        top_stocks = results_sorted.head(top_n)
        
        print("\n" + "="*60)
        print(f"🔥 상승 확률 TOP {top_n}")
        print("="*60)
        print(top_stocks.to_string(index=False))
        print("="*60)
        
        return top_stocks


def run_prediction():
    """예측 실행 함수"""
    print("="*60)
    print("🔮 주식 상승/하락 예측 시작")
    print("="*60)
    
    # 예측기 초기화
    predictor = StockPredictor()
    
    # GCN 그래프에 있는 종목 사용
    predictor.engineer.load_stock_mapping()
    stock_codes = list(predictor.engineer.stock_mapping.keys())[:20]
    
    print(f"\n📌 예측 대상: {len(stock_codes)}개 종목")
    
    # 전체 예측
    results = predictor.predict_stocks(stock_codes)
    
    if results is not None:
        print("\n" + "="*60)
        print("📊 예측 결과")
        print("="*60)
        print(results.to_string(index=False))
        print("="*60)
        
        up_count = (results['예측'] == '상승 ↑').sum()
        down_count = (results['예측'] == '하락 ↓').sum()
        
        print(f"\n📈 상승 예측: {up_count}개")
        print(f"📉 하락 예측: {down_count}개")
    
    # TOP 5 추천
    predictor.predict_top_stocks(stock_codes, top_n=5)


if __name__ == "__main__":
    run_prediction()