import sys
import os
import pandas as pd
import time
from elasticsearch import Elasticsearch

# 루트 경로 추가 (모듈 import를 위해)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 내부 모듈 Import
from ai_pipeline.trade.config import ES_HOST, DATA_DIR, TARGET_DATES
from ai_pipeline.trade.kis_api import KISMockTrader
from ai_pipeline.trade.data_loader import load_data_and_merge_news
from ai_pipeline.boosting_model.train import StackingEnsemble

def main():
    print(f"🚀 [Auto Trader] 시스템 시작")
    
    # 1. 데이터 로드 및 병합
    es = Elasticsearch(ES_HOST)
    all_data_list = []
    
    print("\n📊 데이터 로딩 중...")
    for date_str in TARGET_DATES:
        print(f"   Processing {date_str}...", end=" ")
        df_day = load_data_and_merge_news(date_str, DATA_DIR, es)
        if df_day is not None:
            print(f"✅ {len(df_day)} rows")
            all_data_list.append(df_day)
        else:
            print("❌ Skipped")
            
    if not all_data_list:
        print("❌ 학습 가능한 데이터가 없습니다.")
        return

    full_df = pd.concat(all_data_list, ignore_index=True)
    
    # 2. 모델 학습
    print("\n🧠 모델 학습 시작...")
    # 학습에 쓸 컬럼만 추출 (정답지, 코드, 날짜, 현재가 제외)
    train_cols = [c for c in full_df.columns if c not in ['target', 'stock_code', 'date', 'timestamp', 'stck_prpr']]
    
    X = full_df[train_cols]
    y = full_df['target']
    
    model = StackingEnsemble()
    model.feature_names = train_cols
    model.train(X, y)
    print("🎉 모델 학습 완료!")

    # 3. 매매 시뮬레이션 (마지막 날짜 데이터 기준)
    print("\n🤖 AI 트레이더 예측 실행...")
    last_date_df = all_data_list[-1]
    
    # 종목별 최신 상태 추출
    current_market = last_date_df.drop_duplicates(subset=['stock_code'], keep='last').copy()
    
    # 예측용 데이터 준비
    X_live = current_market[train_cols]
    X_live = X_live[model.feature_names] # 순서 보장
    
    # 확률 예측
    probs = model.predict_proba(X_live)[:, 1]
    current_market['ai_score'] = probs * 100
    
    # 매수 대상 선정 (예: 80점 이상)
    buy_candidates = current_market[current_market['ai_score'] >= 80].sort_values('ai_score', ascending=False)
    
    print("\n📋 [AI 추천 종목 Top 5]")
    print(buy_candidates[['stock_code', 'ai_score', 'stck_prpr']].head(5))

    # 4. 주문 실행
    kis = KISMockTrader()
    
    if not buy_candidates.empty:
        print("\n💰 주문 실행 중...")
        for idx, row in buy_candidates.head(3).iterrows():
            code = row['stock_code']
            score = row['ai_score']
            price = int(row['stck_prpr'])
            
            # 매수 시도 (현재가)
            print(f"   👉 [{code}] 점수: {score:.1f} -> 매수 주문")
            kis.buy_limit(stock_code=code, price=price, qty=10)
            time.sleep(0.5)
    else:
        print("\n😴 매수할 종목이 없습니다.")

if __name__ == "__main__":
    main()