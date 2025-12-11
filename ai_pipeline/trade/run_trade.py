import sys
import os
import pandas as pd
import numpy as np
import time
from elasticsearch import Elasticsearch

# 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 내부 모듈 Import
from ai_pipeline.trade.config import ES_HOST, DATA_DIR, TARGET_DATES
from ai_pipeline.trade.kis_api import KISMockTrader
from ai_pipeline.trade.data_loader import load_data_and_merge_news
from ai_pipeline.boosting_model.train import StackingEnsemble

# [NEW] 피처 정형화 모듈 Import (검문소)
try:
    from ai_pipeline.utils.feature_manager import align_features
except ImportError:
    print(" [Error] feature_manager를 찾을 수 없습니다. ai_pipeline/utils 폴더를 확인하세요.")
    sys.exit(1)

def main():
    print(f" [Auto Trader] 시스템 시작")
    
    # -------------------------------------------------------------------
    # 1. 데이터 로드 및 병합
    # -------------------------------------------------------------------
    es = Elasticsearch(ES_HOST)
    all_data_list = []
    
    print("\n 데이터 로딩 중...")
    for date_str in TARGET_DATES:
        print(f"   Processing {date_str}...", end=" ")
        df_day = load_data_and_merge_news(date_str, DATA_DIR, es)
        if df_day is not None:
            print(f" {len(df_day)} rows")
            all_data_list.append(df_day)
        else:
            print(" Skipped")
            
    if not all_data_list:
        print(" 학습 가능한 데이터가 없습니다.")
        return

    full_df = pd.concat(all_data_list, ignore_index=True)
    
    # -------------------------------------------------------------------
    # 2. 모델 학습
    # -------------------------------------------------------------------
    print("\n 모델 학습 시작...")
    
    # [수정] Feature Manager를 사용해 학습 데이터 컬럼 강제 통일
    # (train_cols 수동 지정 로직 제거 -> align_features 사용)
    X = align_features(full_df)
    y = full_df['target']
    
    model = StackingEnsemble()
    model.feature_names = list(X.columns) # 정형화된 컬럼 목록 저장
    model.train(X, y)
    print(" 모델 학습 완료!")

    # -------------------------------------------------------------------
    # 3. 매매 시뮬레이션
    # -------------------------------------------------------------------
    print("\n AI 트레이더 예측 실행...")
    last_date_df = all_data_list[-1]
    
    # 종목별 최신 상태 추출
    current_market = last_date_df.drop_duplicates(subset=['stock_code'], keep='last').copy()
    print(f"    예측 대상 종목 수: {len(current_market)}개")

    # Feature Manager를 사용해 예측 데이터 컬럼 강제 통일
    # (수동 for문 로직 제거 -> align_features 사용)
    # 없는 컬럼(ai_score 등)은 알아서 채워지고, 순서도 학습 때와 똑같이 맞춰짐
    X_live = align_features(current_market)
    
    # 확률 예측 (Raw Probability)
    raw_probs = model.predict_proba(X_live)[:, 1]
    
    # -------------------------------------------------------------
    # 점수 보정 (Min-Max Scaling)
    # 기계적인 확률(0.01)을 인간적인 점수(0~100)로 변환합니다.
    # -------------------------------------------------------------
    min_p = raw_probs.min()
    max_p = raw_probs.max()
    
    # 분모가 0이 되는 것을 방지
    if max_p - min_p == 0:
        scaled_score = np.zeros_like(raw_probs)
    else:
        # 공식: (내점수 - 꼴등점수) / (1등점수 - 꼴등점수) * 100
        scaled_score = (raw_probs - min_p) / (max_p - min_p) * 100
        
    current_market['ai_score'] = scaled_score
    current_market['raw_prob'] = raw_probs * 100 # 원본 확률도 참고용으로 저장 (단위 %)

    print(f"    점수 보정 완료 | 원본 최고: {max_p*100:.4f}% -> 보정 후: 100점")

    # 상위 5개 선정
    buy_candidates = current_market.sort_values('ai_score', ascending=False).head(5)

    print("\n [AI 추천 종목 Top 5 (상대평가)]")
    if buy_candidates.empty:
        print("    추천할 종목이 없습니다.")
    else:
        # 보기 좋게 출력
        print(buy_candidates[['stock_code', 'ai_score', 'raw_prob', 'stck_prpr']])

    # -------------------------------------------------------------------
    # 4. 주문 실행
    # -------------------------------------------------------------------
    kis = KISMockTrader()
    
    if not buy_candidates.empty:
        print("\n 주문 실행 중...")
        for idx, row in buy_candidates.iterrows():
            code = row['stock_code']
            score = row['ai_score']
            raw_p = row['raw_prob']
            price = int(row['stck_prpr'])
            
            if price <= 0:
                print(f"    [{code}] 현재가 0원이라 주문 불가")
                continue
            
            # (옵션) 절대 확률이 너무 낮으면 스킵하는 안전장치
            # if raw_p < 0.1:
            #     print(f"    [{code}] 1등이지만 확률이 너무 낮음({raw_p:.2f}%) -> 패스")
            #     continue

            print(f"    [{code}] 점수: {score:.1f}점 (확률 {raw_p:.2f}%) -> 매수 주문 (10주)")
            kis.buy_limit(stock_code=code, price=price, qty=10)
            time.sleep(0.2) # API 과부하 방지
    else:
        print("\n 매수 진행 안 함.")

if __name__ == "__main__":
    main()