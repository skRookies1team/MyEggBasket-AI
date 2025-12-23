# ai_pipeline/utils/feature_manager.py
import pandas as pd
import numpy as np
from ai_pipeline.config.feature_columns import ALL_MODEL_FEATURES

def align_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    입력된 DataFrame을 학습 기준 피처 포맷으로 강제 정렬합니다.
    1. 없는 컬럼은 기본값으로 생성 (ai_score=50.0, 나머지=0.0)
    2. 순서 강제 통일
    """
    df_aligned = df.copy()
    
    # 1. 누락된 컬럼 채우기
    missing_cols = [col for col in ALL_MODEL_FEATURES if col not in df_aligned.columns]
    
    if missing_cols:
        # print(f"🔧 [FeatureManager] 누락된 피처 보정: {len(missing_cols)}개")
        for col in missing_cols:
            # ★ 핵심 해결: ai_score가 없으면 50(중립)으로 채움
            if col == 'ai_score':
                df_aligned[col] = 50.0
            else:
                df_aligned[col] = 0.0

    # 2. 순서 강제 정렬 (모델은 순서에 매우 민감함)
    df_final = df_aligned[ALL_MODEL_FEATURES]
    
    return df_final