"""
공시 데이터(DB 문서 또는 통합 CSV)를 학습용 수치 피처로 변환하는 헬퍼

주요 역할:
- 공시 DataFrame을 받아 종목 코드별로 정규화된 수치형 피처를 생성합니다.
- 한글 문자열 컬럼(`corp_name`, `report_name`)은 기본적으로 제외합니다.
- 파생 지표(debt_to_equity, net_margin 등)를 생성하고 `disc_` 접두사를 붙입니다.

사용 예:
    from ai_pipeline.boosting_model.disclosure_feature_transform import transform_disclosure_df

    df = pd.read_csv('ai_pipeline/disclosure_pipeline/data/integrated_financial_data.csv')
    disc_feats = transform_disclosure_df(df)

    # X는 realtime_feature_loader 또는 feature_engineering이 만든 DataFrame
    X = X.merge(disc_feats, left_on='stock_code', right_index=True, how='left')

주의:
- 트리 기반 모델(XGBoost/LightGBM)은 스케일링이 필요 없으나 결측치는 처리(예: 0으로 채우기 또는 모델에 맡김)하세요.
"""

from typing import Optional
import pandas as pd
import numpy as np


def transform_disclosure_df(df: pd.DataFrame, prefix: str = 'disc_', one_hot_reprt: bool = False) -> pd.DataFrame:
    """
    공시 DataFrame을 받아서 학습에 바로 넣을 수 있는 수치형 피처 DataFrame으로 변환합니다.

    Args:
        df: raw disclosure DataFrame (CSV에서 읽은 것 혹은 Mongo에서 로드한 pandas.DataFrame)
        prefix: 반환 컬럼에 붙일 접두사 (기본 'disc_')
        one_hot_reprt: `reprt_code`를 원-핫 인코딩 할지 여부 (False면 정수로 둠)

    Returns:
        DataFrame indexed by `stock_code` (6자리 문자열) containing numeric features.
    """

    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()

    # --- 종목 코드 정규화 ---
    if 'stock_code' in df.columns:
        df['stock_code'] = df['stock_code'].astype(str).str.strip().str.zfill(6)
    else:
        raise KeyError('입력 DataFrame에 `stock_code` 컬럼이 필요합니다.')

    # --- 중복/버전 정리: rcept_dt가 있으면 최신을 사용 ---
    if 'rcept_dt' in df.columns:
        # rcept_dt가 날짜형 문자열(YYYYMMDD)라고 가정
        try:
            df['rcept_dt_sort'] = pd.to_numeric(df['rcept_dt'], errors='coerce').fillna(0).astype(int)
            df = df.sort_values('rcept_dt_sort')
        except Exception:
            pass

    # keep last per stock_code (+bsns_year if exists)
    if 'bsns_year' in df.columns:
        df = df.drop_duplicates(subset=['stock_code', 'bsns_year'], keep='last')
        df_index = df.set_index('stock_code')
    else:
        df = df.drop_duplicates(subset=['stock_code'], keep='last')
        df_index = df.set_index('stock_code')

    # --- 제외할 컬럼 ---
    for drop_col in ('corp_name', 'report_name'):
        if drop_col in df_index.columns:
            df_index = df_index.drop(columns=[drop_col])

    # --- 기본적으로 수치형으로 변환할 컬럼 목록 ---
    numeric_candidates = [
        'capital_change_count',
        'emp_total_count',
        'fin_net_income',
        'fin_op_income',
        'fin_revenue',
        'fin_total_assets',
        'fin_total_equity',
        'fin_total_liabilities'
    ]

    for col in numeric_candidates:
        if col in df_index.columns:
            df_index[col] = pd.to_numeric(df_index[col], errors='coerce')

    # --- 파생 피처 생성 ---
    out = pd.DataFrame(index=df_index.index)

    def safe_div(a, b):
        return a / (b.replace({0: np.nan}) + 1e-8)

    # 직접 사용 가능한 숫자 피처 복사
    for col in numeric_candidates:
        if col in df_index.columns:
            out[prefix + col] = df_index[col].fillna(0)

    # debt to equity
    if 'fin_total_liabilities' in df_index.columns and 'fin_total_equity' in df_index.columns:
        out[prefix + 'de_ratio'] = safe_div(df_index['fin_total_liabilities'].fillna(0), df_index['fin_total_equity'].fillna(0)).fillna(0)

    # net margin (net_income / revenue)
    if 'fin_net_income' in df_index.columns and 'fin_revenue' in df_index.columns:
        out[prefix + 'net_margin'] = safe_div(df_index['fin_net_income'].fillna(0), df_index['fin_revenue'].fillna(0)).fillna(0)

    # operating margin
    if 'fin_op_income' in df_index.columns and 'fin_revenue' in df_index.columns:
        out[prefix + 'op_margin'] = safe_div(df_index['fin_op_income'].fillna(0), df_index['fin_revenue'].fillna(0)).fillna(0)

    # return on assets
    if 'fin_net_income' in df_index.columns and 'fin_total_assets' in df_index.columns:
        out[prefix + 'roa'] = safe_div(df_index['fin_net_income'].fillna(0), df_index['fin_total_assets'].fillna(0)).fillna(0)

    # net income sign / profitability flags
    if 'fin_net_income' in df_index.columns:
        out[prefix + 'is_profitable'] = (df_index['fin_net_income'] > 0).astype(int)

    # treasury stock event -> binary flag
    if 'treasury_stock_event' in df_index.columns:
        out[prefix + 'treasury_event'] = df_index['treasury_stock_event'].astype(str).str.upper().eq('Y').astype(int)

    # reprt_code: 정수형으로 넣기(또는 원-핫)
    if 'reprt_code' in df_index.columns:
        try:
            df_index['reprt_code_int'] = pd.to_numeric(df_index['reprt_code'], errors='coerce').fillna(0).astype(int)
            if one_hot_reprt:
                dummies = pd.get_dummies(df_index['reprt_code_int'], prefix=prefix + 'reprt')
                out = pd.concat([out, dummies], axis=1)
            else:
                out[prefix + 'reprt_code'] = df_index['reprt_code_int']
        except Exception:
            pass

    # 기타: emp_total_count가 있으면 인구밀도 관련 ratio 생성
    if 'emp_total_count' in df_index.columns and 'fin_total_assets' in df_index.columns:
        out[prefix + 'assets_per_emp'] = safe_div(df_index['fin_total_assets'].fillna(0), df_index['emp_total_count'].replace({0: np.nan}).fillna(0)).fillna(0)

    # 최종: NaN을 0으로 채우되, 필요에 따라 예측 시 다른 전략을 사용
    out = out.fillna(0)

    return out


if __name__ == '__main__':
    import os
    # 간단한 로컬 테스트(파일 경로는 프로젝트 구조에 따라 조정)
    path = os.path.join(os.path.dirname(__file__), '..', 'disclosure_pipeline', 'data', 'integrated_financial_data.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, encoding='utf-8-sig')
        feats = transform_disclosure_df(df)
        print(feats.head())
