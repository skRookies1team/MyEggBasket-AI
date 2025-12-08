"""원시 공시 CSV를 전처리하여 모델 입력용 통합 CSV로 저장합니다.

이 스크립트는 노트북에서 사용한 일반적인 정규화 단계를 적용합니다:
- `stock_code` 존재 보장 (stock_code_x/stock_code_y 또는 corp_code 기반)
- `bsns_year` 존재 보장 (report_nm 또는 rcept_dt에서 추출)
- 숫자형 컬럼 정규화 및 일부 재무 컬럼 명칭 통일
- 기본적으로 `data/integrated_financial_data.csv`로 저장

사용법:
    python preprocess_disclosures_csv.py --input data/raw_disclosures_20251128.csv
    python preprocess_disclosures_csv.py --input data/financial_disclosure_data.csv --out data/integrated_financial_data.csv
"""
import argparse
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "data", "financial_disclosure_data.csv")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "data", "integrated_financial_data.csv")


def normalize_stock_code(df):
    # 명시적인 종목코드 컬럼을 우선 사용합니다
    for cand in ("stock_code", "stock_code_x", "stock_code_y", "stck_shrn_iscd"):
        if cand in df.columns:
            df['stock_code'] = df[cand].astype(str).str.strip().str.zfill(6)
            return df
    # corp_code가 있으면 별도 매핑이 필요할 수 있으나 우선 빈 칼럼을 둡니다
    if 'corp_code' in df.columns:
        df['stock_code'] = df.get('stock_code', pd.Series([None]*len(df)))
    return df


def ensure_bsns_year(df):
    # bsns_year 컬럼이 없으면 report_nm 또는 rcept_dt에서 추출합니다
    if 'bsns_year' in df.columns:
        return df

    # report_nm에서 연도 추출 시도
    if 'report_nm' in df.columns:
        years = df['report_nm'].astype(str).str.extract(r"(\d{4})")
        if years is not None and not years.empty:
            df['bsns_year'] = years[0]

    # 실패 시 rcept_dt로부터 연도 추출
    if 'bsns_year' not in df.columns or df['bsns_year'].isnull().all():
        if 'rcept_dt' in df.columns:
            df['bsns_year'] = df['rcept_dt'].astype(str).str[:4]
        else:
            df['bsns_year'] = 0

    # 안전하게 정수로 변환
    try:
        df['bsns_year'] = pd.to_numeric(df['bsns_year'], errors='coerce').fillna(0).astype(int)
    except Exception:
        pass
    return df


def rename_financial_columns(df):
    # 노트북/CSV에서 자주 보이는 컬럼명 매핑
    mapping = {
        'fin_매출액': 'fin_revenue',
        'fin_영업이익': 'fin_op_income',
        'fin_자산총계': 'fin_total_assets',
        'fin_부채총계': 'fin_total_liabilities',
        'fin_자본총계': 'fin_total_equity',
        'fin_당기순이익': 'fin_net_income',
        'emp_count_sample': 'emp_total_count',
        'emp_total_count': 'emp_total_count'
    }

    for k, v in mapping.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # 숫자형으로 변환 가능한 컬럼들에 대해 안전하게 변환
    for col in df.columns:
        if col.startswith('fin_') or col.startswith('emp_') or col.endswith('_count'):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

    return df


def derive_reprt_code(df):
    # small helper to set reprt_code based on report name
    def code_from_name(nm):
        s = str(nm)
        if '.12)' in s or '사업보고서' in s:
            return '11011'
        if '.03)' in s or '1분기' in s:
            return '11013'
        if '.06)' in s or '반기' in s:
            return '11012'
        if '.09)' in s or '3분기' in s:
            return '11014'
        return ''

    if 'reprt_code' not in df.columns and 'report_nm' in df.columns:
        df['reprt_code'] = df['report_nm'].apply(code_from_name)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='원시 공시 CSV 경로', default=DEFAULT_INPUT)
    parser.add_argument('--out', help='출력 통합 CSV 경로', default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # If the provided input path doesn't exist, try resolving it relative
    # to the disclosure_pipeline folder (convenience for CLI usage).
    input_path = args.input
    if not os.path.exists(input_path):
        candidate = os.path.join(os.path.dirname(__file__), args.input)
        if os.path.exists(candidate):
            input_path = candidate
        else:
            print(f"입력 파일을 찾을 수 없습니다: {args.input}")
            return 2

    df = pd.read_csv(input_path, encoding='utf-8-sig')
    print(f"로딩 완료: {input_path}에서 {len(df)}행 불러옴")

    df = normalize_stock_code(df)
    df = ensure_bsns_year(df)
    df = rename_financial_columns(df)
    df = derive_reprt_code(df)

    # drop duplicates by stock_code + bsns_year keeping latest rcept_dt if present
    if 'stock_code' in df.columns:
        sort_key = 'rcept_dt' if 'rcept_dt' in df.columns else None
        if sort_key:
            df = df.sort_values(sort_key, ascending=False)
        df = df.drop_duplicates(subset=['stock_code', 'bsns_year'], keep='first')

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f"통합 공시 CSV 저장 완료: {args.out} ({len(df)} 행)")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
