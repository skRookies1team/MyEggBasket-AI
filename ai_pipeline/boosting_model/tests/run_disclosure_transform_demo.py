import sys
import os
import pandas as pd

# 프로젝트 루트를 PYTHONPATH에 추가 (패키지 import를 위해)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai_pipeline.boosting_model.disclosure_feature_transform import transform_disclosure_df


def main():
    # 원본 문서 (사용자 제공 예시)
    doc = {
        '_id': "ObjectId('693673e00fd15a9b7c7291be')",
        'stock_code': '950210',
        'bsns_year': 2025,
        'capital_change_count': 1,
        'corp_code': 1510489,
        'corp_name': '프레스티지바이오파마',
        'emp_total_count': 8,
        'fin_net_income': 32953738,
        'fin_op_income': -10946913,
        'fin_revenue': None,
        'fin_total_assets': 541398347,
        'fin_total_equity': 368958940,
        'fin_total_liabilities': 172439407,
        'rcept_dt': '20251128',
        'rcept_no': '20251128002047',
        'report_name': '분기보고서 (2025.09)',
        'reprt_code': 11014,
        'treasury_stock_event': 'Y'
    }

    print('\n=== 원본 문서 (dict) ===')
    for k, v in doc.items():
        print(f"{k}: {v}")

    df = pd.DataFrame([doc])
    feats = transform_disclosure_df(df)

    print('\n=== 변환된 disclosure 피처 (index=stock_code) ===')
    if feats.empty:
        print('변환 결과가 비어있습니다')
    else:
        # 보기 좋게 transpose 해서 출력
        print(feats.T)


if __name__ == '__main__':
    main()
