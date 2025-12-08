"""통합 공시 CSV를 MongoDB에 업로드(종목코드 + 결산연도 기준 업서트).

사용법:
    python upload_integrated_to_mongo.py --input data/integrated_financial_data.csv

`.env` 또는 환경변수의 `MONGO_URI` 또는 `MONGODB_URI`를 사용해 Mongo에 연결합니다.
"""
import os
import sys
import argparse
import pandas as pd
from pymongo import MongoClient, ASCENDING

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_mongo_uri():
    return os.getenv('MONGO_URI') or os.getenv('MONGODB_URI')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=os.path.join(os.path.dirname(__file__), 'data', 'integrated_financial_data.csv'))
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        # try relative to this script
        candidate = os.path.join(os.path.dirname(__file__), args.input)
        if os.path.exists(candidate):
            input_path = candidate
        else:
            print(f"입력 파일을 찾을 수 없습니다: {args.input}")
            return 2

    mongo_uri = get_mongo_uri()
    if not mongo_uri:
        print("⚠ MONGO_URI 또는 MONGODB_URI가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
        return 3

    print(f"Mongo URI 확인됨, 연결 시도...")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client['stockdb']
    coll = db['disclosure_features']

    # 복합 인덱스 생성 (stock_code, bsns_year) — 중복시 업서트 용도
    coll.create_index([('stock_code', ASCENDING), ('bsns_year', ASCENDING)], unique=True)

    df = pd.read_csv(input_path, encoding='utf-8-sig')
    print(f"로딩 완료: {input_path} ({len(df)} 행)")

    # 컬럼 값 정규화 (종목코드 포맷, 결산연도 변환)
    if 'stock_code' in df.columns:
        df['stock_code'] = df['stock_code'].astype(str).str.strip().str.zfill(6)
    if 'bsns_year' in df.columns:
        df['bsns_year'] = pd.to_numeric(df['bsns_year'], errors='coerce').fillna(0).astype(int)
    else:
        df['bsns_year'] = 0

    inserted = 0
    updated = 0

    for _, row in df.iterrows():
        doc = row.to_dict()
        # NaN 값을 None으로 변환 (MongoDB에 저장하기 위함)
        for k, v in list(doc.items()):
            if pd.isna(v):
                doc[k] = None

        if 'stock_code' not in doc or not doc.get('stock_code'):
            print("⚠ 종목코드 누락 행 스킵")
            continue

        filt = {'stock_code': doc['stock_code'], 'bsns_year': doc.get('bsns_year', 0)}
        try:
            res = coll.update_one(filt, {'$set': doc}, upsert=True)
            if res.upserted_id:
                inserted += 1
            else:
                updated += 1
        except Exception as e:
            print(f"저장 실패 ({doc.get('stock_code')}): {e}")

    print(f"업로드 완료 — 신규: {inserted}, 업데이트: {updated}")
    client.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
