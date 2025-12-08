"""DisclosureAutoCollector를 사용해 공시 원시 데이터를 수집하여 `data/`에 CSV로 저장합니다.

사용법:
    python collect_disclosures_to_csv.py --date 20251128
    python collect_disclosures_to_csv.py --start 20251101 --end 20251130

OpenDART API 키나 Mongo 연결 정보가 없을 경우, 로컬의
`data/financial_disclosure_data.csv` 파일을 대체로 복사합니다.
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="수집할 단일 날짜 YYYYMMDD (기본: 어제)")
    parser.add_argument("--start", help="수집 시작일 YYYYMMDD")
    parser.add_argument("--end", help="수집 종료일 YYYYMMDD")
    parser.add_argument("--out", help="출력 CSV 경로 (기본: disclosure_pipeline/data)")
    args = parser.parse_args()

    out_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if args.out:
        out_path = args.out
    else:
        target = args.date or args.start or datetime.now().strftime("%Y%m%d")
        out_path = os.path.join(out_dir, f"raw_disclosures_{target}.csv")

    # Try to use existing collector if available. If standard import fails,
    # attempt to load the collector module directly from file path.
    DisclosureAutoCollector = None
    load_error = None
    try:
        from ai_pipeline.disclosure_pipeline.disclosure_auto_collector import DisclosureAutoCollector
    except Exception as e:
        load_error = e
        # attempt to load module by file path as a fallback
        try:
            import importlib.util
            module_path = os.path.join(os.path.dirname(__file__), 'disclosure_auto_collector.py')
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location('disclosure_auto_collector', module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                DisclosureAutoCollector = getattr(module, 'DisclosureAutoCollector', None)
        except Exception as e2:
            load_error = e2

    if DisclosureAutoCollector is not None:
        try:
            collector = DisclosureAutoCollector()

            if args.start and args.end:
                bgn = args.start
                end = args.end
            elif args.date:
                bgn = end = args.date
            else:
                yesterday = datetime.now()
                bgn = end = yesterday.strftime("%Y%m%d")

            print(f"공시 수집 시작: {bgn} ~ {end} ...")
            # 실제 수집 호출: OpenDART API 등 외부 서비스에서 공시 데이터를 받아옵니다.
            # collector는 pandas DataFrame을 반환해야 합니다.
            df = collector.collect_disclosures(bgn, end)
            if df is None or df.empty:
                print("데이터를 수집하지 못했거나 수집기가 사용 불가합니다. 로컬 CSV로 대체 시도합니다.")
                raise RuntimeError("No data")

            df.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"원시 공시 CSV를 저장했습니다: {out_path}")
            try:
                collector.close()
            except Exception:
                pass
            return 0
        except Exception as e:
            load_error = e

    # If we reach here, collector is not usable — fallback to copying existing local CSV
    # 수집기 사용 불가시 로컬 파일을 복사하는 안전장치
    print(f"수집기 사용 불가 또는 실패: {load_error}")
    fallback = os.path.join(os.path.dirname(__file__), "data", "financial_disclosure_data.csv")
    if os.path.exists(fallback):
        import shutil
        shutil.copy2(fallback, out_path)
        print(f"로컬 대체 CSV를 복사했습니다: {out_path}")
        return 0
    else:
        print("대체할 로컬 CSV를 찾을 수 없습니다. 작업을 종료합니다.")
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
