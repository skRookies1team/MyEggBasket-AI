"""수집 -> 전처리 -> MongoDB 업로드를 한 번에 실행하는 편의 스크립트.

이 파일 하나만 실행하면 다음 3단계가 순차적으로 수행됩니다:
 1) 공시 원시 데이터 수집 (collect_disclosures_to_csv.py)
 2) 전처리하여 통합 CSV 생성 (preprocess_disclosures_csv.py)
 3) MongoDB에 업서트(업로드) (upload_integrated_to_mongo.py)

사용법 예:
    python ai_pipeline/disclosure_pipeline/run_full_pipeline.py --date 20251128
    python ai_pipeline/disclosure_pipeline/run_full_pipeline.py --start 20251101 --end 20251130

모든 출력과 주석은 한국어로 작성되어 있습니다.
"""
import os
import sys
import argparse

# 스크립트 위치에 맞춰 상위 경로를 모듈 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def call_module_main(module, argv_list):
    """주어진 모듈의 main()을 argv_list로 실행하고 리턴코드를 반환합니다.

    module: 모듈 객체
    argv_list: 스크립트 이름 포함한 argv 리스트
    """
    old_argv = sys.argv
    try:
        sys.argv = argv_list
        # 각 모듈은 main()을 정의하고 내부에서 argparse를 사용하므로
        # 직접 main()을 호출하면 정상 동작합니다.
        rc = module.main()
        return int(rc) if rc is not None else 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 0
    except Exception as e:
        print(f"오류: 모듈 실행 중 예외 발생: {e}")
        return 10
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(description='공시 파이프라인 전체 실행기')
    parser.add_argument('--date', help='단일 수집 날짜 YYYYMMDD (옵션)')
    parser.add_argument('--start', help='수집 시작일 YYYYMMDD (옵션)')
    parser.add_argument('--end', help='수집 종료일 YYYYMMDD (옵션)')
    parser.add_argument('--raw-out', help='원시 CSV 출력 경로 (옵션)')
    parser.add_argument('--integrated-out', help='통합 CSV 출력 경로 (옵션)')
    parser.add_argument('--no-upload', action='store_true', help='Mongo 업로드 단계 건너뜀')
    args = parser.parse_args()

    # 각 스크립트의 기본 경로
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 기본 파일명 결정
    target = args.date or args.start or os.path.basename('')
    if args.raw_out:
        raw_out = args.raw_out
    else:
        # 날짜가 지정되지 않으면 기존 스크립트의 기본(현재일자) 로직을 따릅니다.
        if args.date:
            raw_out = os.path.join(data_dir, f"raw_disclosures_{args.date}.csv")
        else:
            raw_out = os.path.join(data_dir, f"raw_disclosures.csv")

    integrated_out = args.integrated_out or os.path.join(data_dir, 'integrated_financial_data.csv')

    print("[1/3] 공시 수집 단계 시작...")
    try:
        from ai_pipeline.disclosure_pipeline import collect_disclosures_to_csv as collect_mod
    except Exception as e:
        print(f"수집 모듈을 불러오는 중 오류: {e}")
        return 20

    collect_argv = [
        'collect_disclosures_to_csv.py'
    ]
    if args.start and args.end:
        collect_argv += ['--start', args.start, '--end', args.end, '--out', raw_out]
    elif args.date:
        collect_argv += ['--date', args.date, '--out', raw_out]
    else:
        # 인자를 제공하지 않으면 collector가 자체적으로 어제 날짜 등으로 처리합니다.
        collect_argv += ['--out', raw_out]

    rc = call_module_main(collect_mod, collect_argv)
    if rc != 0:
        print(f"수집 단계 실패(종료코드 {rc}) — 파이프라인 중단")
        return rc
    print(f"원시 CSV 생성 완료: {raw_out}")

    print("[2/3] 전처리 단계 시작...")
    try:
        from ai_pipeline.disclosure_pipeline import preprocess_disclosures_csv as prep_mod
    except Exception as e:
        print(f"전처리 모듈을 불러오는 중 오류: {e}")
        return 21

    prep_argv = ['preprocess_disclosures_csv.py', '--input', raw_out, '--out', integrated_out]
    rc = call_module_main(prep_mod, prep_argv)
    if rc != 0:
        print(f"전처리 단계 실패(종료코드 {rc}) — 파이프라인 중단")
        return rc
    print(f"통합 CSV 생성 완료: {integrated_out}")

    if args.no_upload:
        print("[3/3] 업로드 단계는 건너뜁니다 (--no-upload). 완료.")
        return 0

    print("[3/3] MongoDB 업로드 단계 시작...")
    try:
        from ai_pipeline.disclosure_pipeline import upload_integrated_to_mongo as upload_mod
    except Exception as e:
        print(f"업로드 모듈을 불러오는 중 오류: {e}")
        return 22

    upload_argv = ['upload_integrated_to_mongo.py', '--input', integrated_out]
    rc = call_module_main(upload_mod, upload_argv)
    if rc != 0:
        print(f"업로드 단계 실패(종료코드 {rc})")
        return rc

    print("파이프라인 전체 완료: 수집 -> 전처리 -> 업로드가 모두 정상적으로 실행되었습니다.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
