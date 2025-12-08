import pandas as pd
import os
import sys

# 1. 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

# CSV 파일 경로
CSV_FILE_PATH = os.path.join(project_root, "data", "report_analysis_results.csv")

def check_csv():
    print("\n" + "="*60)
    print("📂 [데이터 확인] 저장된 CSV 파일 조회")
    print("="*60)

    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ 파일이 없습니다: {CSV_FILE_PATH}")
        print("   -> 먼저 'report_analysis_runner.py'를 실행해서 데이터를 수집하세요.")
        return

    try:
        # CSV 읽기
        df = pd.read_csv(CSV_FILE_PATH)
        
        # 1. 전체 개수
        print(f"📊 총 저장된 데이터 개수: {len(df)}건")
        print("-" * 60)

        if len(df) == 0:
            print("⚠️ 데이터가 비어있습니다.")
            return

        # 2. 데이터 미리보기 (최신 5개)
        # 보고 싶은 컬럼만 선택해서 출력
        cols = ['종목명', '종목코드', '핵심요약(투자의견)', '수집일시']
        
        # 만약 컬럼이 없을 수도 있으니 안전하게 처리
        available_cols = [c for c in cols if c in df.columns]
        
        print("[최신 저장된 5개 목록]")
        print(df[available_cols].head(5).to_string(index=False))
        
        print("-" * 60)
        
        # 3. 특정 종목의 투자 의견 자세히 보기 (첫 번째 데이터)
        first_row = df.iloc[0]
        print(f"\n🔍 [{first_row['종목명']}] 상세 내용 예시:")
        print(f"   - 투자의견 요약: {first_row.get('핵심요약(투자의견)', '없음')}")
        # 본문은 너무 기니까 앞부분만 출력
        content_preview = str(first_row.get('본문전체', ''))[:100].replace('\n', ' ')
        print(f"   - 본문 미리보기: {content_preview}...")

    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")

if __name__ == "__main__":
    check_csv()