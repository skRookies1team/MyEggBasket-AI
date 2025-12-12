import sys
import os
import logging
import re
import pandas as pd
from datetime import datetime

# 1. 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

# 모듈 가져오기
from ai_pipeline.nlp.pdf_parser import extract_text_from_pdf

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logging.getLogger("pdfminer").setLevel(logging.ERROR) 

# 경로 설정
REPORT_DIR = os.path.join(project_root, "data", "reports")
BASE_CSV_PATH = os.path.join(project_root, "data", "final_dataset_with_gcn.csv") # 기존 데이터셋
OUTPUT_CSV = os.path.join(project_root, "data", "final_dataset_with_reports.csv") # 최종 저장 파일

def preprocess_text(text):
    """텍스트 전처리"""
    if not text: return ""
    return re.sub(r'\s+', ' ', text).strip()

def extract_investment_info(text):
    """본문에서 투자의견/목표가 관련 핵심 문장 추출"""
    if not text: return ""
    lines = text.split('\n')
    if len(lines) < 5: lines = text.split('. ')

    keywords = [
        r"투자의견", r"목표주가", r"목표가", r"Target Price", r"TP",
        r"매수", r"Buy", r"매도", r"Sell", r"중립", r"Hold", r"Outperform",
        r"상향", r"하향", r"유지", r"전망", r"기대", r"실적"
    ]
    
    extracted_lines = []
    search_limit = min(len(lines), 50) 
    
    for i in range(search_limit):
        line = lines[i].strip()
        if len(line) < 5: continue 
        for key in keywords:
            if re.search(key, line, re.IGNORECASE):
                clean_line = re.sub(r'\s+', ' ', line).strip()
                extracted_lines.append(clean_line)
                break
    
    unique_lines = list(dict.fromkeys(extracted_lines))
    return " | ".join(unique_lines[:5])

def run_merge_and_save_test(limit_count=5):
    print("\n" + "="*60)
    print(f"🧪 [테스트] 리포트 {limit_count}개만 분석하여 병합 시작")
    print("="*60)
    
    if not os.path.exists(REPORT_DIR):
        print(f" 리포트 폴더가 없습니다: {REPORT_DIR}")
        return

    # 파일 목록 가져오기
    all_files = [f for f in os.listdir(REPORT_DIR) if f.endswith('.pdf')]
    
    if not all_files:
        print("❌ 저장된 PDF 파일이 없습니다.")
        return

    # [테스트용] 앞에서부터 지정한 개수만큼만 자름
    target_files = all_files[:limit_count]
    print(f"📄 전체 {len(all_files)}개 중 상위 {len(target_files)}개만 처리합니다.\n")

    report_data_list = []

    for idx, filename in enumerate(target_files):
        try:
            # 파일명 파싱
            name_body = filename.replace('.pdf', '')
            parts = name_body.split('_')
            
            if len(parts) >= 2:
                stock_code = parts[0]
                stock_name = parts[1]
                
                # 날짜 처리
                report_date = datetime.now().strftime("%Y-%m-%d")
                if len(parts) >= 3 and len(parts[2]) == 6:
                     report_date = f"20{parts[2][:2]}-{parts[2][2:4]}-{parts[2][4:]}"
                
                # 제목 처리
                report_title = "_".join(parts[3:]) if len(parts) > 3 else "제목없음"
            else:
                continue

        except Exception:
            continue

        print(f"[{idx+1}/{len(target_files)}] 분석 중: {stock_name} ({stock_code})...", end="\r")

        # 텍스트 추출 및 전처리
        path = os.path.join(REPORT_DIR, filename)
        raw_text = extract_text_from_pdf(path)
        
        if not raw_text or len(raw_text) < 50:
            continue

        clean_content = preprocess_text(raw_text)
        investment_info = extract_investment_info(raw_text)

        report_data_list.append({
            "stock_code": stock_code,     
            "stock_name": stock_name,
            "report_date": report_date,
            "report_title": report_title,
            "investment_info": investment_info, 
            "full_content": clean_content       
        })

    print(f"\n\n✅ 리포트 분석 완료: {len(report_data_list)}건")
    
    # 2. DataFrame 변환
    df_reports = pd.DataFrame(report_data_list)

    # 3. 기존 데이터셋 로드 및 병합
    if os.path.exists(BASE_CSV_PATH):
        print(f"📂 기존 데이터셋 로드 중: {BASE_CSV_PATH}")
        try:
            df_base = pd.read_csv(BASE_CSV_PATH)
            
            # stock_code 포맷 통일 (005930)
            if 'stock_code' in df_base.columns:
                df_base['stock_code'] = df_base['stock_code'].apply(lambda x: str(x).zfill(6))
            
            print("🔄 데이터 병합 중...")
            # 전체 데이터 살리기 (outer join)
            df_merged = pd.merge(df_base, df_reports, on='stock_code', how='outer')
            
            # 컬럼 정리
            if 'stock_name_x' in df_merged.columns:
                df_merged['stock_name'] = df_merged['stock_name_x'].fillna(df_merged['stock_name_y'])
                df_merged.drop(columns=['stock_name_x', 'stock_name_y'], inplace=True)

        except Exception as e:
            print(f"⚠️ 병합 실패 (리포트만 저장): {e}")
            df_merged = df_reports
    else:
        print("⚠️ 기존 데이터셋이 없어 리포트 데이터만 저장합니다.")
        df_merged = df_reports

    # 4. 저장
    df_merged.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n🎉 저장 완료! 파일 경로: {OUTPUT_CSV}")

if __name__ == "__main__":
    # 테스트로 5개만 실행 (원하면 숫자 변경)
    run_merge_and_save_test(limit_count=5)