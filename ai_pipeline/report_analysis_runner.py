import sys
import os
import logging
import re
from datetime import datetime

# 1. 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

# 모듈 가져오기
from ai_pipeline.nlp.pdf_parser import extract_text_from_pdf
from ai_pipeline.db.mongo_uploader import MongoUploader

# 로깅 설정
logging.basicConfig(level=logging.INFO)
REPORT_DIR = os.path.join(project_root, "data", "reports")

def extract_investment_info(text):
    """본문에서 매수/매도/목표가 관련 핵심 문장 추출"""
    if not text: return ""
    lines = text.split('\n')
    if len(lines) < 5: lines = text.split('. ')

    keywords = [
        r"투자의견", r"목표주가", r"목표가", r"Target Price", r"TP",
        r"매수", r"Buy", r"매도", r"Sell", r"중립", r"Hold", r"Outperform",
        r"상향", r"하향", r"유지", r"전망", r"기대"
    ]
    
    extracted_lines = []
    search_limit = min(len(lines), 50) 
    
    for i in range(search_limit):
        line = lines[i].strip()
        if len(line) < 5: continue 
        for key in keywords:
            if re.search(key, line, re.IGNORECASE):
                # 불필요한 공백 제거
                clean_line = re.sub(r'\s+', ' ', line).strip()
                extracted_lines.append(clean_line)
                break
    
    unique_lines = list(dict.fromkeys(extracted_lines))
    return " | ".join(unique_lines[:5])

def run_extraction_and_save():
    print("\n" + "="*60)
    print(" [ETL] 리포트 텍스트 추출 및 MongoDB 저장 (날짜/매수매도 포함)")
    print("="*60)
    
    if not os.path.exists(REPORT_DIR):
        print(f" 리포트 폴더가 없습니다: {REPORT_DIR}")
        return

    files = [f for f in os.listdir(REPORT_DIR) if f.endswith('.pdf')]
    if not files:
        print(" 분석할 PDF 파일이 없습니다.")
        return

    uploader = MongoUploader()
    success_count = 0

    for idx, filename in enumerate(files):
        # 파일명 파싱: 005930_삼성전자_240521.pdf
        try:
            parts = filename.replace('.pdf', '').split('_')
            if len(parts) >= 3:
                stock_code = parts[0]
                stock_name = parts[1]
                # 날짜 변환 (240521 -> 2024-05-21)
                date_str = parts[2]
                if len(date_str) == 6: # YYMMDD
                    report_date = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:]}"
                else:
                    report_date = datetime.now().strftime("%Y-%m-%d")
            else:
                stock_code = "N/A"
                stock_name = filename
                report_date = datetime.now().strftime("%Y-%m-%d")
        except:
            stock_code = "N/A"
            stock_name = filename
            report_date = datetime.now().strftime("%Y-%m-%d")

        print(f"[{idx+1}/{len(files)}] 처리 중: {stock_name} ({report_date})...", end="\r")

        # 1. 텍스트 추출
        path = os.path.join(REPORT_DIR, filename)
        full_text = extract_text_from_pdf(path)
        
        if not full_text or len(full_text) < 50:
            continue

        # 2. 매수/매도 정보 추출
        investment_info_text = extract_investment_info(full_text)

        # 3. MongoDB 저장 데이터 생성 (요청하신 포맷)
        doc = {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "report_date": report_date,          # 리포트 작성일
            "investment_info": investment_info_text, # 매수/매도 의견 (본문 추출)
            "full_content": full_text,           # 전체 본문
            "crawled_at": datetime.now()         # 수집 시점
        }

        uploader.save_report(doc)
        success_count += 1

    uploader.close()
    print(f"\n\n 작업 완료! 총 {success_count}개의 리포트가 MongoDB에 저장되었습니다.")

if __name__ == "__main__":
    run_extraction_and_save()