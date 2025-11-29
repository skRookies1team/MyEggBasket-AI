import pandas as pd
import os
import re

def map_value_chain_codes():
    print("🔄 밸류체인 종목코드 매핑 시작...")

    # 1. 파일 경로 설정 (스마트 탐색)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "../../"))
    
    search_dirs = [
        os.path.join(project_root, "data"),
        project_root,
        os.getcwd()
    ]
    
    stock_file_name = "data_2218_20251128.csv"
    vc_file_name = "value_chain_data.csv"
    
    found_dir = None
    
    print(f"🔍 파일 탐색 중...")
    for directory in search_dirs:
        if os.path.exists(directory):
            s_path = os.path.join(directory, stock_file_name)
            v_path = os.path.join(directory, vc_file_name)
            
            if os.path.exists(s_path) and os.path.exists(v_path):
                found_dir = directory
                print(f"✅ 파일을 찾았습니다: {found_dir}")
                break
    
    if found_dir is None:
        print("\n❌ 오류: 파일을 찾을 수 없습니다.")
        return

    stock_list_path = os.path.join(found_dir, stock_file_name)
    value_chain_path = os.path.join(found_dir, vc_file_name)
    output_path = os.path.join(found_dir, "value_chain_result.csv")

    # [핵심 수정] 인코딩을 안전하게 처리하는 함수
    def read_csv_safe(path):
        try:
            # 1순위: utf-8 시도
            return pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # 2순위: cp949 (윈도우/엑셀 한글) 시도
                return pd.read_csv(path, encoding='cp949')
            except UnicodeDecodeError:
                # 3순위: euc-kr 시도
                return pd.read_csv(path, encoding='euc-kr')

    # 2. 데이터 로드
    try:
        df_stock = read_csv_safe(stock_list_path)
        df_vc = read_csv_safe(value_chain_path)
        
        print(f"📊 데이터 로드 성공")
        print(f"   - 종목 데이터: {len(df_stock)}개")
        print(f"   - 밸류체인 데이터: {len(df_vc)}행")

    except Exception as e:
        print(f"❌ 파일 읽기 중 에러 발생: {e}")
        return

    # 3. 매핑 사전(Dictionary) 만들기
    name_to_code = {}

    # 컬럼명에 공백이 있을 수 있으므로 정리 (strip)
    df_stock.columns = df_stock.columns.str.strip()

    for _, row in df_stock.iterrows():
        # 단축코드를 6자리 문자열로 변환 (앞에 0 채우기)
        code = str(row['단축코드']).zfill(6)
        
        # 1) 한글 종목약명 (예: 삼성전자)
        name_short = str(row['한글 종목약명']).strip()
        name_to_code[name_short] = code
        
        # 2) 한글 종목명 (예: 삼성전자보통주)
        name_full = str(row['한글 종목명']).strip()
        name_to_code[name_full] = code
        
        # 3) (주) 제거 버전
        name_clean = name_full.replace("(주)", "").strip()
        name_to_code[name_clean] = code

    # 4. 매핑 함수 정의
    def add_code_to_names(companies_str):
        if pd.isna(companies_str):
            return ""
        
        # 쉼표(,)로 구분된 기업들을 리스트로 분리
        # 엑셀 데이터 특성상 줄바꿈(\n)이나 탭 등이 섞일 수 있어 정리 필요
        clean_str = str(companies_str).replace('\n', ',').replace('\r', '')
        company_list = [c.strip() for c in clean_str.split(',')]
        
        new_list = []
        for company in company_list:
            if not company: continue
            
            # 괄호나 특수문자가 섞인 경우 정리 (예: "삼성전자(주)")
            company_clean = company.replace("(주)", "").strip()

            # 사전에 있는지 확인
            if company in name_to_code:
                code = name_to_code[company]
                new_list.append(f"{company} ({code})")
            elif company_clean in name_to_code:
                code = name_to_code[company_clean]
                new_list.append(f"{company} ({code})")
            else:
                # 못 찾으면 이름만 유지
                new_list.append(company)
                
        return ", ".join(new_list)

    # 5. 적용 및 저장
    # 컬럼명 공백 정리
    df_vc.columns = df_vc.columns.str.strip()
    
    if '기업' in df_vc.columns:
        df_vc['기업_코드포함'] = df_vc['기업'].apply(add_code_to_names)
        
        # 저장할 때는 utf-8-sig (엑셀에서 한글 안 깨지게)
        df_vc.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 변환 완료! 파일이 저장되었습니다.")
        print(f"📂 저장 경로: {output_path}")
        
        print("\n[결과 미리보기]")
        # 출력용 컬럼 선택
        cols = df_vc.columns.tolist()
        target_cols = cols[:2] + ['기업_코드포함'] # 앞의 2개 컬럼 + 결과 컬럼
        print(df_vc[target_cols].head(3))
        
    else:
        print(f"❌ 밸류체인 파일에 '기업' 컬럼이 없습니다. 현재 컬럼: {df_vc.columns.tolist()}")

if __name__ == "__main__":
    map_value_chain_codes()