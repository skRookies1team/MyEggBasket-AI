"""
공시 데이터 자동 수집 → 전처리 → MongoDB 저장 파이프라인
매일 오전 9시 자동 실행 (전날 데이터만 수집)
"""
import os
import sys
import time
import re
import requests
import pandas as pd
import numpy as np
import signal  # ✅ 추가
from io import BytesIO
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.config import settings
try:
    from ai_pipeline.analysis.disclosure_feature_engineering import DisclosureFeatureEngineer
except Exception:
    DisclosureFeatureEngineer = None
try:
    # optional dotenv support for local .env files
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ✅ 전역 종료 플래그
interrupted = False

def signal_handler(signum, frame):
    """Ctrl+C 신호 처리"""
    global interrupted
    print("\n\n⚠️  종료 신호 감지! 현재 작업을 중단합니다...")
    interrupted = True

# ✅ 신호 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class DisclosureAutoCollector:
    """공시 데이터 자동 수집 및 MongoDB 저장"""
    
    def __init__(self):
        # API 키
        self.api_key = os.getenv("DART_API_KEY")
        if not self.api_key:
            raise ValueError("⚠ DART_API_KEY가 환경변수에 설정되지 않았습니다.")
        
        # MongoDB 연결
        # 환경변수 우선순위: MONGO_URI -> MONGODB_URI
        self.mongo_uri = os.getenv('MONGO_URI') or os.getenv('MONGODB_URI')
        if not self.mongo_uri:
            raise ValueError("⚠ MONGO_URI 또는 MONGODB_URI가 환경변수에 설정되어 있지 않습니다. .env 또는 시스템 환경변수에 설정해주세요.")
        self.client = None
        self.db = None
        self.collection = None
        
        # OpenDART API 기본 URL
        self.base_url = "https://opendart.fss.or.kr/api"
        
        # 임시 파일 경로
        self.temp_csv_path = "temp_integrated_financial_data.csv"
        # 디버그: 환경변수 로드 상태 (민감 정보는 출력하지 않음)
        dart_present = bool(os.getenv('DART_API_KEY'))
        mongo_present = bool(self.mongo_uri)
        print(f"환경: DART_API_KEY 설정됨={dart_present}, Mongo URI 설정됨={mongo_present}")
    
    def _init_mongodb(self):
        """MongoDB 연결 초기화"""
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.db = self.client['stockdb']
            self.collection = self.db['disclosure_features']
            
            # 인덱스 생성
            self.collection.create_index(
                [("stock_code", ASCENDING), ("bsns_year", ASCENDING)],
                unique=True
            )
            
            print("✅ MongoDB 연결 성공")
            return True
        except Exception as e:
            print(f"❌ MongoDB 연결 실패: {e}")
            return False
    
    def _request(self, endpoint, params):
        """OpenDART API 요청"""
        global interrupted
        if interrupted:
            return None
            
        url = f"{self.base_url}/{endpoint}"
        params['crtfc_key'] = self.api_key
        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
            if data.get('status') == '000':
                return data.get('list', [])
            return None
        except Exception as e:
            if not interrupted:
                print(f"⚠ API 요청 실패 ({endpoint}): {e}")
            return None
    
    def get_corp_code(self):
        """고유번호 가져오기"""
        url = f"{self.base_url}/corpCode.xml"
        params = {'crtfc_key': self.api_key}
        try:
            resp = requests.get(url, params=params, timeout=30)
            with zipfile.ZipFile(BytesIO(resp.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
            
            data = []
            for child in root.findall('list'):
                data.append({
                    'corp_code': child.find('corp_code').text,
                    'stock_code': child.find('stock_code').text.strip(),
                })
            return pd.DataFrame(data)
        except Exception as e:
            print(f"❌ 고유번호 가져오기 실패: {e}")
            return pd.DataFrame()
    
    def get_disclosure_list(self, bgn_de, end_de, corp_cls='Y'):
        """공시 검색 (정기보고서만)"""
        global interrupted
        
        endpoint = "list.json"
        results = []
        page_no = 1
        
        while not interrupted:
            params = {
                'bgn_de': bgn_de, 
                'end_de': end_de, 
                'corp_cls': corp_cls,
                'page_no': page_no, 
                'page_count': 100, 
                'pblntf_ty': 'A'
            }
            data = self._request(endpoint, params)
            if not data: 
                break
            results.extend(data)
            if len(data) < 100: 
                break
            page_no += 1
            time.sleep(0.1)
        
        return results
    
    def process_finance(self, data):
        """재무제표에서 핵심 계정만 추출"""
        if not data: 
            return {}
        
        df = pd.DataFrame(data)
        target_df = df[df['fs_div'] == 'CFS']
        if target_df.empty:
            target_df = df[df['fs_div'] == 'OFS']
        
        result = {}
        mappings = {
            '매출액': 'fin_revenue',
            '수익(매출액)': 'fin_revenue',
            '영업이익': 'fin_op_income',
            '영업이익(손실)': 'fin_op_income',
            '당기순이익': 'fin_net_income',
            '당기순이익(손실)': 'fin_net_income',
            '자산총계': 'fin_total_assets',
            '부채총계': 'fin_total_liabilities',
            '자본총계': 'fin_total_equity'
        }
        
        for _, row in target_df.iterrows():
            acct = row['account_nm'].replace(" ", "")
            amt = row['thstrm_amount']
            
            try:
                val = float(amt.replace(',', ''))
            except:
                val = 0
            
            for key, col_name in mappings.items():
                if key in acct:
                    result[col_name] = val
        
        return result
    
    def process_dividend(self, data):
        """배당금 (보통주 현금배당 기준)"""
        if not data: 
            return {}
        for item in data:
            if item.get('se') == '주당 현금배당금(원)' and item.get('stock_knd') == '보통주':
                return {'div_dps_common': item.get('thstrm', 0)}
        return {}
    
    def process_employees(self, data):
        """직원 현황 (합계)"""
        if not data: 
            return {}
        
        for item in data:
            if '합계' in item.get('se', '') or '남여 공통' in item.get('se', ''):
                return {
                    'emp_total_count': item.get('sm'),
                    'emp_avg_salary': item.get('jan_avrg_salary')
                }
        
        if data:
            return {'emp_total_count': data[0].get('sm')}
        return {}
    
    def process_simple_sum(self, data, field_name, out_col):
        """단순 잔액 합계"""
        if not data: 
            return {}
        total = 0.0
        for item in data:
            val_str = item.get(field_name, '0')
            try:
                total += float(val_str.replace(',', ''))
            except:
                pass
        return {out_col: total} if total > 0 else {}
    
    def collect_disclosures(self, bgn_de, end_de, limit_companies=None):
        """
        공시 데이터 수집 (특정 기간)
        
        Args:
            bgn_de: 시작일 (YYYYMMDD)
            end_de: 종료일 (YYYYMMDD)
            limit_companies: 테스트용 제한 개수
        """
        global interrupted
        
        print("\n" + "="*60)
        print("📊 공시 데이터 수집 시작")
        print("="*60)
        
        # 1. 종목코드 매핑
        corp_df = self.get_corp_code()
        if corp_df.empty or interrupted:
            print("❌ 종목코드 가져오기 실패")
            return None
        
        # 2. 공시 목록 조회
        print(f"🔍 기간: {bgn_de} ~ {end_de}")
        
        reports_kospi = self.get_disclosure_list(bgn_de, end_de, 'Y')
        reports_kosdaq = self.get_disclosure_list(bgn_de, end_de, 'K')
        
        if interrupted:
            print("⚠️  중단됨")
            return None
        
        all_reports = reports_kospi + reports_kosdaq
        print(f"✅ 총 {len(all_reports)}건의 정기보고서 발견")
        
        if not all_reports:
            print("⚠ 수집할 공시가 없습니다.")
            return None
        
        # 3. 상세 데이터 수집
        final_data_list = []
        count = 0
        
        for report in all_reports:
            if interrupted:  # ✅ 중단 체크
                print(f"\n⚠️  수집 중단됨 (처리: {count}건)")
                break
                
            if limit_companies and count >= limit_companies:
                print(f"🛑 테스트 제한({limit_companies}개) 도달")
                break
            
            # 보고서 정보 파싱
            report_nm = report['report_nm']
            year_match = re.search(r'\((\d{4})', report_nm)
            if not year_match: 
                continue
            
            bsns_year = year_match.group(1)
            
            # 보고서 코드 결정
            if ".12)" in report_nm or "사업보고서" in report_nm: 
                reprt_code = "11011"
            elif ".03)" in report_nm or "1분기" in report_nm: 
                reprt_code = "11013"
            elif ".06)" in report_nm or "반기" in report_nm: 
                reprt_code = "11012"
            elif ".09)" in report_nm or "3분기" in report_nm: 
                reprt_code = "11014"
            else: 
                continue
            
            if count % 50 == 0:
                print(f"[{count+1}] 처리 중...")
            
            # 기본 데이터 구성
            row_data = {
                'corp_code': report['corp_code'],
                'corp_name': report['corp_name'],
                'report_name': report_nm,
                'rcept_no': report['rcept_no'],
                'rcept_dt': report['rcept_dt'],
                'bsns_year': bsns_year,
                'reprt_code': reprt_code
            }
            
            params = {
                'corp_code': report['corp_code'], 
                'bsns_year': bsns_year, 
                'reprt_code': reprt_code
            }
            
            # API 호출 및 데이터 병합
            fin_data = self._request("fnlttSinglAcnt.json", params.copy())
            if interrupted: break
            row_data.update(self.process_finance(fin_data))
            
            if reprt_code == '11011':
                div_data = self._request("alotMatter.json", params.copy())
                if interrupted: break
                row_data.update(self.process_dividend(div_data))
            
            emp_data = self._request("empSttus.json", params.copy())
            if interrupted: break
            row_data.update(self.process_employees(emp_data))
            
            stock_data = self._request("tesstkAcqsDspsSttus.json", params.copy())
            if interrupted: break
            if stock_data:
                row_data['treasury_stock_event'] = 'Y'
            
            inc_data = self._request("irdsSttus.json", params.copy())
            if interrupted: break
            if inc_data:
                row_data['capital_change_count'] = len(inc_data)
            
            final_data_list.append(row_data)
            count += 1
        
        if interrupted:
            print("⚠️  수집 프로세스가 중단되었습니다")
            return None
        
        # 4. DataFrame 생성 및 종목코드 병합
        if not final_data_list:
            print("❌ 수집된 데이터가 없습니다.")
            return None
        
        df = pd.DataFrame(final_data_list)
        if not corp_df.empty:
            df = pd.merge(df, corp_df, on='corp_code', how='left')
        
        print(f"✅ {len(df)}개 기업 데이터 수집 완료")
        
        # 5. CSV 저장 (전처리용)
        df.to_csv(self.temp_csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 임시 CSV 저장: {self.temp_csv_path}")
        
        return df
    
    def preprocess_and_save_to_mongodb(self):
        """전처리 후 MongoDB에 저장"""
        global interrupted
        
        print("\n" + "="*60)
        print("🔧 데이터 전처리 및 MongoDB 저장")
        print("="*60)
        
        if interrupted:
            return False
        
        if not os.path.exists(self.temp_csv_path):
            print("❌ 임시 CSV 파일이 없습니다.")
            return False
        
        try:
            # 1. DisclosureFeatureEngineer로 전처리 시도
            final_df = None
            feature_cols = []

            try:
                # 우선 기존의 DisclosureFeatureEngineer가 있으면 사용
                from ai_pipeline.analysis.disclosure_feature_engineering import DisclosureFeatureEngineer
            except Exception:
                DisclosureFeatureEngineer = None

            if DisclosureFeatureEngineer:
                engineer = DisclosureFeatureEngineer(self.temp_csv_path)
                final_df, feature_cols = engineer.prepare_ml_features()
            else:
                # 대체: 로컬 전처리 스크립트(preprocess_disclosures_csv.py)를 사용
                try:
                    from ai_pipeline.disclosure_pipeline import preprocess_disclosures_csv as preproc
                except Exception:
                    # 파일 경로에서 직접 로드
                    import importlib.util
                    module_path = os.path.join(os.path.dirname(__file__), 'preprocess_disclosures_csv.py')
                    spec = importlib.util.spec_from_file_location('preprocess_disclosures_csv', module_path)
                    preproc = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(preproc)

                df = pd.read_csv(self.temp_csv_path, encoding='utf-8-sig')
                df = preproc.normalize_stock_code(df)
                df = preproc.ensure_bsns_year(df)
                df = preproc.rename_financial_columns(df)
                df = preproc.derive_reprt_code(df)

                # 중복 제거: stock_code + bsns_year
                if 'stock_code' in df.columns:
                    sort_key = 'rcept_dt' if 'rcept_dt' in df.columns else None
                    if sort_key:
                        df = df.sort_values(sort_key, ascending=False)
                    df = df.drop_duplicates(subset=['stock_code', 'bsns_year'], keep='first')

                final_df = df
                # 피처 컬럼 추정: 메타컬럼 제외
                exclude = set(['stock_code','bsns_year','corp_code','corp_name','report_name','rcept_no','rcept_dt','reprt_code','flr_nm','rm'])
                feature_cols = [c for c in final_df.columns if c not in exclude]

            if interrupted or final_df is None or final_df.empty:
                print("❌ 피처 생성 실패 또는 중단됨")
                return False

            print(f"✅ 피처 생성 완료: {len(feature_cols)}개 피처, {len(final_df)}개 종목")
            
            # 2. MongoDB 연결
            if not self._init_mongodb():
                return False
            
            # 3. MongoDB에 저장
            inserted_count = 0
            updated_count = 0
            
            for idx, row in final_df.iterrows():
                if interrupted:  # ✅ 중단 체크
                    print(f"\n⚠️  저장 중단됨 (처리: {idx}건)")
                    break
                
                doc = row.to_dict()
                
                # NaN을 None으로 변환
                for key, value in doc.items():
                    if pd.isna(value):
                        doc[key] = None
                
                try:
                    filter_query = {
                        'stock_code': doc['stock_code'],
                        'bsns_year': doc['bsns_year']
                    }
                    
                    result = self.collection.update_one(
                        filter_query,
                        {'$set': doc},
                        upsert=True
                    )
                    
                    if result.upserted_id:
                        inserted_count += 1
                    else:
                        updated_count += 1
                    
                except Exception as e:
                    print(f"⚠ 저장 실패 ({doc.get('stock_code')}): {e}")
            
            if not interrupted:
                print(f"\n✅ MongoDB 저장 완료!")
                print(f"   신규 삽입: {inserted_count}개")
                print(f"   업데이트: {updated_count}개")
            
            # 4. 임시 파일 삭제
            if os.path.exists(self.temp_csv_path):
                os.remove(self.temp_csv_path)
                print(f"🗑️ 임시 파일 삭제: {self.temp_csv_path}")
            
            return not interrupted
            
        except Exception as e:
            if not interrupted:
                print(f"❌ 전처리 및 저장 실패: {e}")
                import traceback
                traceback.print_exc()
            return False
    
    def run_daily_collection(self, target_date=None):
        """
        일일 자동 수집 실행 (전날 데이터만 수집)
        
        Args:
            target_date: 수집할 날짜 (None이면 어제)
        """
        global interrupted
        
        print("\n" + "="*70)
        print(f"🤖 공시 데이터 자동 수집 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # 어제 날짜 계산
        if target_date is None:
            target_date = datetime.now() - timedelta(days=1)
        
        # 어제 하루만 수집
        bgn_de = target_date.strftime("%Y%m%d")
        end_de = target_date.strftime("%Y%m%d")
        
        print(f"🔍 수집 대상: {bgn_de} (전날 데이터)")
        
        # 1. 공시 수집
        df = self.collect_disclosures(bgn_de, end_de)
        
        if interrupted:
            print("⚠️  수집 작업이 사용자에 의해 중단되었습니다")
            return False
        
        if df is None or df.empty:
            print("\n" + "="*70)
            print(f"💤 {bgn_de}에는 정기보고서 공시가 없습니다 (정상)")
            print("   → 다음 수집 시각: 내일 오전 9시")
            print("="*70)
            return True  # ✅ 데이터 없음도 정상 처리
        
        # 2. 전처리 및 MongoDB 저장
        success = self.preprocess_and_save_to_mongodb()
        
        print("\n" + "="*70)
        if success and not interrupted:
            print(f"✅ 공시 데이터 자동 수집 완료! ({bgn_de})")
        elif interrupted:
            print("⚠️  작업이 중단되었습니다")
        else:
            print("❌ 공시 데이터 자동 수집 실패")
        print("="*70)
        
        return success
    
    def close(self):
        """MongoDB 연결 종료"""
        if self.client:
            self.client.close()
            print("✅ MongoDB 연결 종료")


# 테스트 실행
if __name__ == "__main__":
    collector = DisclosureAutoCollector()
    
    try:
        # 어제 데이터 수집
        collector.run_daily_collection()
    except KeyboardInterrupt:
        print("\n⚠️  강제 종료됨")
    finally:
        collector.close()