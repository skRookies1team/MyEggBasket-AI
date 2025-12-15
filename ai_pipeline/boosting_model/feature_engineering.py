import pandas as pd
import numpy as np
import torch
import json
import os
import sys
import re
import glob
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 필요한 클래스 import (없으면 무시)
try:
    from ai_pipeline.boosting_model.realtime_feature_loader import RealtimeFeatureLoader
    from ai_pipeline.boosting_model.feature_expander import FeatureExpander
except ImportError:
    pass

# GAE 모델 로드
try:
    from ai_pipeline.gcn_model.model import get_gae_model
except ImportError:
    print(" GCN 모델 파일을 찾을 수 없습니다. (ai_pipeline/gcn_model/model.py 확인 필요)")
    get_gae_model = None

# =========================================================
# ✅ GCN 로더 클래스
# =========================================================
class GCNFeatureExtractor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 데이터 파일(.pt) 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pt_path = os.path.abspath(os.path.join(current_dir, "../../finance_graph_data.pt"))
        
        if not os.path.exists(pt_path):
            print(f" [GCN] 데이터 파일이 없습니다: {pt_path}")
            self.data = None
            return

        try:
            # ✅ [핵심 수정] PyTorch 2.6+ 보안 경고 우회 (weights_only=False 명시)
            # 그래프 데이터 구조체(Data)는 단순 가중치가 아니므로 False여야 함
            self.data = torch.load(pt_path, map_location=self.device, weights_only=False)
        except TypeError:
            # 구버전 PyTorch 호환용
            self.data = torch.load(pt_path, map_location=self.device)
        except Exception as e:
            print(f" [GCN] 데이터 로드 실패: {e}")
            self.data = None
            return

        # 2. 모델 초기화 (NewsStockGCN -> get_gae_model 로 변경)
        # 데이터의 피처 수(x.shape[1])를 입력 차원으로 설정
        if self.data is not None and get_gae_model is not None:
            num_features = self.data.x.shape[1]
            # 출력 차원은 학습시 설정한 값 (보통 16 또는 64)
            self.model = get_gae_model(in_channels=num_features, out_channels=16).to(self.device)
        else:
            self.model = None
            return
        
        # 3. 모델 가중치 로드
        if model_path is None:
            # 경로 자동 탐색
            model_path = os.path.abspath(os.path.join(current_dir, "../../best_gcn_model.pth"))
            
        if os.path.exists(model_path):
            try:
                # strict=False로 형상이 약간 달라도 로드 시도
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True), strict=False)
                self.model.eval()
            except Exception as e:
                print(f" 모델 가중치 로드 실패 (초기화 상태 사용): {e}")
                self.model.eval()
        else:
            print(" 학습된 모델 파일이 없습니다. (초기화 상태 사용)")
            self.model.eval()

    def _load_json_mapping(self):
        """JSON 매핑 파일 로드"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.abspath(os.path.join(current_dir, "../../ai_pipeline/graph_build/node_mapping.json"))
        
        if not os.path.exists(target_path):
             target_path = os.path.abspath(os.path.join(current_dir, "../../node_mapping.json"))

        if os.path.exists(target_path):
            try:
                with open(target_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f) 
                    return {int(v): k for k, v in mapping.items()}
            except Exception:
                pass
        return None

    def get_embeddings(self):
        if self.data is None or self.model is None: return {}

        with torch.no_grad():
            try:
                embeddings = self.model.encode(self.data.x, self.data.edge_index)
            except AttributeError:
                embeddings = self.model(self.data.x, self.data.edge_index)

        emb_np = embeddings.cpu().numpy()

        mapping = {}

        # [수정] 인덱스(v)를 int로 변환하여 저장
        if hasattr(self.data, 'stock_to_idx'):
            idx_to_stock = {int(v): k for k, v in self.data.stock_to_idx.items()}
        else:
            idx_to_stock = self._load_json_mapping()

        if idx_to_stock:
            for idx, vector in enumerate(emb_np):
                # idx는 정수이므로 이제 매칭됨
                if idx in idx_to_stock:
                    code = idx_to_stock[idx]
                    mapping[code] = vector
        else:
            print("  매핑 정보 없음: GCN 피처를 사용할 수 없습니다.")

        return mapping

    def add_gcn_features(self, df, code_col='code'):
        target_col = code_col
        # 컬럼 이름 정규화
        if 'stck_shrn_iscd' in df.columns: target_col = 'stck_shrn_iscd'
        elif 'code' in df.columns: target_col = 'code'
        elif 'stock_code' in df.columns: target_col = 'stock_code'
            
        emb_dict = self.get_embeddings()
        if not emb_dict: 
            return df

        # Dict -> DataFrame
        emb_df = pd.DataFrame.from_dict(emb_dict, orient='index')
        emb_df.columns = [f'gcn_emb_{i}' for i in range(emb_df.shape[1])]
        emb_df.index.name = target_col
        emb_df = emb_df.reset_index()
        
        # 타입 통일 (문자열)
        df[target_col] = df[target_col].astype(str).str.strip().str.zfill(6)
        emb_df[target_col] = emb_df[target_col].astype(str).str.strip().str.zfill(6)
        
        merged_df = pd.merge(df, emb_df, on=target_col, how='left')
        
        gcn_cols = [c for c in merged_df.columns if c.startswith('gcn_')]
        merged_df[gcn_cols] = merged_df[gcn_cols].fillna(0)
        
        return merged_df


# =========================================================
# ✅ 메인 FeatureEngineer 클래스
# =========================================================
class FeatureEngineer:
    def __init__(self, data_dir=None, csv_path=None):
        self.data_dir = data_dir
        self.csv_path = csv_path 
        try:
            self.expander = FeatureExpander()
        except Exception:
            # ta 라이브러리 등 의존성 문제로 FeatureExpander가 로드되지 않을 수 있음
            # 이 경우 확장 기능 없이 기본 피처만 사용하도록 None으로 설정
            self.expander = None
        try:
            self.gcn_loader = GCNFeatureExtractor()
        except Exception as e:
            print(f" GCN 로더 초기화 실패: {e}")
            self.gcn_loader = None

        try:
            self.es = Elasticsearch("http://localhost:9200")
            if not self.es.ping(): self.es = None
        except:
            self.es = None
        # 통합된 공시 데이터 로드 시도
        self.disclosure_df = None
        try:
            disc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../disclosure_pipeline/data/integrated_financial_data.csv"))
            if os.path.exists(disc_path):
                ddf = pd.read_csv(disc_path, encoding='utf-8-sig')
                if 'stock_code' in ddf.columns:
                    ddf['stock_code'] = ddf['stock_code'].astype(str).str.strip().str.zfill(6)
                    # 최신 연도 우선(있다면)
                    if 'bsns_year' in ddf.columns:
                        try:
                            ddf['bsns_year'] = pd.to_numeric(ddf['bsns_year'], errors='coerce').fillna(0).astype(int)
                            ddf = ddf.sort_values('bsns_year', ascending=False).drop_duplicates('stock_code', keep='first')
                        except Exception:
                            pass
                    # 숫자형 컬럼만 선택하여 인덱스 설정
                    num_cols = ddf.select_dtypes(include=[np.number]).columns.tolist()
                    keep_cols = ['stock_code'] + [c for c in num_cols if c != 'stock_code']
                    if len(keep_cols) > 1:
                        # 접두사로 공시 컬럼을 구분합니다 (충돌 방지)
                        disc_df = ddf[keep_cols].set_index('stock_code')
                        # 숫자형 컬럼 이름에 접두사 추가 (stock_code 제외)
                        new_cols = {}
                        for c in disc_df.columns:
                            if c == 'stock_code':
                                new_cols[c] = c
                            else:
                                new_cols[c] = f"disc_{c}"
                        disc_df = disc_df.rename(columns=new_cols)
                        self.disclosure_df = disc_df
                        try:
                            print(f" 공시 데이터 로드됨: 종목 {self.disclosure_df.shape[0]}개, 컬럼 {self.disclosure_df.shape[1]}개")
                            print(f"    경로: {disc_path}")
                            print(f"    공시 피처 컬럼명 샘플: {list(self.disclosure_df.columns)[:20]}")
                            if len(self.disclosure_df) > 0:
                                print(f"    공시 데이터 샘플 (상위 3개):")
                                print(self.disclosure_df.head(3).to_string())
                        except Exception:
                            pass
        except Exception:
            self.disclosure_df = None
    
    def _get_date_from_filename(self, filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'(\d{8})', basename)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d")
            except: pass
        return None

    def merge_sentiment_scores(self, X, stock_codes, current_file_path):
        """
        [수정] 파일명 기준 단일 집계가 아니라, 데이터의 타임스탬프를 기준으로
        '일자별(Daily)' 감성 점수/변동성/트렌드를 매핑합니다.
        """
        if isinstance(stock_codes, pd.Series): stock_codes = stock_codes.tolist()
        unique_codes = list(set(stock_codes))

        # 1. 초기화
        X['sentiment_score'] = 0.0
        X['sentiment_volatility'] = 0.0
        X['sentiment_trend'] = 0.0

        if not self.es or 'timestamp' not in X.columns:
            return X

        # 2. 데이터의 기간 확인 (Min/Max Date)
        # 문자열이 아닌 datetime 객체여야 합니다.
        if not pd.api.types.is_datetime64_any_dtype(X['timestamp']):
            try:
                X['timestamp'] = pd.to_datetime(X['timestamp'])
            except:
                return X

        min_date = X['timestamp'].min()
        max_date = X['timestamp'].max()

        # 검색 범위 설정 (데이터 시작일 - 2일 ~ 종료일)
        start_dt = min_date - timedelta(days=2)
        end_dt = max_date + timedelta(days=1)  # 넉넉하게

        # 3. ES 집계 쿼리: 종목별 -> 일자별(Daily) -> 통계
        body = {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"published_date": {"gte": start_dt.isoformat(), "lte": end_dt.isoformat()}}},
                        {"terms": {"related_stocks.keyword": unique_codes}}  # 현재 파일의 종목만 필터링
                    ]
                }
            },
            "aggs": {
                "by_stock": {
                    "terms": {"field": "related_stocks.keyword", "size": 100},  # 종목별 버킷
                    "aggs": {
                        "by_day": {
                            "date_histogram": {
                                "field": "published_date",
                                "calendar_interval": "day",  # 일별 집계
                                "format": "yyyy-MM-dd"
                            },
                            "aggs": {
                                "avg_sent": {"avg": {"field": "sentiment_score"}},
                                "avg_vol": {"avg": {"field": "analysis_results.sentiment_volatility"}},
                                "avg_trend": {"avg": {"field": "analysis_results.sentiment_trend"}}
                            }
                        }
                    }
                }
            }
        }

        try:
            resp = self.es.search(index="news_articles", body=body)

            # 4. 조회 결과 딕셔너리로 변환 {(종목코드, 날짜): {features...}}
            sentiment_map = {}

            if 'aggregations' in resp and 'by_stock' in resp['aggregations']:
                for stock_bucket in resp['aggregations']['by_stock']['buckets']:
                    code = stock_bucket['key']

                    for date_bucket in stock_bucket['by_day']['buckets']:
                        date_str = date_bucket['key_as_string']  # "2025-12-05"

                        vals = {
                            'score': date_bucket['avg_sent']['value'],
                            'vol': date_bucket['avg_vol']['value'],
                            'trend': date_bucket['avg_trend']['value']
                        }
                        # None 체크
                        vals = {k: (v if v is not None else 0.0) for k, v in vals.items()}

                        sentiment_map[(code, date_str)] = vals

            # 5. DataFrame에 매핑
            # X에 임시 날짜 컬럼 생성 (문자열 매칭용)
            X['temp_date_key'] = X['timestamp'].dt.strftime('%Y-%m-%d')

            # 매핑 함수 정의
            def get_sent_values(row):
                # stock_code 컬럼명이 'stock_code' 혹은 'stck_shrn_iscd' 라고 가정
                code = str(row.get('stock_code', row.get('stck_shrn_iscd', ''))).zfill(6)
                date_key = row['temp_date_key']

                # (종목, 날짜)로 조회
                if (code, date_key) in sentiment_map:
                    data = sentiment_map[(code, date_key)]
                    return pd.Series([data['score'], data['vol'], data['trend']])
                else:
                    return pd.Series([0.0, 0.0, 0.0])

            # apply 적용 (속도를 위해 벡터화할 수도 있지만, 가독성을 위해 apply 사용)
            # 데이터 양이 많다면 merge 방식이 더 빠릅니다.

            # [성능 개선 버전: Merge 사용]
            # 맵을 DF로 변환
            map_data = []
            for (c, d), v in sentiment_map.items():
                map_data.append({
                    'join_code': c,
                    'join_date': d,
                    'sent_score_daily': v['score'],
                    'sent_vol_daily': v['vol'],
                    'sent_trend_daily': v['trend']
                })

            if map_data:
                sent_df = pd.DataFrame(map_data)

                # 원본 DF에 조인 키 생성
                # stock_code가 여러 컬럼명일 수 있으니 확인
                code_col_name = 'stck_shrn_iscd' if 'stck_shrn_iscd' in X.columns else 'stock_code'

                # 병합 전 타입 통일
                X['join_code'] = X[code_col_name].astype(str).str.zfill(6)
                X['join_date'] = X['temp_date_key']

                # Left Join
                X = pd.merge(X, sent_df, on=['join_code', 'join_date'], how='left')

                # NaN 채우기 및 컬럼명 정리
                X['sentiment_score'] = X['sent_score_daily'].fillna(0.0)
                X['sentiment_volatility'] = X['sent_vol_daily'].fillna(0.0)
                X['sentiment_trend'] = X['sent_trend_daily'].fillna(0.0)

                # 임시 컬럼 삭제
                X.drop(columns=['join_code', 'join_date', 'sent_score_daily', 'sent_vol_daily', 'sent_trend_daily'],
                       inplace=True, errors='ignore')

            X.drop(columns=['temp_date_key'], inplace=True, errors='ignore')

        except Exception as e:
            print(f" [ES Error] 일자별 감성 집계 실패: {e}")
            pass

        return X

    def _process_single_file(self, csv_file):
        print(f" 처리 중: {os.path.basename(csv_file)}")
        loader = RealtimeFeatureLoader(csv_file)
        try:
            load_result = loader.prepare_features()
            if len(load_result) == 3: X, y, stock_codes = load_result
            else: X, y = load_result; stock_codes = []
        except: return None, None, None
        
        if X is None or X.empty: return None, None, None
        
        # 병합을 위해 임시로 종목코드 추가
        temp_code_col = 'stck_shrn_iscd'
        X[temp_code_col] = stock_codes
        
        # 기술적 지표 추가
        if self.expander:
            X = self.expander.add_technical_indicators(X)
        
        # GCN 피처 추가
        if self.gcn_loader:
            X = self.gcn_loader.add_gcn_features(X, code_col=temp_code_col)
            
        # 감성 점수 추가
        X = self.merge_sentiment_scores(X, stock_codes, csv_file)
        X = X.fillna(0)
        
        return X, y, stock_codes
    
    def create_final_features(self):
        print("\n" + "="*60)
        print(f" 통합 데이터셋 생성 시작")
        print("="*60)
        
        csv_files = []
        if self.csv_path and os.path.exists(self.csv_path):
            csv_files = [self.csv_path]
        elif self.data_dir and os.path.isdir(self.data_dir):
            # ✅ [수정] 파일명 패턴 완화: 모든 csv 파일 대상
            all_csvs = glob.glob(os.path.join(self.data_dir, "*.csv"))
            # 필요한 경우 파일명 필터링 (예: KRX나 날짜가 포함된 것)
            csv_files = [
                f for f in all_csvs
                if 'KRX' in f
                   or re.search(r'\d{8}', f)  # 기존 날짜 형식 (20251205.csv)
                   or '_1Year' in f  # [New] 종목별 1년치 파일 (000270_1Year.csv)
            ]
            csv_files.sort()
        else:
            print(" 처리할 CSV 파일이나 데이터 폴더가 지정되지 않았습니다.")
            return None, None, None

        if not csv_files:
            print(f" '{self.data_dir}' 경로에 CSV 파일이 없습니다.")
            return None, None, None

        all_X, all_y, all_codes = [], [], []
        for f in csv_files:
            X_part, y_part, codes_part = self._process_single_file(f)
            if X_part is not None:
                all_X.append(X_part)
                all_y.append(y_part)
                all_codes.extend(codes_part)

        if not all_X: return None, None, None

        final_X = pd.concat(all_X, ignore_index=True)
        final_y = pd.concat(all_y, ignore_index=True)
        final_codes = pd.Series(all_codes, name='stck_shrn_iscd')
        
        print("\n 모델 입력을 위한 데이터 클리닝 (문자열 제거)...")
        
        drop_cols = ['stck_shrn_iscd', 'stock_code', 'code', 'date', 'timestamp']
        final_X = final_X.drop(columns=[c for c in drop_cols if c in final_X.columns], errors='ignore')
        
        # 안전장치: 숫자형 컬럼만 남기기
        final_X = final_X.select_dtypes(include=[np.number])
        
        # 공시 피처 포함 여부 확인 및 출력
        disc_cols = [c for c in final_X.columns if c.startswith('disc_')]
        if disc_cols:
            print(f"\n ✓ 공시 피처가 최종 데이터셋에 포함되었습니다! (disc_ 접두사 컬럼: {len(disc_cols)}개)")
            print(f"    공시 피처 목록: {disc_cols}")
        else:
            print(f"\n ⚠ 공시 피처가 최종 데이터셋에 포함되지 않았습니다 (disclosure_df가 비어있거나 병합 실패)")
        
        print(f"\n 통합 완료!")
        print(f" 총 파일 수: {len(csv_files)}개")
        print(f" 총 샘플 수: {len(final_X):,}")
        print(f" 최종 피처 수: {len(final_X.columns)}개")
        print("="*60)
        
        return final_X, final_y, final_codes

if __name__ == "__main__":
    # 데이터 폴더 경로 확인하세요!
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/krx_data"))
    
    if not os.path.exists(data_dir):
        # 경로가 없으면 기본 data 폴더로 시도
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

    if os.path.exists(data_dir):
        engineer = FeatureEngineer(data_dir=data_dir)
        X, y, _ = engineer.create_final_features()
        if X is not None:
            # 저장
            save_path = os.path.join(os.path.dirname(data_dir), "final_train_data.csv")
            final_df = pd.concat([X, y], axis=1)
            final_df.to_csv(save_path, index=False)
            print(f" {save_path} 에 저장 완료")
            print(X.dtypes)