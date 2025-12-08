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
            # ✅ [수정] GAE 모델은 encode 메서드를 사용해야 함
            try:
                embeddings = self.model.encode(self.data.x, self.data.edge_index)
            except AttributeError:
                # 일반 GCN일 경우 forward 호출
                embeddings = self.model(self.data.x, self.data.edge_index)
                
        emb_np = embeddings.cpu().numpy()
        
        mapping = {}
        
        # 1순위: .pt 파일 내장 정보
        if hasattr(self.data, 'stock_to_idx'):
            idx_to_stock = {v: k for k, v in self.data.stock_to_idx.items()}
        # 2순위: JSON 파일
        else:
            idx_to_stock = self._load_json_mapping()
            
        if idx_to_stock:
            for idx, vector in enumerate(emb_np):
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
                            print(f" 공시 데이터 로드됨: 종목 {self.disclosure_df.shape[0]}개, 컬럼 {self.disclosure_df.shape[1]}개 (경로: {disc_path})")
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
        if isinstance(stock_codes, pd.Series): stock_codes = stock_codes.tolist()
        X['sentiment_score'] = 0.0
        if not self.es: return X

        target_date = self._get_date_from_filename(current_file_path)
        if target_date:
            end_dt = target_date.replace(hour=23, minute=59, second=59)
            start_dt = end_dt - timedelta(days=2)
            range_filter = {"range": {"timestamp": {"gte": start_dt.isoformat(), "lte": end_dt.isoformat()}}}
        else:
            range_filter = {"match_all": {}}

        try:
            body = {
                "size": 0, "query": range_filter,
                "aggs": {
                    "by_stock": {
                        "terms": {"field": "related_stocks.keyword", "size": 3000},
                        "aggs": {"avg_sentiment": {"avg": {"field": "sentiment_score"}}}
                    }
                }
            }
            resp = self.es.search(index="news_articles", body=body)
            # 안전하게 버킷 접근
            if 'aggregations' in resp and 'by_stock' in resp['aggregations']:
                buckets = resp['aggregations']['by_stock']['buckets']
                score_map = {b['key']: b['avg_sentiment']['value'] for b in buckets if b['avg_sentiment']['value'] is not None}
                
                sentiment_list = [score_map.get(str(code).zfill(6), 0.0) for code in stock_codes]
                X['sentiment_score'] = sentiment_list
        except: pass 
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

        # 공시 데이터 병합 (종목코드 기준) — 가능한 경우에만
        try:
            if self.disclosure_df is not None and temp_code_col in X.columns:
                X[temp_code_col] = X[temp_code_col].astype(str).str.zfill(6)
                # 대용량 병합 성능을 위해 컬럼별 매핑(map) 방식으로 병합
                # disclosure_df는 index가 stock_code이며 숫자형 컬럼만 포함
                # 인덱스와 값을 numpy로 미리 준비 (벡터화된 인덱싱)
                disc_index = self.disclosure_df.index.astype(str).str.strip().str.zfill(6)
                for c in self.disclosure_df.columns:
                    try:
                        arr = self.disclosure_df[c].to_numpy()
                        keys = disc_index
                        # X의 코드 배열 (정규화: strip + zfill)
                        codes = X[temp_code_col].astype(str).str.strip().str.zfill(6).to_numpy()
                        # get_indexer를 사용하면 벡터화된 인덱싱이 가능
                        idx = keys.get_indexer(codes)
                        # idx == -1 은 없는 값 -> 0 채움
                        import numpy as _np
                        vals = _np.where(idx >= 0, arr[idx], 0)
                        X[c] = vals
                        # 안전하게 숫자형으로 변환
                        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
                    except Exception:
                        if c in X.columns:
                            del X[c]
        except Exception:
            pass
        
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
            csv_files = [f for f in all_csvs if 'KRX' in f or re.search(r'\d{8}', f)]
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