import pandas as pd
import numpy as np
import torch
import json
import os
import sys
import gc
from tqdm import tqdm
import re
import glob
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from ai_pipeline.config.settings import ES_HOST

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 필요한 클래스 import
try:
    from ai_pipeline.boosting_model.realtime_feature_loader import RealtimeFeatureLoader
    from ai_pipeline.boosting_model.feature_expander import FeatureExpander
except ImportError:
    pass

try:
    from ai_pipeline.gcn_model.model import get_gae_model
except ImportError:
    print(" GCN 모델 파일을 찾을 수 없습니다.")
    get_gae_model = None


# =========================================================
# ✅ 메모리 최적화 함수 (날짜 타입 오류 방지 적용)
# =========================================================
def reduce_mem_usage(df):
    """ 데이터프레임의 numeric 컬럼 타입을 다운캐스팅하여 메모리를 절약합니다. """
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        # Object(문자열)나 Datetime(날짜)이 아닌 경우에만 처리
        if col_type != object and not pd.api.types.is_datetime64_any_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()

            # 정수형 처리
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

            # 실수형 처리 (float)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)

    return df


# =========================================================
# ✅ GCN 로더 클래스
# =========================================================
class GCNFeatureExtractor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.gcn_dir = os.path.abspath(os.path.join(current_dir, "../../data"))

        # 1순위: 저장된 결과 파일 (.npy + .csv)
        self.npy_path = os.path.join(self.gcn_dir, "gcn_embeddings.npy")
        self.csv_path = os.path.join(self.gcn_dir, "gcn_node_list.csv")

        # 2순위: 모델 및 데이터 파일 (Fallback)
        self.pt_path = os.path.abspath(os.path.join(current_dir, "../../finance_graph_data.pt"))
        self.model = None
        self.data = None

        # 초기화 시 .npy 파일이 있는지 확인
        if os.path.exists(self.npy_path) and os.path.exists(self.csv_path):
            print(f" [GCN] 저장된 임베딩 파일 사용 (.npy): {os.path.basename(self.npy_path)}")
            self.use_static_file = True
        else:
            print(f" [GCN] 저장된 .npy 파일이 없습니다. 모델 로드를 시도합니다.")
            self.use_static_file = False
            self._init_model(model_path)

    def _init_model(self, model_path):
        if not os.path.exists(self.pt_path):
            return
        try:
            self.data = torch.load(self.pt_path, map_location=self.device, weights_only=False)
        except:
            self.data = torch.load(self.pt_path, map_location=self.device)

        if self.data is not None and get_gae_model is not None:
            num_features = self.data.x.shape[1]
            self.model = get_gae_model(in_channels=num_features, out_channels=16).to(self.device)
            if model_path is None:
                model_path = os.path.abspath(os.path.join(self.gcn_dir, "../../best_gcn_model.pth"))
            if os.path.exists(model_path):
                try:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True),
                                               strict=False)
                except:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
                self.model.eval()

    def get_embeddings(self):
        if self.use_static_file:
            try:
                node_df = pd.read_csv(self.csv_path, dtype=str)
                if 'code' in node_df.columns:
                    codes = node_df['code'].tolist()
                elif 'stock_code' in node_df.columns:
                    codes = node_df['stock_code'].tolist()
                else:
                    return {}

                embs = np.load(self.npy_path)
                mapping = {}
                for i, code in enumerate(codes):
                    code_str = str(code).strip().zfill(6)
                    if i < len(embs): mapping[code_str] = embs[i]
                return mapping
            except Exception as e:
                print(f" [GCN] .npy 로드 실패: {e}")
                return {}

        if self.data is None or self.model is None: return {}
        with torch.no_grad():
            try:
                embeddings = self.model.encode(self.data.x, self.data.edge_index)
            except AttributeError:
                embeddings = self.model(self.data.x, self.data.edge_index)
        emb_np = embeddings.cpu().numpy()
        mapping = {}
        if hasattr(self.data, 'stock_to_idx'):
            idx_to_stock = {v: k for k, v in self.data.stock_to_idx.items()}
            for idx, vector in enumerate(emb_np):
                if idx in idx_to_stock: mapping[str(idx_to_stock[idx]).zfill(6)] = vector
        return mapping

    def add_gcn_features(self, df, code_col='code'):
        target_col = code_col
        if 'stck_shrn_iscd' in df.columns:
            target_col = 'stck_shrn_iscd'
        elif 'code' in df.columns:
            target_col = 'code'
        elif 'stock_code' in df.columns:
            target_col = 'stock_code'

        emb_dict = self.get_embeddings()
        if not emb_dict: return df

        emb_df = pd.DataFrame.from_dict(emb_dict, orient='index')
        emb_df.columns = [f'gcn_emb_{i}' for i in range(emb_df.shape[1])]
        emb_df.index.name = target_col
        emb_df = emb_df.reset_index()

        df[target_col] = df[target_col].astype(str).str.strip().str.zfill(6)
        emb_df[target_col] = emb_df[target_col].astype(str).str.strip().str.zfill(6)

        merged_df = pd.merge(df, emb_df, on=target_col, how='left')
        gcn_cols = [c for c in merged_df.columns if c.startswith('gcn_')]
        merged_df[gcn_cols] = merged_df[gcn_cols].fillna(0)
        return merged_df

    def get_features(self, code):
        mapping = self.get_embeddings()
        code = str(code).strip().zfill(6)
        if code in mapping:
            return {f'gcn_emb_{i}': val for i, val in enumerate(mapping[code])}
        return {}


# =========================================================
# ✅ 메인 FeatureEngineer 클래스
# =========================================================
class FeatureEngineer:
    def __init__(self, data_dir=None, csv_path=None):
        self.data_dir = data_dir
        self.csv_path = csv_path
        try:
            self.expander = FeatureExpander()
        except:
            self.expander = None
        try:
            self.gcn_loader = GCNFeatureExtractor()
        except Exception as e:
            print(f" GCN 로더 초기화 실패: {e}")
            self.gcn_loader = None
        try:
            self.es = Elasticsearch(ES_HOST)
            if not self.es.ping(): self.es = None
        except:
            self.es = None

        # 공시 데이터 로드
        self.disclosure_df = None
        try:
            disc_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../disclosure_pipeline/data/integrated_financial_data.csv"))
            if os.path.exists(disc_path):
                ddf = pd.read_csv(disc_path, encoding='utf-8-sig')
                if 'stock_code' in ddf.columns:
                    ddf['stock_code'] = ddf['stock_code'].astype(str).str.strip().str.zfill(6)
                    num_cols = ddf.select_dtypes(include=[np.number]).columns.tolist()
                    keep_cols = ['stock_code'] + [c for c in num_cols if c != 'stock_code']
                    if len(keep_cols) > 1:
                        disc_df = ddf[keep_cols].set_index('stock_code')
                        new_cols = {c: f"disc_{c}" for c in disc_df.columns if c != 'stock_code'}
                        disc_df = disc_df.rename(columns=new_cols)
                        self.disclosure_df = disc_df
        except:
            pass

    def merge_sentiment_scores(self, X, stock_codes, current_file_path):
        if isinstance(stock_codes, pd.Series): stock_codes = stock_codes.tolist()
        unique_codes = list(set([str(c).strip().zfill(6) for c in stock_codes]))

        X['sentiment_score'] = 0.0
        X['sentiment_volatility'] = 0.0
        X['sentiment_trend'] = 0.0

        if not self.es or 'timestamp' not in X.columns: return X
        if not pd.api.types.is_datetime64_any_dtype(X['timestamp']):
            try:
                X['timestamp'] = pd.to_datetime(X['timestamp'])
            except:
                return X

        min_date = X['timestamp'].min()
        max_date = X['timestamp'].max()
        start_dt = min_date - timedelta(days=2)
        end_dt = max_date + timedelta(days=1)

        body = {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"published_date": {"gte": start_dt.isoformat(), "lte": end_dt.isoformat()}}},
                        {"terms": {"related_stocks.keyword": unique_codes}}
                    ]
                }
            },
            "aggs": {
                "by_stock": {
                    "terms": {"field": "related_stocks.keyword", "size": 1000},
                    "aggs": {
                        "by_day": {
                            "date_histogram": {"field": "published_date", "calendar_interval": "day",
                                               "format": "yyyy-MM-dd"},
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
            sentiment_map = {}
            if 'aggregations' in resp and 'by_stock' in resp['aggregations']:
                for stock_bucket in resp['aggregations']['by_stock']['buckets']:
                    code = stock_bucket['key']
                    for date_bucket in stock_bucket['by_day']['buckets']:
                        date_str = date_bucket['key_as_string']
                        vals = {
                            'score': date_bucket['avg_sent']['value'],
                            'vol': date_bucket['avg_vol']['value'],
                            'trend': date_bucket['avg_trend']['value']
                        }
                        sentiment_map[(code, date_str)] = {k: (v if v is not None else 0.0) for k, v in vals.items()}

            code_col = None
            for c in ['stck_shrn_iscd', 'stock_code', 'code']:
                if c in X.columns: code_col = c; break

            if code_col:
                X['join_code'] = X[code_col].astype(str).str.strip().str.zfill(6)
                X['join_date'] = X['timestamp'].dt.strftime('%Y-%m-%d')

                map_data = []
                for (c, d), v in sentiment_map.items():
                    map_data.append({'join_code': c, 'join_date': d, 's_score': v['score'], 's_vol': v['vol'],
                                     's_trend': v['trend']})

                if map_data:
                    sent_df = pd.DataFrame(map_data)
                    X = pd.merge(X, sent_df, on=['join_code', 'join_date'], how='left')
                    for col, src in [('sentiment_score', 's_score'), ('sentiment_volatility', 's_vol'),
                                     ('sentiment_trend', 's_trend')]:
                        X[col] = X[src].fillna(0.0)
                    X.drop(columns=['s_score', 's_vol', 's_trend'], inplace=True, errors='ignore')
                X.drop(columns=['join_code', 'join_date'], inplace=True, errors='ignore')
        except:
            pass
        return X

    def _process_single_file(self, csv_file):
        print(f" 처리 중: {os.path.basename(csv_file)}")
        loader = RealtimeFeatureLoader(csv_file)
        try:
            load_result = loader.prepare_features()
            if len(load_result) == 3:
                X, y, stock_codes = load_result
            else:
                X, y = load_result;
                stock_codes = []
        except:
            return None, None, None

        if X is None or X.empty: return None, None, None

        temp_code_col = 'stck_shrn_iscd'
        X[temp_code_col] = stock_codes

        # [핵심 수정] Target(y)을 X에 임시 포함하여 처리 중 누락/중복 방지
        X['target_temp'] = y.values

        if self.expander: X = self.expander.add_technical_indicators(X)
        if self.gcn_loader: X = self.gcn_loader.add_gcn_features(X, code_col=temp_code_col)

        X = self.merge_sentiment_scores(X, stock_codes, csv_file)
        X = X.fillna(0)

        # [수정] 메모리 최적화 적용
        X = reduce_mem_usage(X)

        # 처리 후 Target 분리
        if 'target_temp' in X.columns:
            y_final = X['target_temp'].copy()
            X = X.drop(columns=['target_temp'])
        else:
            # 혹시라도 target_temp가 사라졌다면(그럴 일은 없어야 함) NaN 채움
            y_final = pd.Series(0, index=X.index)

        return X, y_final, stock_codes

    def create_final_features(self, use_cache=True, force_update=False):
        """
        통합 데이터셋 생성 (캐싱 기능 포함)
        """
        print("\n" + "=" * 60)
        print(f" 통합 데이터셋 생성 시작")
        print("=" * 60)

        if self.data_dir:
            cache_file = os.path.join(self.data_dir, "cached_final_features.csv")
        else:
            cache_file = "cached_final_features.csv"

        # 1. 캐시 확인 및 로드
        if use_cache and not force_update and os.path.exists(cache_file):
            print(f" [Cache] 기존 통합 데이터 발견! 로드 중... ({cache_file})")
            try:
                df_cache = pd.read_csv(cache_file)
                df_cache = reduce_mem_usage(df_cache)

                if 'target' not in df_cache.columns:
                    print(" [Cache Error] 'target' 컬럼이 없습니다. 재생성합니다.")
                else:
                    y = df_cache['target']
                    if 'stck_shrn_iscd' in df_cache.columns:
                        codes = df_cache['stck_shrn_iscd']
                        exclude_cols = ['target', 'stck_shrn_iscd', 'timestamp', 'close', 'date', 'time']
                        X = df_cache.drop(columns=[c for c in exclude_cols if c in df_cache.columns], errors='ignore')
                    else:
                        codes = pd.Series([0] * len(df_cache))
                        X = df_cache.drop(columns=['target'])

                    X = X.select_dtypes(include=[np.number])

                    print(f" [Cache] 로드 완료! (샘플 수: {len(X):,})")
                    return X, y, codes

            except Exception as e:
                print(f" [Cache Error] 로드 중 오류 발생: {e}. 데이터를 재생성합니다.")

        # 2. 캐시가 없으면 생성
        csv_files = []
        if self.csv_path and os.path.exists(self.csv_path):
            csv_files = [self.csv_path]
        elif self.data_dir and os.path.isdir(self.data_dir):
            all_csvs = glob.glob(os.path.join(self.data_dir, "*.csv"))
            csv_files = [f for f in all_csvs if '_1Year' in f and 'cached_final_features.csv' not in f]
            csv_files.sort()
        else:
            print(" CSV 파일이나 데이터 폴더가 지정되지 않았습니다.")
            return None, None, None

        if not csv_files:
            print(" [Warning] 처리할 '_1Year.csv' 파일이 없습니다.")
            return None, None, None

        print(f" 총 {len(csv_files)}개의 분봉 데이터 파일을 처리합니다.")

        all_X, all_y, all_codes = [], [], []

        # [수정] 루프 내 메모리 관리
        for i, f in enumerate(csv_files):
            X_part, y_part, codes_part = self._process_single_file(f)
            if X_part is not None:
                all_X.append(X_part)
                all_y.append(y_part)

                # [안전장치] 행 개수가 늘어났다면 코드 리스트도 맞춰줌
                if len(codes_part) != len(X_part):
                    if 'stck_shrn_iscd' in X_part.columns:
                        codes_part = X_part['stck_shrn_iscd'].tolist()
                    else:
                        factor = len(X_part) // len(codes_part)
                        codes_part = codes_part * factor

                all_codes.extend(codes_part)

            if i % 10 == 0:
                gc.collect()

        if not all_X: return None, None, None

        print(" [Concat] 데이터 병합 중...")
        final_X = pd.concat(all_X, ignore_index=True)
        final_y = pd.concat(all_y, ignore_index=True)
        final_codes = pd.Series(all_codes, name='stck_shrn_iscd')

        del all_X, all_y, all_codes
        gc.collect()

        print("\n 모델 입력을 위한 데이터 클리닝...")

        # [수정] CSV 저장용 메타데이터 백업
        meta_data = {}
        if 'Close' in final_X.columns:
            meta_data['close'] = final_X['Close']
        elif 'close' in final_X.columns:
            meta_data['close'] = final_X['close']

        req_cols = ['timestamp', 'open', 'high', 'low', 'volume', 'date', 'time']
        for col in req_cols:
            if col in final_X.columns:
                meta_data[col] = final_X[col]
            elif col.capitalize() in final_X.columns:
                meta_data[col] = final_X[col.capitalize()]

        drop_cols = ['stck_shrn_iscd', 'stock_code', 'code', 'date', 'timestamp', 'Close', 'close']
        drop_cols.extend(list(meta_data.keys()))

        final_X = final_X.drop(columns=[c for c in drop_cols if c in final_X.columns], errors='ignore')
        final_X = final_X.select_dtypes(include=[np.number])

        # 마지막으로 한 번 더 최적화
        final_X = reduce_mem_usage(final_X)

        # 3. 캐시 저장
        print(f" [Cache] 다음 실행 속도 향상을 위해 데이터를 저장합니다... ({cache_file})")

        meta_df = pd.DataFrame(meta_data)
        meta_df['target'] = final_y.values
        meta_df['stck_shrn_iscd'] = final_codes.values

        # final_X와 meta_df 병합
        save_df = pd.concat([final_X, meta_df], axis=1)

        save_df.to_csv(cache_file, index=False)
        del save_df, meta_df
        gc.collect()

        print(" [Cache] 저장 완료!")
        print(f"\n 통합 완료! 총 샘플 수: {len(final_X):,}, 피처 수: {len(final_X.columns)}")
        print("=" * 60)

        return final_X, final_y, final_codes


def process_and_save_data(data_dir, cache_file):
    # (이전 코드와 동일, reduce_mem_usage만 새 버전 사용됨)
    print(f"\n[Info] 데이터 처리 시작... (경로: {data_dir})")
    # ... (생략, 위 클래스 로직으로 충분함) ...
    pass  # 실제 실행은 FeatureEngineer.create_final_features()로 수행됨


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    data_dir = os.path.join(project_root, "data")

    eng = FeatureEngineer(data_dir=data_dir)
    eng.create_final_features()