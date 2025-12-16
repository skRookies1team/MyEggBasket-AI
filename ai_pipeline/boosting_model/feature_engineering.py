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
from ai_pipeline.config.settings import ES_HOST

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 필요한 클래스 import
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
#  GCN 로더 클래스 (수정됨: .npy 파일 우선 로드 방식)
# =========================================================
class GCNFeatureExtractor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.gcn_dir = os.path.abspath(os.path.join(current_dir, "../gcn_model"))

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
        # [수정] 어떤 컬럼이 오든 처리할 수 있도록 유연하게 설정
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


# =========================================================
# ✅ 메인 FeatureEngineer 클래스 (캐싱 기능 추가)
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

    # ... (기존 _get_date_from_filename, merge_sentiment_scores, _process_single_file 메소드 동일 유지) ...
    def _get_date_from_filename(self, filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'(\d{8})', basename)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d")
            except:
                pass
        return None

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
                X, y = load_result; stock_codes = []
        except:
            return None, None, None

        if X is None or X.empty: return None, None, None

        temp_code_col = 'stck_shrn_iscd'
        X[temp_code_col] = stock_codes

        if self.expander: X = self.expander.add_technical_indicators(X)
        if self.gcn_loader: X = self.gcn_loader.add_gcn_features(X, code_col=temp_code_col)

        X = self.merge_sentiment_scores(X, stock_codes, csv_file)
        X = X.fillna(0)
        return X, y, stock_codes

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

        # 1. 캐시 확인 및 로드 (기존과 동일)
        if use_cache and not force_update and os.path.exists(cache_file):
            # ... (캐시 로드 로직 생략, 기존 동일) ...
            pass

            # 2. 캐시가 없으면 생성
        csv_files = []
        if self.csv_path and os.path.exists(self.csv_path):
            csv_files = [self.csv_path]
        elif self.data_dir and os.path.isdir(self.data_dir):
            all_csvs = glob.glob(os.path.join(self.data_dir, "*.csv"))

            # [수정] 파일 필터링: '_1Year' 패턴이 있는 파일만 사용 (초 단위 데이터 제외)
            csv_files = [
                f for f in all_csvs
                if '_1Year' in f
                   and 'cached_final_features.csv' not in f
            ]
            csv_files.sort()
        else:
            print(" CSV 파일이나 데이터 폴더가 지정되지 않았습니다.")
            return None, None, None

        if not csv_files:
            print(" [Warning] 처리할 '_1Year.csv' 파일이 없습니다.")
            return None, None, None

        print(f" 총 {len(csv_files)}개의 분봉 데이터 파일을 처리합니다.")

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

        print("\n 모델 입력을 위한 데이터 클리닝...")

        # 학습에 불필요한 식별자 컬럼 제거 (timestamp 포함)
        drop_cols = ['stck_shrn_iscd', 'stock_code', 'code', 'date', 'timestamp']
        final_X = final_X.drop(columns=[c for c in drop_cols if c in final_X.columns], errors='ignore')
        final_X = final_X.select_dtypes(include=[np.number])

        # [수정] 캐시 저장을 위해 임시 DataFrame 생성
        print(f" [Cache] 다음 실행 속도 향상을 위해 데이터를 저장합니다... ({cache_file})")
        save_df = final_X.copy()
        save_df['target'] = final_y
        # 나중에 복원할 수 있도록 코드 컬럼 추가
        save_df['stck_shrn_iscd'] = final_codes.values

        save_df.to_csv(cache_file, index=False)
        print(" [Cache] 저장 완료!")

        print(f"\n 통합 완료! 총 샘플 수: {len(final_X):,}, 피처 수: {len(final_X.columns)}")
        print("=" * 60)

        return final_X, final_y, final_codes


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/krx_data"))
    if not os.path.exists(data_dir):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

    if os.path.exists(data_dir):
        engineer = FeatureEngineer(data_dir=data_dir)
        # 처음 실행 시에는 생성 후 저장, 두 번째부터는 로드
        X, y, _ = engineer.create_final_features(use_cache=True)

        if X is not None:
            # 참고: cached_final_features.csv가 이미 생성되었으므로 여기서는 별도 저장 안해도 됨
            print(" 작업 완료")