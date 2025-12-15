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

# GAE 모델 로드
try:
    from ai_pipeline.gcn_model.model import get_gae_model
except ImportError:
    print(" GCN 모델 파일을 찾을 수 없습니다.")
    get_gae_model = None


# =========================================================
# ✅ GCN 로더 클래스 (수정됨: .npy 파일 우선 로드 방식)
# =========================================================
class GCNFeatureExtractor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # GCN 모델 폴더 (gcn_embeddings.npy가 있는 곳)
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
        """기존 방식: .pt 파일과 모델을 로드하여 추론"""
        if not os.path.exists(self.pt_path):
            print(f" [GCN] 데이터 파일이 없습니다: {self.pt_path}")
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
        # 1. Static 파일(.npy)이 있으면 그걸 읽어서 리턴 (가장 확실함)
        if self.use_static_file:
            try:
                # 노드 리스트(종목코드) 로드
                node_df = pd.read_csv(self.csv_path, dtype=str)
                if 'code' in node_df.columns:
                    codes = node_df['code'].tolist()
                elif 'stock_code' in node_df.columns:
                    codes = node_df['stock_code'].tolist()
                else:
                    return {}

                # 임베딩 로드
                embs = np.load(self.npy_path)

                # 딕셔너리로 매핑
                mapping = {}
                for i, code in enumerate(codes):
                    # 6자리 문자열로 변환
                    code_str = str(code).strip().zfill(6)
                    if i < len(embs):
                        mapping[code_str] = embs[i]

                return mapping
            except Exception as e:
                print(f" [GCN] .npy 로드 실패: {e}")
                return {}

        # 2. 없으면 기존 모델 추론 방식 사용
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
                if idx in idx_to_stock:
                    code = str(idx_to_stock[idx]).zfill(6)
                    mapping[code] = vector

        return mapping

    def add_gcn_features(self, df, code_col='code'):
        # [수정] 어떤 컬럼이 오든 처리할 수 있도록 유연하게 설정
        target_col = code_col
        # 컬럼 이름 정규화
        if 'stck_shrn_iscd' in df.columns:
            target_col = 'stck_shrn_iscd'
        elif 'code' in df.columns:
            target_col = 'code'
        elif 'stock_code' in df.columns:
            target_col = 'stock_code'

        emb_dict = self.get_embeddings()
        if not emb_dict: return df

        # Dict -> DataFrame
        emb_df = pd.DataFrame.from_dict(emb_dict, orient='index')
        emb_df.columns = [f'gcn_emb_{i}' for i in range(emb_df.shape[1])]
        emb_df.index.name = target_col
        emb_df = emb_df.reset_index()

        # 타입 통일 (문자열, 6자리)
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

        # 공시 데이터 로드 (생략 가능하나 유지)
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

    def _get_date_from_filename(self, filepath):
        # 파일명 날짜 추출은 보조 수단으로만 사용
        basename = os.path.basename(filepath)
        match = re.search(r'(\d{8})', basename)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d")
            except:
                pass
        return None

    def merge_sentiment_scores(self, X, stock_codes, current_file_path):
        """
        [수정] 데이터의 timestamp 컬럼을 기준으로 '일자별' 뉴스 데이터를 정확히 매핑합니다.
        """
        if isinstance(stock_codes, pd.Series): stock_codes = stock_codes.tolist()

        # 검색 대상 종목코드 정규화
        unique_codes = list(set([str(c).strip().zfill(6) for c in stock_codes]))

        # 피처 초기화
        X['sentiment_score'] = 0.0
        X['sentiment_volatility'] = 0.0
        X['sentiment_trend'] = 0.0

        # [중요] timestamp가 없으면 진행 불가
        if not self.es or 'timestamp' not in X.columns:
            return X

        # 날짜 범위 파악
        if not pd.api.types.is_datetime64_any_dtype(X['timestamp']):
            try:
                X['timestamp'] = pd.to_datetime(X['timestamp'])
            except:
                return X

        min_date = X['timestamp'].min()
        max_date = X['timestamp'].max()

        # 뉴스 반영 기간: 데이터 날짜 기준 2일 전 ~ 데이터 날짜 + 1일
        start_dt = min_date - timedelta(days=2)
        end_dt = max_date + timedelta(days=1)

        # ES 집계 쿼리 (일자별 히스토그램)
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
                            "date_histogram": {
                                "field": "published_date",
                                "calendar_interval": "day",
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

            # 매핑 데이터 생성 {(종목, 날짜): {features}}
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

            # DataFrame 병합 (Merge)
            # 1. 병합 키 준비
            code_col = 'stck_shrn_iscd' if 'stck_shrn_iscd' in X.columns else 'stock_code'
            X['join_code'] = X[code_col].astype(str).str.strip().str.zfill(6)
            X['join_date'] = X['timestamp'].dt.strftime('%Y-%m-%d')

            # 2. 매핑 데이터 DF 변환
            map_data = []
            for (c, d), v in sentiment_map.items():
                map_data.append({
                    'join_code': c,
                    'join_date': d,
                    's_score': v['score'],
                    's_vol': v['vol'],
                    's_trend': v['trend']
                })

            if map_data:
                sent_df = pd.DataFrame(map_data)
                # Left Join
                X = pd.merge(X, sent_df, on=['join_code', 'join_date'], how='left')

                # 결측치 채우기
                X['sentiment_score'] = X['s_score'].fillna(0.0)
                X['sentiment_volatility'] = X['s_vol'].fillna(0.0)
                X['sentiment_trend'] = X['s_trend'].fillna(0.0)

                # 임시 컬럼 정리
                X.drop(columns=['s_score', 's_vol', 's_trend'], inplace=True, errors='ignore')

            # 조인용 임시 컬럼 삭제
            X.drop(columns=['join_code', 'join_date'], inplace=True, errors='ignore')

        except Exception as e:
            # print(f" [ES Error] 뉴스 집계 실패: {e}")
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

        # 병합을 위해 임시로 종목코드 추가
        temp_code_col = 'stck_shrn_iscd'
        X[temp_code_col] = stock_codes

        # 기술적 지표 추가
        if self.expander:
            X = self.expander.add_technical_indicators(X)

        # GCN 피처 추가
        if self.gcn_loader:
            X = self.gcn_loader.add_gcn_features(X, code_col=temp_code_col)

        # 감성 점수 추가 (일자별 매핑)
        X = self.merge_sentiment_scores(X, stock_codes, csv_file)
        X = X.fillna(0)

        return X, y, stock_codes

    def create_final_features(self):
        print("\n" + "=" * 60)
        print(f" 통합 데이터셋 생성 시작")
        print("=" * 60)

        csv_files = []
        if self.csv_path and os.path.exists(self.csv_path):
            csv_files = [self.csv_path]
        elif self.data_dir and os.path.isdir(self.data_dir):
            all_csvs = glob.glob(os.path.join(self.data_dir, "*.csv"))
            csv_files = [
                f for f in all_csvs
                if 'KRX' in f or re.search(r'\d{8}', f) or '_1Year' in f
            ]
            csv_files.sort()
        else:
            print(" CSV 파일이나 데이터 폴더가 지정되지 않았습니다.")
            return None, None, None

        if not csv_files:
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

        print("\n 모델 입력을 위한 데이터 클리닝...")

        # 학습에 불필요한 식별자 컬럼 제거 (timestamp 포함)
        drop_cols = ['stck_shrn_iscd', 'stock_code', 'code', 'date', 'timestamp']
        final_X = final_X.drop(columns=[c for c in drop_cols if c in final_X.columns], errors='ignore')

        final_X = final_X.select_dtypes(include=[np.number])

        disc_cols = [c for c in final_X.columns if c.startswith('disc_')]
        if disc_cols:
            print(f" ✓ 공시 피처 포함: {len(disc_cols)}개")

        print(f"\n 통합 완료! 총 샘플 수: {len(final_X):,}, 피처 수: {len(final_X.columns)}")
        print("=" * 60)

        return final_X, final_y, final_codes


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/krx_data"))
    if not os.path.exists(data_dir):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

    if os.path.exists(data_dir):
        engineer = FeatureEngineer(data_dir=data_dir)
        X, y, _ = engineer.create_final_features()
        if X is not None:
            save_path = os.path.join(os.path.dirname(data_dir), "final_train_data.csv")
            pd.concat([X, y], axis=1).to_csv(save_path, index=False)
            print(f" {save_path} 에 저장 완료")