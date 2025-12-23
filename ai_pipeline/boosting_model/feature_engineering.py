import pandas as pd
import numpy as np
import torch
import json
import os
import sys
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
# ✅ GCN 로더 클래스 (수정됨: .npy 파일 우선 로드 방식)
# =========================================================
class GCNFeatureExtractor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.gcn_dir = os.path.abspath(os.path.join(current_dir, "../../data"))

        # 1순위: 저장된 결과 파일 (.npy + .csv)
        self.npy_path = os.path.join(self.gcn_dir, "gcn_embeddings.npy")
        self.csv_path = os.path.join(self.gcn_dir, "gcn_node_list.csv")

        # 2순위: 모델 및 데이터 파일 (Fallback)
        self.pt_path = os.path.abspath(
            os.path.join(current_dir, "../../finance_graph_data.pt")
        )
        self.model = None
        self.data = None

        # 초기화 시 .npy 파일이 있는지 확인
        if os.path.exists(self.npy_path) and os.path.exists(self.csv_path):
            print(
                f" [GCN] 저장된 임베딩 파일 사용 (.npy): {os.path.basename(self.npy_path)}"
            )
            self.use_static_file = True
        else:
            print(f" [GCN] 저장된 .npy 파일이 없습니다. 모델 로드를 시도합니다.")
            self.use_static_file = False
            self._init_model(model_path)

    def _init_model(self, model_path):
        if not os.path.exists(self.pt_path):
            return
        try:
            self.data = torch.load(
                self.pt_path, map_location=self.device, weights_only=False
            )
        except:
            self.data = torch.load(self.pt_path, map_location=self.device)

        if self.data is not None and get_gae_model is not None:
            num_features = self.data.x.shape[1]
            self.model = get_gae_model(in_channels=num_features, out_channels=16).to(
                self.device
            )
            if model_path is None:
                model_path = os.path.abspath(
                    os.path.join(self.gcn_dir, "../../best_gcn_model.pth")
                )
            if os.path.exists(model_path):
                try:
                    self.model.load_state_dict(
                        torch.load(
                            model_path, map_location=self.device, weights_only=True
                        ),
                        strict=False,
                    )
                except:
                    self.model.load_state_dict(
                        torch.load(model_path, map_location=self.device), strict=False
                    )
                self.model.eval()

    def get_embeddings(self):
        if self.use_static_file:
            try:
                node_df = pd.read_csv(self.csv_path, dtype=str)
                if "code" in node_df.columns:
                    codes = node_df["code"].tolist()
                elif "stock_code" in node_df.columns:
                    codes = node_df["stock_code"].tolist()
                else:
                    return {}

                embs = np.load(self.npy_path)
                mapping = {}
                for i, code in enumerate(codes):
                    code_str = str(code).strip().zfill(6)
                    if i < len(embs):
                        mapping[code_str] = embs[i]
                return mapping
            except Exception as e:
                print(f" [GCN] .npy 로드 실패: {e}")
                return {}

        if self.data is None or self.model is None:
            return {}
        with torch.no_grad():
            try:
                embeddings = self.model.encode(self.data.x, self.data.edge_index)
            except AttributeError:
                embeddings = self.model(self.data.x, self.data.edge_index)
        emb_np = embeddings.cpu().numpy()
        mapping = {}
        if hasattr(self.data, "stock_to_idx"):
            idx_to_stock = {v: k for k, v in self.data.stock_to_idx.items()}
            for idx, vector in enumerate(emb_np):
                if idx in idx_to_stock:
                    mapping[str(idx_to_stock[idx]).zfill(6)] = vector
        return mapping

    def add_gcn_features(self, df, code_col="code"):
        # [수정] 어떤 컬럼이 오든 처리할 수 있도록 유연하게 설정
        target_col = code_col
        if "stck_shrn_iscd" in df.columns:
            target_col = "stck_shrn_iscd"
        elif "code" in df.columns:
            target_col = "code"
        elif "stock_code" in df.columns:
            target_col = "stock_code"

        emb_dict = self.get_embeddings()
        if not emb_dict:
            return df

        emb_df = pd.DataFrame.from_dict(emb_dict, orient="index")
        emb_df.columns = [f"gcn_emb_{i}" for i in range(emb_df.shape[1])]
        emb_df.index.name = target_col
        emb_df = emb_df.reset_index()

        df[target_col] = df[target_col].astype(str).str.strip().str.zfill(6)
        emb_df[target_col] = emb_df[target_col].astype(str).str.strip().str.zfill(6)

        merged_df = pd.merge(df, emb_df, on=target_col, how="left")
        gcn_cols = [c for c in merged_df.columns if c.startswith("gcn_")]
        merged_df[gcn_cols] = merged_df[gcn_cols].fillna(0)
        return merged_df


# =========================================================
# ✅ 메인 FeatureEngineer 클래스 (캐싱 기능 포함)
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
            if not self.es.ping():
                self.es = None
        except:
            self.es = None

        # 공시 데이터 로드
        self.disclosure_df = None
        try:
            disc_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../disclosure_pipeline/data/integrated_financial_data.csv",
                )
            )
            if os.path.exists(disc_path):
                ddf = pd.read_csv(disc_path, encoding="utf-8-sig")
                if "stock_code" in ddf.columns:
                    ddf["stock_code"] = (
                        ddf["stock_code"].astype(str).str.strip().str.zfill(6)
                    )
                    num_cols = ddf.select_dtypes(include=[np.number]).columns.tolist()
                    keep_cols = ["stock_code"] + [
                        c for c in num_cols if c != "stock_code"
                    ]
                    if len(keep_cols) > 1:
                        disc_df = ddf[keep_cols].set_index("stock_code")
                        new_cols = {
                            c: f"disc_{c}" for c in disc_df.columns if c != "stock_code"
                        }
                        disc_df = disc_df.rename(columns=new_cols)
                        self.disclosure_df = disc_df
        except:
            pass

    def merge_sentiment_scores(self, X, stock_codes, current_file_path):
        if isinstance(stock_codes, pd.Series):
            stock_codes = stock_codes.tolist()
        unique_codes = list(set([str(c).strip().zfill(6) for c in stock_codes]))

        X["sentiment_score"] = 0.0
        X["sentiment_volatility"] = 0.0
        X["sentiment_trend"] = 0.0

        if not self.es or "timestamp" not in X.columns:
            return X
        if not pd.api.types.is_datetime64_any_dtype(X["timestamp"]):
            try:
                X["timestamp"] = pd.to_datetime(X["timestamp"])
            except:
                return X

        min_date = X["timestamp"].min()
        max_date = X["timestamp"].max()
        start_dt = min_date - timedelta(days=2)
        end_dt = max_date + timedelta(days=1)

        body = {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [
                        {
                            "range": {
                                "published_date": {
                                    "gte": start_dt.isoformat(),
                                    "lte": end_dt.isoformat(),
                                }
                            }
                        },
                        {"terms": {"related_stocks.keyword": unique_codes}},
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
                                "format": "yyyy-MM-dd",
                            },
                            "aggs": {
                                "avg_sent": {"avg": {"field": "sentiment_score"}},
                                "avg_vol": {
                                    "avg": {
                                        "field": "analysis_results.sentiment_volatility"
                                    }
                                },
                                "avg_trend": {
                                    "avg": {"field": "analysis_results.sentiment_trend"}
                                },
                            },
                        }
                    },
                }
            },
        }
        try:
            resp = self.es.search(index="news_articles", body=body)
            sentiment_map = {}
            if "aggregations" in resp and "by_stock" in resp["aggregations"]:
                for stock_bucket in resp["aggregations"]["by_stock"]["buckets"]:
                    code = stock_bucket["key"]
                    for date_bucket in stock_bucket["by_day"]["buckets"]:
                        date_str = date_bucket["key_as_string"]
                        vals = {
                            "score": date_bucket["avg_sent"]["value"],
                            "vol": date_bucket["avg_vol"]["value"],
                            "trend": date_bucket["avg_trend"]["value"],
                        }
                        sentiment_map[(code, date_str)] = {
                            k: (v if v is not None else 0.0) for k, v in vals.items()
                        }

            code_col = None
            for c in ["stck_shrn_iscd", "stock_code", "code"]:
                if c in X.columns:
                    code_col = c
                    break

            if code_col:
                X["join_code"] = X[code_col].astype(str).str.strip().str.zfill(6)
                X["join_date"] = X["timestamp"].dt.strftime("%Y-%m-%d")

                map_data = []
                for (c, d), v in sentiment_map.items():
                    map_data.append(
                        {
                            "join_code": c,
                            "join_date": d,
                            "s_score": v["score"],
                            "s_vol": v["vol"],
                            "s_trend": v["trend"],
                        }
                    )

                if map_data:
                    sent_df = pd.DataFrame(map_data)
                    X = pd.merge(X, sent_df, on=["join_code", "join_date"], how="left")
                    for col, src in [
                        ("sentiment_score", "s_score"),
                        ("sentiment_volatility", "s_vol"),
                        ("sentiment_trend", "s_trend"),
                    ]:
                        X[col] = X[src].fillna(0.0)
                    X.drop(
                        columns=["s_score", "s_vol", "s_trend"],
                        inplace=True,
                        errors="ignore",
                    )
                X.drop(
                    columns=["join_code", "join_date"], inplace=True, errors="ignore"
                )
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
                X, y = load_result
                stock_codes = []
        except:
            return None, None, None

        if X is None or X.empty:
            return None, None, None

        temp_code_col = "stck_shrn_iscd"
        X[temp_code_col] = stock_codes

        if self.expander:
            X = self.expander.add_technical_indicators(X)
        if self.gcn_loader:
            X = self.gcn_loader.add_gcn_features(X, code_col=temp_code_col)

        X = self.merge_sentiment_scores(X, stock_codes, csv_file)
        # 1) "0도 의미 있는 값"인 피처들 (절대 건드리지 않음)
        value_sensitive_keywords = (
            'prdy',        # 전일대비 등락률
            'return',      # 수익률
            'momentum',    # 모멘텀
            'roc',         # Rate of Change
        )
        
        # 2) NaN → 0 처리해도 되는 피처들
        safe_zero_prefixes = (
            'hist_',       # 기술적 지표
            'gcn_',        # GCN 임베딩
            'sentiment_',  # 감성 피처
        )

        zero_fill_prefixes = ("hist_", "gcn_", "sentiment_")
        zero_fill_cols = [c for c in X.columns if c.startswith(zero_fill_prefixes)]
        X[zero_fill_cols] = X[zero_fill_cols].fillna(0.0)

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

        # -----------------------------------------------------
        # 1. 캐시 확인 및 로드
        # -----------------------------------------------------
        if use_cache and not force_update and os.path.exists(cache_file):
            print(f" [Cache] 기존 통합 데이터 발견! 로드 중... ({cache_file})")
            try:
                df_cache = pd.read_csv(cache_file)
                if "target" not in df_cache.columns:
                    print(" [Cache Error] 'target' 컬럼이 없습니다. 재생성합니다.")
                else:
                    y = df_cache["target"]
                    if "stck_shrn_iscd" in df_cache.columns:
                        codes = df_cache["stck_shrn_iscd"]
                        # 학습용 X에서는 타겟, 코드, 그리고 메타데이터(close, timestamp 등) 제외
                        exclude_cols = [
                            "target",
                            "stck_shrn_iscd",
                            "timestamp",
                            "close",
                            "date",
                            "time",
                        ]
                        X = df_cache.drop(
                            columns=[c for c in exclude_cols if c in df_cache.columns],
                            errors="ignore",
                        )
                    else:
                        codes = pd.Series([0] * len(df_cache))
                        X = df_cache.drop(columns=["target"])

                    # 숫자형 데이터만 남김 (안전장치)
                    X = X.select_dtypes(include=[np.number])

                    print(f" [Cache] 로드 완료! (샘플 수: {len(X):,})")
                    return X, y, codes

            except Exception as e:
                print(f" [Cache Error] 로드 중 오류 발생: {e}. 데이터를 재생성합니다.")

        # -----------------------------------------------------
        # 2. 캐시가 없으면 생성
        # -----------------------------------------------------
        csv_files = []
        if self.csv_path and os.path.exists(self.csv_path):
            csv_files = [self.csv_path]
        elif self.data_dir and os.path.isdir(self.data_dir):
            all_csvs = glob.glob(os.path.join(self.data_dir, "*.csv"))
            csv_files = [
                f
                for f in all_csvs
                if "_1Year" in f and "cached_final_features.csv" not in f
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

        if not all_X:
            return None, None, None

        final_X = pd.concat(all_X, ignore_index=True)
        final_y = pd.concat(all_y, ignore_index=True)
        final_codes = pd.Series(all_codes, name="stck_shrn_iscd")

        print("\n 모델 입력을 위한 데이터 클리닝...")

        # [수정] CSV 저장용 메타데이터 백업 (학습에는 제외, 파일에는 저장)
        meta_data = {}

        # 1) Close 컬럼 처리: 대소문자 구분 없이 찾아 소문자 'close'로 저장
        if "Close" in final_X.columns:
            meta_data["close"] = final_X["Close"]
        elif "close" in final_X.columns:
            meta_data["close"] = final_X["close"]

        # 2) 기타 필수 컬럼 백업
        req_cols = ["timestamp", "open", "high", "low", "volume", "date", "time"]
        for col in req_cols:
            # 대문자로 시작하는 경우 등 다양한 케이스 대응 가능하게 확장 가능
            if col in final_X.columns:
                meta_data[col] = final_X[col]
            elif col.capitalize() in final_X.columns:  # Open, High 등 대응
                meta_data[col] = final_X[col.capitalize()]

        # [수정] 학습용 데이터셋(X) 생성: 식별자 및 메타데이터(Raw Price 등) 제거
        # Close/close가 포함되어 있다면 학습 데이터에서 제거합니다.
        drop_cols = [
            "stck_shrn_iscd",
            "stock_code",
            "code",
            "date",
            "timestamp",
            "Close",
            "close",
        ]

        # 메타데이터에 들어간 컬럼들도 학습용 X에서는 제거 (중복 방지 및 Raw Price 학습 방지)
        drop_cols.extend(list(meta_data.keys()))

        final_X = final_X.drop(
            columns=[c for c in drop_cols if c in final_X.columns], errors="ignore"
        )
        final_X = final_X.select_dtypes(include=[np.number])

        # -----------------------------------------------------
        # 3. 캐시 저장
        # -----------------------------------------------------
        print(
            f" [Cache] 다음 실행 속도 향상을 위해 데이터를 저장합니다... ({cache_file})"
        )
        save_df = final_X.copy()
        save_df["target"] = final_y
        save_df["stck_shrn_iscd"] = final_codes.values

        # 백업해둔 메타데이터 복구 (이때 키 값이 컬럼명이 됨 -> 'close'로 저장됨)
        for col_name, col_data in meta_data.items():
            save_df[col_name] = col_data

        save_df.to_csv(cache_file, index=False)
        print(" [Cache] 저장 완료!")

        print(
            f"\n 통합 완료! 총 샘플 수: {len(final_X):,}, 피처 수: {len(final_X.columns)}"
        )
        print("=" * 60)

        return final_X, final_y, final_codes


def process_and_save_data(data_dir, cache_file):
    print(f"\n[Info] 데이터 처리 시작... (경로: {data_dir})")

    csv_files = glob.glob(os.path.join(data_dir, "*_1Year.csv"))
    if not csv_files:
        print("❌ 처리할 CSV 파일이 없습니다.")
        return

    # 1. 로더 초기화
    expander = (
        FeatureExpander()
        if "FeatureExpander" in globals() and FeatureExpander
        else None
    )
    gcn_extractor = GCNFeatureExtractor()

    all_data_list = []

    for file_path in tqdm(csv_files, desc="파일 처리 중"):
        try:
            filename = os.path.basename(file_path)
            code = filename.split("_")[0]

            # CSV 로드
            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]  # 컬럼 소문자 통일

            # 가격 컬럼명 보정
            if "close" not in df.columns and "price" in df.columns:
                df.rename(columns={"price": "close"}, inplace=True)

            if "close" not in df.columns:
                continue

            # [중요] 원본 데이터 백업 (피처 엔지니어링 후 복구용)
            raw_close = df["close"].copy()

            # Timestamp 생성
            if "timestamp" not in df.columns:
                if "date" in df.columns and "time" in df.columns:
                    df["timestamp"] = pd.to_datetime(
                        df["date"].astype(str) + df["time"].astype(str).str.zfill(6),
                        format="%Y%m%d%H%M%S",
                        errors="coerce",
                    )
                else:
                    # 날짜가 없으면 현재 시간 기준 역산 (임시)
                    df["timestamp"] = pd.date_range(
                        end=datetime.now(), periods=len(df), freq="10min"
                    )

            raw_timestamp = df["timestamp"].copy()

            # 2. 피처 엔지니어링 적용 (기술적 지표 추가)
            if expander:
                # add_technical_indicators 메서드가 있는지 확인 후 호출
                if hasattr(expander, "add_technical_indicators"):
                    df = expander.add_technical_indicators(df)
                elif hasattr(expander, "expand"):
                    df = expander.expand(df)

            # GCN 피처 추가
            gcn_feats = gcn_extractor.get_features(code)
            for k, v in gcn_feats.items():
                df[k] = v

            # 식별자 추가
            df["code"] = code
            df["stck_shrn_iscd"] = code

            # [핵심] 필수 컬럼(가격, 시간) 강제 복구
            df["close"] = raw_close.values
            df["timestamp"] = raw_timestamp.values

            # 타겟 변수 (Target) - 필요 시 사용
            if "target" not in df.columns:
                df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

            # 결측치 제거 (지표 계산으로 인한 앞부분 NaN)
            df.dropna(inplace=True)

            all_data_list.append(df)

        except Exception as e:
            print(f"❌ {file_path} 처리 실패: {e}")

    if not all_data_list:
        print("❌ 병합할 데이터가 없습니다.")
        return

    # 3. 통합 및 저장
    final_df = pd.concat(all_data_list, ignore_index=True)
    final_df.sort_values(["timestamp", "code"], inplace=True)

    print(f"\n📊 통합 데이터 크기: {final_df.shape}")

    # 저장 전 검증
    if "close" not in final_df.columns:
        print("🚨 'close' 컬럼 누락됨! 강제 복구 시도...")
        # (혹시라도 누락되었을 경우를 대비한 비상 로직은 여기서 처리 불가하므로 위 루프에서 해결해야 함)

    final_df.to_csv(cache_file, index=False)
    print(f"✅ 저장 완료! ({cache_file})")
    print(f"🔎 포함된 컬럼(일부): {list(final_df.columns)[:10]} ...")


if __name__ == "__main__":
    # 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))  # 프로젝트 루트

    # 데이터 폴더 (사용자 환경에 맞게 수정)
    # 예: 프로젝트루트/data
    data_dir = os.path.join(project_root, "data")

    # 저장할 파일명
    cache_file = os.path.join(data_dir, "cached_final_features.csv")

    # 실행
    process_and_save_data(data_dir, cache_file)
