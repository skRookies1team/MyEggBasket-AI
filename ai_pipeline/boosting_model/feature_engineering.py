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

from ai_pipeline.boosting_model.realtime_feature_loader import RealtimeFeatureLoader
from ai_pipeline.boosting_model.feature_expander import FeatureExpander

# GCN 모델 import 시도
try:
    from ai_pipeline.gcn_model.model import NewsStockGCN
except ImportError:
    pass

# =========================================================
# ✅ GCN 로더 클래스 (파일 내장형)
# =========================================================
class GCNFeatureExtractor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 데이터 파일(.pt) 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # boosting_model 폴더 기준 ../../finance_graph_data.pt
        pt_path = os.path.abspath(os.path.join(current_dir, "../../finance_graph_data.pt"))
        
        if not os.path.exists(pt_path):
            print(f"❌ [GCN] 데이터 파일이 없습니다: {pt_path}")
            self.data = None
            return

        try:
            # PyTorch 버전 호환성을 위해 weights_only=False 설정
            self.data = torch.load(pt_path, weights_only=False)
        except:
            self.data = torch.load(pt_path) 

        self.data = self.data.to(self.device)
        
        # 2. 모델 초기화
        self.model = NewsStockGCN(in_channels=3, hidden_channels=16, out_channels=16).to(self.device)
        
        # 3. 모델 가중치 로드
        if model_path is None:
            # boosting_model 기준 ../gcn_model/models/gcn_best_model.pth
            model_path = os.path.abspath(os.path.join(current_dir, "../gcn_model/models/gcn_best_model.pth"))
            
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except:
                self.model.eval() # 로드 실패시 초기화 상태로 사용
        else:
            self.model.eval()

    def _load_json_mapping(self):
        """[내장] JSON 파일 위치 강제 지정"""
        # print("   🔎 [GCN] 매핑 정보(JSON) 찾는 중...")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.abspath(os.path.join(current_dir, "../../ai_pipeline/graph_build/node_mapping.json"))
        
        if not os.path.exists(target_path):
             # 백업 경로 (루트)
             target_path = os.path.abspath(os.path.join(current_dir, "../../node_mapping.json"))

        if os.path.exists(target_path):
            try:
                with open(target_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f) 
                    return {int(v): k for k, v in mapping.items()}
            except Exception as e:
                print(f"   ⚠️ JSON 읽기 에러: {e}")
        
        return None

    def get_embeddings(self):
        if self.data is None: return {}

        with torch.no_grad():
            embeddings = self.model(self.data)
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
            # print(f"   ✅ GCN 매핑 성공 ({len(mapping)}개 종목)")
        else:
            print("   ⚠️ 매핑 정보 없음: GCN 피처를 사용할 수 없습니다.")
            
        return mapping

    def add_gcn_features(self, df, code_col='code'):
        target_col = code_col
        if 'stck_shrn_iscd' in df.columns: target_col = 'stck_shrn_iscd'
        elif 'code' in df.columns: target_col = 'code'
            
        # print(f"🧬 [GCN] 임베딩 병합 시작...")

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
        
        # print(f"✅ GCN 병합 완료 (+{len(gcn_cols)}개 피처)")
        return merged_df


# =========================================================
# ✅ 메인 FeatureEngineer 클래스
# =========================================================
class FeatureEngineer:
    def __init__(self, data_dir=None, csv_path=None):
        # 단일 파일 모드 지원을 위해 csv_path 추가
        self.data_dir = data_dir
        self.csv_path = csv_path 
        self.expander = FeatureExpander()
        try:
            self.gcn_loader = GCNFeatureExtractor()
        except Exception as e:
            print(f"⚠️ GCN 로더 초기화 실패: {e}")
            self.gcn_loader = None

        try:
            self.es = Elasticsearch("http://localhost:9200")
            if not self.es.ping(): self.es = None
        except:
            self.es = None
    
    def _get_date_from_filename(self, filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'(\d{8})', basename)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d")
            except: pass
        return None

    def merge_sentiment_scores(self, X, stock_codes, current_file_path):
        # print("\n📰 뉴스 감성 점수(FinBERT) 병합 중...")
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
            score_map = {b['key']: b['avg_sentiment']['value'] for b in resp['aggregations']['by_stock']['buckets'] if b['avg_sentiment']['value'] is not None}
            # print(f"   📊 뉴스 매칭: {len(score_map)}개 종목")
            sentiment_list = [score_map.get(str(code).zfill(6), 0.0) for code in stock_codes]
            X['sentiment_score'] = sentiment_list
            # print(f"✅ 감성 점수 병합 완료")
        except: pass 
        return X

    def _process_single_file(self, csv_file):
        print(f"   📄 처리 중: {os.path.basename(csv_file)}")
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
        
        X = self.expander.add_technical_indicators(X)
        if self.gcn_loader:
            X = self.gcn_loader.add_gcn_features(X, code_col=temp_code_col)
        X = self.merge_sentiment_scores(X, stock_codes, csv_file)
        X = X.fillna(0)
        
        return X, y, stock_codes
    
    def create_final_features(self):
        print("\n" + "="*60)
        print(f"🏗️ 통합 데이터셋 생성 시작")
        print("="*60)
        
        # csv_path가 있으면 단일 파일 처리, 아니면 data_dir 처리
        csv_files = []
        if self.csv_path and os.path.exists(self.csv_path):
            csv_files = [self.csv_path]
        elif self.data_dir and os.path.isdir(self.data_dir):
            all_csvs = glob.glob(os.path.join(self.data_dir, "*.csv"))
            csv_files = [f for f in all_csvs if re.match(r'^\d{8}\.csv$', os.path.basename(f))]
            csv_files.sort()
        else:
            print("❌ 처리할 CSV 파일이나 데이터 폴더가 지정되지 않았습니다.")
            return None, None, None

        if not csv_files:
            print("❌ CSV 파일 없음")
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
        
        # ====================================================================
        # 🚨 [핵심 수정] 학습용 데이터(X)에서 문자열(종목코드 등) 제거
        # XGBoost는 숫자 데이터만 받을 수 있으므로 Object 타입 컬럼은 삭제해야 함
        # ====================================================================
        print("\n🧹 모델 입력을 위한 데이터 클리닝 (문자열 제거)...")
        
        # 제거할 컬럼 목록 (종목코드, 날짜 등)
        drop_cols = ['stck_shrn_iscd', 'stock_code', 'code', 'date', 'timestamp']
        final_X = final_X.drop(columns=[c for c in drop_cols if c in final_X.columns], errors='ignore')
        
        # 안전장치: 숫자형 컬럼만 남기기
        final_X = final_X.select_dtypes(include=[np.number])
        
        print(f"\n✅ 통합 완료!")
        print(f"   총 파일 수: {len(csv_files)}개")
        print(f"   총 샘플 수: {len(final_X):,}")
        print(f"   최종 피처 수: {len(final_X.columns)}개")
        print("="*60)
        
        return final_X, final_y, final_codes

if __name__ == "__main__":
    # 테스트
    data_dir = r"C:\Users\user\project\MyEggBasket-AI\data"
    if os.path.exists(data_dir):
        engineer = FeatureEngineer(data_dir=data_dir)
        X, y, _ = engineer.create_final_features()
        if X is not None:
            print(X.dtypes) # 모두 float/int인지 확인