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

class FeatureEngineer:
    """
    체결 정보 + GCN 임베딩 + 뉴스 감성 점수(FinBERT) 결합
    """
    
    def __init__(self, data_dir=None):
        self.gcn_embeddings = None
        self.stock_mapping = None
        self.data_dir = data_dir
        
        # ES 클라이언트 연결
        try:
            self.es = Elasticsearch("http://localhost:9200")
            if not self.es.ping():
                self.es = None
        except:
            self.es = None
            print("⚠️ Elasticsearch 연결 실패 (감성 점수 0점 처리)")
    
    def load_gcn_embeddings(self):
        """저장된 GCN 임베딩 로드"""
        current_dir = os.path.dirname(__file__)
        root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
        embedding_path = os.path.join(root_dir, "gcn_node_embeddings.pt")
        
        if not os.path.exists(embedding_path):
            return None
        
        # CPU 환경 호환성 확보
        self.gcn_embeddings = torch.load(embedding_path, map_location=torch.device('cpu'), weights_only=True)
        return self.gcn_embeddings
    
    def load_stock_mapping(self):
        """종목 코드 매핑 로드"""
        mapping_path = os.path.join(os.path.dirname(__file__), "../graph_build/node_mapping.json")
        
        if not os.path.exists(mapping_path):
            return None
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            idx_to_node = json.load(f)
        
        self.stock_mapping = {}
        for idx, node_id in idx_to_node.items():
            if node_id.isdigit():
                self.stock_mapping[node_id] = int(idx)
        return self.stock_mapping

    def _get_date_from_filename(self, filepath):
        """파일 경로에서 날짜 추출 (20251120.csv -> datetime 객체)"""
        basename = os.path.basename(filepath)
        match = re.search(r'(\d{8})', basename)
        if match:
            d_str = match.group(1)
            try:
                return datetime.strptime(d_str, "%Y%m%d")
            except:
                return None
        return None

    def merge_sentiment_scores(self, X, stock_codes, current_file_path):
        """
        [핵심] ES에 저장된 FinBERT 감성 점수를 가져와 병합
        - 성능 최적화: 개별 쿼리 대신 Aggregation 사용
        """
        print("\n📰 뉴스 감성 점수(FinBERT) 병합 중...")
        
        # 기본값 0.0 (중립)으로 초기화
        X['sentiment_score'] = 0.0
        
        if not self.es:
            return X

        # 1. 해당 CSV 파일의 날짜 추출
        target_date = self._get_date_from_filename(current_file_path)
        
        if target_date:
            # 해당 날짜 기준 과거 48시간 뉴스 조회
            end_dt = target_date.replace(hour=23, minute=59, second=59)
            start_dt = end_dt - timedelta(days=2)
            
            range_filter = {
                "range": {
                    "timestamp": {
                        "gte": start_dt.isoformat(),
                        "lte": end_dt.isoformat()
                    }
                }
            }
        else:
            # 날짜 파싱 실패 시 전체 뉴스 (비상시)
            range_filter = {"match_all": {}}

        try:
            # 2. ES Aggregation 쿼리 (종목별 평균 점수 산출)
            # 'related_stocks' 필드에 있는 종목코드로 그룹핑 -> sentiment_score 평균
            body = {
                "size": 0, # 개별 문서는 안 가져옴
                "query": range_filter,
                "aggs": {
                    "by_stock": {
                        "terms": {
                            "field": "related_stocks.keyword",
                            "size": 3000 # 충분히 크게
                        },
                        "aggs": {
                            "avg_sentiment": {
                                "avg": {"field": "sentiment_score"}
                            }
                        }
                    }
                }
            }
            
            resp = self.es.search(index="news_articles", body=body)
            buckets = resp['aggregations']['by_stock']['buckets']
            
            # 3. 결과 매핑 (Dict 변환)
            # 예: {'005930': 0.45, '000660': -0.12}
            score_map = {}
            for b in buckets:
                code = b['key']
                score = b['avg_sentiment']['value']
                if score is not None:
                    score_map[code] = score
            
            print(f"   📊 뉴스 데이터 매칭: {len(score_map)}개 종목 감성 점수 확보")
            
            # 4. DataFrame에 매핑 적용
            # stock_codes 리스트 순서대로 점수 매핑
            sentiment_list = []
            for code in stock_codes:
                # 6자리 포맷 통일 (005930)
                code_str = str(code).zfill(6)
                score = score_map.get(code_str, 0.0) # 뉴스 없으면 0점
                sentiment_list.append(score)
            
            X['sentiment_score'] = sentiment_list
            print(f"✅ 감성 점수 병합 완료 (전체 평균: {np.mean(sentiment_list):.4f})")
            
        except Exception:
            pass # 에러 시 0점 유지
            
        return X

    def merge_gcn_embeddings(self, X, stock_codes):
        """GCN 임베딩 병합 (고속 벡터화)"""
        print("\n🔗 GCN 임베딩 병합 중...")
        if self.gcn_embeddings is None: self.load_gcn_embeddings()
        if self.stock_mapping is None: self.load_stock_mapping()
        
        if self.gcn_embeddings is None: return X
        
        emb_dim = self.gcn_embeddings.shape[1]
        emb_cols = [f'gcn_emb_{i}' for i in range(emb_dim)]
        
        emb_list = []
        zero_emb = np.zeros(emb_dim)
        gcn_emb_numpy = self.gcn_embeddings.numpy()
        
        for code in stock_codes:
            str_code = str(code).zfill(6)
            if str_code in self.stock_mapping:
                node_idx = self.stock_mapping[str_code]
                emb_list.append(gcn_emb_numpy[node_idx])
            else:
                emb_list.append(zero_emb)

        emb_df = pd.DataFrame(emb_list, columns=emb_cols, index=X.index)
        X_final = pd.concat([X, emb_df], axis=1)
        
        print(f"✅ GCN 임베딩 병합 완료")
        return X_final
    

    def _process_single_file(self, csv_file):
        """[내부 함수] 파일 1개를 처리하여 피처셋 반환"""
        print(f"   📄 처리 중: {os.path.basename(csv_file)}")
        
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
        
        if X is None or X.empty: return None, None, None
        
        # GCN 병합
        X = self.merge_gcn_embeddings(X, stock_codes)
        # 감성 점수 병합 (파일 날짜 기준)
        X = self.merge_sentiment_scores(X, stock_codes, csv_file)
        
        return X, y, stock_codes
    
    
    def create_final_features(self):
        """
        폴더 내의 모든 CSV를 읽어서 하나로 합칩니다.
        """
        print("\n" + "="*60)
        print(f"🏗️ 통합 데이터셋 생성 시작 (폴더: {self.data_dir})")
        print("="*60)
        
        if self.data_dir is None: raise ValueError("데이터 폴더 경로가 없습니다.")
        
        # 1. 파일 목록 찾기 (폴더면 *.csv 검색, 파일이면 그 파일만)
        if os.path.isdir(self.data_dir):
            all_csvs = glob.glob(os.path.join(self.data_dir, "*.csv"))
            
            # 파일명 필터링 (숫자 8자리.csv 만 통과)
            csv_files = []
            for f in all_csvs:
                basename = os.path.basename(f)
                # 정규식: 숫자 8개 + .csv 로 끝나는지 확인
                if re.match(r'^\d{8}\.csv$', basename):
                    csv_files.append(f)
                else:
                    print(f"   ⚠️ [Skip] 주가 데이터 아님: {basename}")
            
            csv_files.sort()
            
        elif os.path.isfile(self.data_dir):
            csv_files = [self.data_dir]
        else:
            return None, None, None

        if not csv_files:
            print("❌ 처리할 주가 데이터 CSV가 없습니다.")
            return None, None, None

        # 2. 모든 파일 순회하며 데이터 수집
        all_X = []
        all_y = []
        last_stock_codes = [] # 마지막 파일의 코드를 반환용으로 저장
        
        for f in csv_files:
            X_part, y_part, codes_part = self._process_single_file(f)
            if X_part is not None:
                all_X.append(X_part)
                all_y.append(y_part)
                last_stock_codes = codes_part # 마지막 파일 기준

        if not all_X:
            return None, None, None

        # 3. 데이터 합치기 (Concatenate)
        final_X = pd.concat(all_X, ignore_index=True)
        final_y = pd.concat(all_y, ignore_index=True)
        
        print(f"\n✅ 통합 완료!")
        print(f"   총 파일 수: {len(csv_files)}개")
        print(f"   총 샘플 수: {len(final_X):,}")
        print(f"   타겟 분포(1=상승): {(final_y==1).sum():,}개 ({(final_y==1).sum()/len(final_y)*100:.1f}%)")
        print("="*60)
        
        return final_X, final_y, last_stock_codes

if __name__ == "__main__":
    data_dir = r"C:\Users\user\project\MyEggBasket-AI\data"
    engineer = FeatureEngineer(data_dir=data_dir)
    X, y, _ = engineer.create_final_features()