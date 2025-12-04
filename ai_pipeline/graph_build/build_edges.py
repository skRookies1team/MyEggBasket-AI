import os
import sys
import pandas as pd
import itertools
import re
from elasticsearch import Elasticsearch

# 프로젝트 루트 경로 찾기
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.append(project_root)

# ES 연결
es = Elasticsearch("http://localhost:9200")


# 1. 밸류체인(고정 관계) 로드 함수 (User CSV 맞춤형)

def load_value_chain_edges():
    """
    같은 행(테마/그룹)에 있는 종목들끼리 모두 연결합니다.
    """
    csv_path = os.path.join(project_root, "data", "value_chain_result.csv")
    
    if not os.path.exists(csv_path):
        print(f"⚠️ [Skip] 밸류체인 파일이 없습니다: {csv_path}")
        return []

    try:
        # 인코딩 호환성을 위해 try-except 적용 (value_chain.py 로직 참고)
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(csv_path, encoding='cp949')

        target_col = '기업_코드포함'
        
        if target_col not in df.columns:
            print(f"❌ CSV에 '{target_col}' 컬럼이 없습니다. 컬럼명: {df.columns}")
            return []

        print(f"📂 밸류체인 파일 로드됨: {len(df)}개 그룹(행) 발견")
        
        edges = []
        
        # 각 행을 순회하며 같은 그룹 내 종목끼리 연결
        for idx, row in df.iterrows():
            companies_str = str(row.get(target_col, ''))
            
            # 정규식으로 코드만 추출: "두산밥캣 (241560)" -> "241560"
            # value_chain.py와 동일한 로직 사용
            items = re.findall(r'\(([\d]{6})\)', companies_str)
            
            # 중복 제거 (혹시 같은 코드가 두 번 써있을 경우 대비)
            codes = list(set(items))
            
            # 종목이 2개 이상이어야 연결 가능
            if len(codes) < 2:
                continue
                
            # 해당 그룹 내의 모든 가능한 쌍(Pair) 생성 (예: A, B, C -> AB, AC, BC)
            for src, dst in itertools.combinations(codes, 2):
                edges.append((src, dst))
        
        print(f"🔩 밸류체인 고정 관계 {len(edges)}개 추출 완료!")
        return edges

    except Exception as e:
        print(f"❌ 밸류체인 로드 중 에러 발생: {e}")
        return []





def fetch_filtered_stocks_from_es(index_name="news_articles", max_size=10000):
    """
    ES에서 뉴스를 가져오되, '시황/요약' 기사는 걸러냅니다.
    """
    query = {
        "query": {"match_all": {}},
        "_source": ["related_stocks"], 
        "size": max_size,
        "sort": [{"timestamp": "desc"}]
    }
    
    try:
        resp = es.search(index=index_name, body=query)
        hits = resp['hits']['hits']
        
        valid_data_list = []
        dropped_count = 0
        
        for hit in hits:
            stocks = hit['_source'].get("related_stocks", [])
            
            # 문자열이면 리스트로 변환 (혹시 모를 에러 방지)
            if isinstance(stocks, str):
                stocks = [stocks]
            
            # 리스트가 아니거나 비어있으면 패스
            if not isinstance(stocks, list):
                continue

            stock_cnt = len(stocks)

            # -----------------------------------------------------------
            # [핵심 필터링 로직]
            # 1. 종목이 2개 미만이면 연결 불가 -> 패스
            # 2. 종목이 6개 이상이면 '시황 요약/테마 나열' 기사 -> 패스 (과잉 평준화 원인!)
            # -----------------------------------------------------------
            if len(stocks) < 2: continue
            if len(stocks) >= 6:
                dropped_count += 1
                continue
            
            valid_data_list.append(stocks)
            
        print(f"✅ ES 뉴스 필터링 완료: {len(valid_data_list)}건 (제거된 시황기사: {dropped_count}건)")
        return valid_data_list
        
    except Exception as e:
        print(f"❌ ES 데이터 로딩 실패: {e}")
        return []
        
   
def build_graph_structure():
    print("🏗️ [ES 기반] 엄격한 필터링 그래프 생성 시작...")

    edges = set()

    # (1) 밸류체인 엣지 추가 (강력한 연결)
    vc_edges = load_value_chain_edges() 
    for src, dst in vc_edges:
        # 양방향 연결을 위해 정렬해서 저장 (항상 작은코드 -> 큰코드)
        if src > dst: src, dst = dst, src
        edges.add((src, dst))

    # (2) 뉴스 공기(Co-occurrence) 엣지 추가
    news_stock_lists = fetch_filtered_stocks_from_es()
    for stocks in news_stock_lists:
        for src, dst in itertools.combinations(stocks, 2):
            if src > dst: src, dst = dst, src
            edges.add((src, dst))
            
    print(f"🔗 최종 통합 엣지 개수: {len(edges)}개")

    # (3) 저장
    save_path = os.path.join(project_root, "data", "graph_edges.csv")
    df_edges = pd.DataFrame(list(edges), columns=['source', 'target'])
    df_edges.to_csv(save_path, index=False)
    
    print(f"💾 그래프 저장 완료: {save_path}")
    print("👀 샘플:")
    print(df_edges.head())

if __name__ == "__main__":
    build_graph_structure()

