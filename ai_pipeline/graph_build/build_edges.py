import os
import sys
import pandas as pd
from elasticsearch import Elasticsearch

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# ES 연결
es = Elasticsearch("http://localhost:9200")

def fetch_all_news_from_es(index_name="news_articles", max_size=10000):
    """
    ES에서 뉴스 데이터를 몽땅 가져옵니다.
    """
    query = {
        "query": {"match_all": {}},
        "size": max_size,
        "sort": [{"timestamp": "desc"}]
    }
    
    try:
        resp = es.search(index=index_name, body=query)
        hits = resp['hits']['hits']
        
        data_list = []
        for hit in hits:
            source = hit['_source']
            data_list.append({
                "news_id": hit['_id'],
                "title": source.get("url", "No Title"), # 임시로 URL을 식별자로 씀
                "sentiments": source.get("sentiments", []),
                "related_stocks": source.get("related_stocks", [])
            })
            
        print(f"✅ ES에서 {len(data_list)}개의 뉴스 데이터를 불러왔습니다.")
        return pd.DataFrame(data_list)
        
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return pd.DataFrame()

def build_graph_structure():
    print("🏗️ 그래프 구조 생성 시작...")
    
    # 1. 데이터 가져오기
    df = fetch_all_news_from_es()
    if df.empty:
        print("❌ 데이터가 없어서 종료합니다.")
        return

    # 2. 노드 정의 (Node Definition)
    # 뉴스 노드: 모든 뉴스 기사
    news_nodes = df['news_id'].unique().tolist()
    
    # 종목 노드: 뉴스에 언급된 모든 종목 코드
    stock_nodes = set()
    for stocks in df['related_stocks']:
        if isinstance(stocks, list):
            for s in stocks:
                stock_nodes.add(s)
    stock_nodes = list(stock_nodes)

    print(f"   - 뉴스 노드 개수: {len(news_nodes)}개")
    print(f"   - 종목 노드 개수: {len(stock_nodes)}개")
    print(f"   - 총 노드 개수: {len(news_nodes) + len(stock_nodes)}개")

    # 3. ID 매핑 (String ID -> Integer Index)
    # 컴퓨터는 문자열을 모르므로 0, 1, 2... 숫자로 바꿔줘야 합니다.
    # 0 ~ N-1 : 뉴스 노드
    # N ~ N+M : 종목 노드
    
    node_to_idx = {}
    current_idx = 0
    
    # 뉴스부터 번호표 부여
    for n_id in news_nodes:
        node_to_idx[n_id] = current_idx
        current_idx += 1
        
    # 그 다음 종목 번호표 부여
    for s_code in stock_nodes:
        node_to_idx[s_code] = current_idx
        current_idx += 1

    # 4. 엣지 생성 (Edge Construction)
    # [시작점 리스트, 도착점 리스트] 형태
    src_list = []
    dst_list = []

    for _, row in df.iterrows():
        news_idx = node_to_idx[row['news_id']]
        
        related = row['related_stocks']
        if isinstance(related, list):
            for stock_code in related:
                if stock_code in node_to_idx:
                    stock_idx = node_to_idx[stock_code]
                    
                    # 뉴스 -> 종목 연결
                    src_list.append(news_idx)
                    dst_list.append(stock_idx)
                    
                    # 종목 -> 뉴스 연결 (양방향 그래프인 경우)
                    src_list.append(stock_idx)
                    dst_list.append(news_idx)

    print(f"🔗 생성된 연결(Edge) 개수: {len(src_list)}개")

    # 5. 결과 저장 (CSV로 임시 저장)
    # 나중에 PyTorch Geometric에서 불러오기 쉽게 저장

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "graph_edges.csv")
    json_path = os.path.join(current_dir, "node_mapping.json")

    edges_df = pd.DataFrame({'source': src_list, 'target': dst_list})
    edges_df.to_csv(csv_path, index=False)
    
    # 노드 매핑 정보도 저장 (누가 0번이고 누가 1번인지 알아야 하니까)
    # 간단하게 리스트로 저장
    import json
    with open(json_path, "w", encoding='utf-8') as f:
        # JSON은 set을 저장 못하므로 리스트 등 처리가 필요하지만
        # 여기선 역매핑(Index -> Name)을 저장
        idx_to_node = {v: k for k, v in node_to_idx.items()}
        json.dump(idx_to_node, f, ensure_ascii=False, indent=4)
        
    print("✅ 그래프 데이터 생성 완료! (graph_edges.csv, node_mapping.json)")

if __name__ == "__main__":
    build_graph_structure()