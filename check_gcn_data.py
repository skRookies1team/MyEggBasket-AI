import torch
import json
import os
import sys

def check_and_fix():
    print("🕵️‍♂️ [GCN 데이터 진단] 파일 분석 시작...")
    
    # 1. 파일 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pt_path = os.path.join(base_dir, "finance_graph_data.pt")
    json_path = os.path.join(base_dir, "node_mapping.json")
    
    # ----------------------------------------------------
    # 1. .pt 파일 뜯어보기
    # ----------------------------------------------------
    if os.path.exists(pt_path):
        try:
            data = torch.load(pt_path, weights_only=False)
            print(f"\n1️⃣ [finance_graph_data.pt] 분석")
            print(f"   - 노드 개수: {data.num_nodes}")
            print(f"   - 가지고 있는 속성들: {data.keys()}")
            
            # 혹시 숨겨진 매핑 정보가 있는지 확인
            if 'stock_to_idx' in data:
                print("   ✅ 대박! 'stock_to_idx'가 파일 안에 숨어있었습니다.")
            elif 'stck_shrn_iscd' in data:
                print("   ✅ 'stck_shrn_iscd' 리스트가 있습니다. 이걸로 매핑 가능합니다.")
            elif 'code' in data:
                print("   ✅ 'code' 리스트가 있습니다. 이걸로 매핑 가능합니다.")
            else:
                print("   ⚠️ 내장된 종목 매핑 정보가 전혀 없습니다.")
                
        except Exception as e:
            print(f"   ❌ .pt 파일 로드 실패: {e}")
    else:
        print(f"   ❌ .pt 파일이 없습니다: {pt_path}")

    # ----------------------------------------------------
    # 2. node_mapping.json 뜯어보기
    # ----------------------------------------------------
    if os.path.exists(json_path):
        print(f"\n2️⃣ [node_mapping.json] 분석")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                
            print(f"   - 데이터 개수: {len(mapping)}")
            # 앞부분 3개만 출력해서 구조 확인
            first_items = list(mapping.items())[:3]
            print(f"   - 데이터 예시: {first_items}")
            
            # 값(Value)이 숫자인지 문자열인지 체크
            sample_val = first_items[0][1]
            if isinstance(sample_val, int) or (isinstance(sample_val, str) and sample_val.isdigit()):
                print("   👉 결론: GCN용 인덱스 매핑 파일이 맞습니다. (코드 오류일 가능성)")
            else:
                print("   👉 결론: 이것은 GCN용이 아닙니다. (Elasticsearch ID 파일 추정)")
                
        except Exception as e:
            print(f"   ❌ JSON 읽기 실패: {e}")
    else:
        print(f"   ❌ JSON 파일이 없습니다.")

if __name__ == "__main__":
    check_and_fix()