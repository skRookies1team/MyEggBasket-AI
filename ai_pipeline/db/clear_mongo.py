import sys
import os
import pymongo

# 1. 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

# 설정 파일에서 접속 정보 가져오기
from ai_pipeline.config.settings import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME

def clear_collection():
    print("\n" + "="*60)
    print(f"🔥 MongoDB 데이터 초기화 도구")
    print("="*60)
    print(f"   - 접속 주소: {MONGO_URI}")
    print(f"   - 데이터베이스: {MONGO_DB_NAME}")
    print(f"   - 컬렉션(테이블): {MONGO_COLLECTION_NAME}")

    try:
        # DB 연결
        client = pymongo.MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        
        # 현재 개수 확인
        count_before = collection.count_documents({})
        print(f"   - 현재 저장된 데이터: {count_before}건")
        print("-" * 60)
        
        if count_before == 0:
            print("✅ 이미 비어있습니다. 삭제할 데이터가 없습니다.")
            return

        # 사용자 확인 (안전장치)
        check = input(f"⚠️ 경고: 모든 데이터를 정말로 삭제하시겠습니까? (y/n): ")
        
        if check.lower() == 'y':
            # 전체 삭제 실행
            result = collection.delete_many({})
            print(f"\n🗑️ 삭제 완료! 총 {result.deleted_count}건의 데이터가 삭제되었습니다.")
        else:
            print("\n취소되었습니다. 데이터를 삭제하지 않았습니다.")
        
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    clear_collection()