import pymongo
import os
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 설정 (settings.py와 동일하게)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "stock_data")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "reports")

def check_data():
    print(f" MongoDB 접속 중... ({MONGO_URI})")
    
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        
        # 1. 전체 개수 확인
        total_count = collection.count_documents({})
        print(f"\n 총 저장된 리포트 개수: {total_count}개")
        print("="*60)

        if total_count == 0:
            print(" 저장된 데이터가 없습니다.")
            return

        # 2. 최신 데이터 5개만 뽑아서 내용 확인
        # _id는 제외하고 출력
        cursor = collection.find({}, {"_id": 0, "content": 0}).sort("uploaded_at", -1).limit(5)
        
        print("[최신 저장된 5개 목록]")
        for idx, doc in enumerate(cursor, 1):
            print(f"{idx}. [{doc.get('stock_code')}] {doc.get('stock_name')} | 파일명: {doc.get('filename')}")
            print(f"   - 저장시간: {doc.get('uploaded_at')}")
            
        print("="*60)
        
        # 3. 첫 번째 데이터의 본문(Content) 앞부분만 살짝 보기
        sample = collection.find_one()
        if sample:
            print("\n [샘플] 첫 번째 리포트 본문 미리보기 (앞 200자):")
            print("-" * 60)
            print(sample.get('content', '')[:200])
            print("..." + "\n" + "-" * 60)

    except Exception as e:
        print(f" 에러 발생: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    check_data()