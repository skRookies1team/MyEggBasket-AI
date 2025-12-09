import pymongo
from datetime import datetime
import sys
import os

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from ai_pipeline.config.settings import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME

class MongoUploader:
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(MONGO_URI)
            self.db = self.client[MONGO_DB_NAME]
            self.collection = self.db[MONGO_COLLECTION_NAME]
            print(f"🔌 MongoDB 연결 성공: {MONGO_DB_NAME}.{MONGO_COLLECTION_NAME}")
        except Exception as e:
            print(f"❌ MongoDB 연결 실패: {e}")
            self.collection = None

    def save_report(self, data):
        """
        리포트 데이터를 DB에 저장합니다. (중복 방지 기능 포함)
        data: { 'stock_code':..., 'stock_name':..., 'filename':..., 'content':... }
        """
        if self.collection is None:
            return

        try:
            # 중복 체크 (파일명이 같은게 이미 있으면 저장 안 함)
            if self.collection.find_one({"filename": data['filename']}):
                print(f"   ⏭️ [DB Skip] 이미 저장된 리포트: {data['filename']}")
                return

            # 저장 날짜 추가
            data['uploaded_at'] = datetime.now()
            
            self.collection.insert_one(data)
            print(f"   💾 [DB 저장] {data['filename']} 저장 완료")
            
        except Exception as e:
            print(f"   ❌ DB 저장 중 에러: {e}")

    def close(self):
        self.client.close()