from pymongo import MongoClient
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")

# MongoDB 연결
client = MongoClient(MONGO_URL)
db = client["webtoon_db"]
webtoon_col = db["webtoons"]
comments_col = db["comments"]