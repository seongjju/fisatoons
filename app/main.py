from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
import json
from fastapi.responses import FileResponse

app = FastAPI()

# 📌 HTML 템플릿 설정
templates = Jinja2Templates(directory="app/templates")

# 📌 정적 파일 (CSS, JS) 서빙
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 📌 MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")
db = client["webtoon_db"]
webtoon_col = db["webtoons"]


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("app/static/favicon.ico")

# ✅ 📌 메인 페이지 - 요일별 웹툰 목록 (테이블 형태)
@app.get("/")
async def home(request: Request):
    # 🔹 요일 목록 (월~일)
    weekdays = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    weekday_labels = {
        "mon": "월요일",
        "tue": "화요일",
        "wed": "수요일",
        "thu": "목요일",
        "fri": "금요일",
        "sat": "토요일",
        "sun": "일요일"
    }

    # 🔹 요일별 웹툰을 저장할 딕셔너리
    webtoon_by_weekday = {day: [] for day in weekdays}

    # 🔹 MongoDB에서 웹툰 가져오기
    webtoons = list(webtoon_col.find({}, {"_id": 1, "title": 1, "weekday": 1}))

    # 🔹 각 웹툰을 요일별로 분류
    for webtoon in webtoons:
        for day in webtoon.get("weekday", []):  # ✅ 웹툰이 여러 요일에 있을 수 있음
            if day in webtoon_by_weekday:
                webtoon_by_weekday[day].append(webtoon)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "webtoon_by_weekday": webtoon_by_weekday,
        "weekday_labels": weekday_labels
    })

# ✅ 📌 웹툰 상세 페이지 - 별점 그래프 표시
@app.get("/webtoon/{webtoon_id}")
async def webtoon_detail(request: Request, webtoon_id: str):
    webtoon = webtoon_col.find_one({"_id": webtoon_id})

    if not webtoon or "episodes" not in webtoon:
        return templates.TemplateResponse("webtoon_detail.html", {"request": request, "webtoon": None})

    # 🔹 JSON 데이터 변환 (화번호 기준 정렬)
    episodes = sorted(webtoon["episodes"], key=lambda x: int(x["episode"]))

    # 🔹 데이터 정리 (각 모델의 예측 값도 포함)
    episode_numbers = [ep["episode"] for ep in episodes]
    actual_ratings = [ep["rating"] for ep in episodes]
    model_1_predictions = [ep.get("model_1", None) for ep in episodes]  # 첫 번째 모델 예측 값
    model_2_predictions = [ep.get("model_2", None) for ep in episodes]  # 두 번째 모델 예측 값
    model_3_predictions = [ep.get("model_3", None) for ep in episodes]  # 세 번째 모델 예측 값

    return templates.TemplateResponse("webtoon_detail.html", {
        "request": request,
        "webtoon": webtoon,
        "episode_numbers": json.dumps(episode_numbers),
        "actual_ratings": json.dumps(actual_ratings),
        "model_1_predictions": json.dumps(model_1_predictions),
        "model_2_predictions": json.dumps(model_2_predictions),
        "model_3_predictions": json.dumps(model_3_predictions),
    })
