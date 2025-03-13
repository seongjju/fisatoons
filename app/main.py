from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pymongo import MongoClient
import json
from app.rating_scraper import generate_genre_rating_graph  # 📌 'app.rating_scraper'로 변경

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
    weekdays = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    weekday_labels = {
        "mon": "월요일", "tue": "화요일", "wed": "수요일",
        "thu": "목요일", "fri": "금요일", "sat": "토요일", "sun": "일요일"
    }

    webtoon_by_weekday = {day: [] for day in weekdays}
    webtoons = list(webtoon_col.find({}, {"_id": 1, "title": 1, "weekday": 1}))

    for webtoon in webtoons:
        for day in webtoon.get("weekday", []):
            if day in webtoon_by_weekday:
                webtoon_by_weekday[day].append(webtoon)

    return templates.TemplateResponse("index.html", {
        "request": request, "webtoon_by_weekday": webtoon_by_weekday, "weekday_labels": weekday_labels
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
    model_1_predictions = [ep.get("model_1", None) for ep in episodes]
    model_2_predictions = [ep.get("model_2", None) for ep in episodes]
    model_3_predictions = [ep.get("model_3", None) for ep in episodes]

    return templates.TemplateResponse("webtoon_detail.html", {
        "request": request,
        "webtoon": webtoon,
        "episode_numbers": json.dumps(episode_numbers),
        "actual_ratings": json.dumps(actual_ratings),
        "model_1_predictions": json.dumps(model_1_predictions),
        "model_2_predictions": json.dumps(model_2_predictions),
        "model_3_predictions": json.dumps(model_3_predictions),
    })

# ✅ 📌 장르별 평균 별점 그래프 라우트 (rating_scraper에서 불러오기)
@app.get("/genre-rating-graph")
async def genre_rating_graph():
    return generate_genre_rating_graph()