from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pymongo import MongoClient
from app.model_MLDL import get_webtoon_prediction, get_sentiment_trend, get_residual_analysis  # 추가된 함수 불러오기

from app.plot_overall_predictions import generate_overall_predictions_graph  # 📌 그래프 함수 불러오기
from app.rating_scraper import generate_genre_rating_graph  # 📌 'app.rating_scraper'로 변경
import os
import pandas as pd

app = FastAPI()

# 📌 HTML 템플릿 설정
templates = Jinja2Templates(directory="app/templates")

# 📌 정적 파일 (CSS, JS) 서빙
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 📌 MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")
db = client["webtoon_db"]
webtoon_col = db["webtoons"]

# CSV 데이터 로드
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "prediction_all.csv")
df = pd.read_csv(csv_path)

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


@app.get("/webtoon/{webtoon_id}")
async def webtoon_detail(request: Request, webtoon_id: int):
    # 📌 MongoDB에서 해당 웹툰의 제목 가져오기
    webtoon_db_entry = webtoon_col.find_one({"_id": str(webtoon_id)}, {"title": 1})
    if not webtoon_db_entry:
        return templates.TemplateResponse("webtoon_detail.html", {
            "request": request,
            "no_data": True,
            "webtoon": {"title": f"웹툰 {webtoon_id}"},  # 기본값 (MongoDB에 없을 경우)
        })

    # 📌 웹툰 데이터 가져오기
    webtoon_data = get_webtoon_prediction(webtoon_id)
    sentiment_data = get_sentiment_trend(webtoon_id)
    residual_data = get_residual_analysis(webtoon_id)

    return templates.TemplateResponse("webtoon_detail.html", {
        "request": request,
        "webtoon": {"title": webtoon_db_entry["title"]},  # 📌 MongoDB에서 가져온 제목 사용
        **webtoon_data,
        "positive_ratio": sentiment_data["positive_ratio"],
        "negative_ratio": sentiment_data["negative_ratio"],
        "residuals": residual_data["residuals"],
        "histogram_bins": residual_data["histogram_bins"],
        "histogram_values": residual_data["histogram_values"]
    })
# ✅ 📌 장르별 평균 별점 그래프 라우트 (rating_scraper에서 불러오기)
@app.get("/genre-rating-graph")
async def genre_rating_graph():
    return generate_genre_rating_graph()



# ✅ 📌 전체 웹툰 예측 비교 그래프 API
@app.get("/overall-predictions-graph", response_class=HTMLResponse)
async def overall_predictions_graph():
    return generate_overall_predictions_graph()