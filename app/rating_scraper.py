import pymongo
import pandas as pd
import plotly.express as px
from bson import Decimal128
from fastapi.responses import HTMLResponse

# ✅ MongoDB 연결
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["webtoon_db"]
collection = db["webtoons"]

def generate_genre_rating_graph():
    # ✅ 1. MongoDB에서 데이터 조회
    data = list(collection.find({}, {"_id": 0, "genre": 1, "rating": 1}))

    # ✅ 2. DataFrame 변환 및 NaN 제거
    df = pd.DataFrame(data).dropna()

    # ✅ 3. Decimal128 → float 변환
    df["rating"] = df["rating"].apply(lambda x: float(str(x)) if isinstance(x, Decimal128) else float(x))

    # ✅ 4. 장르별 데이터 확장
    expanded_rows = []
    for _, row in df.iterrows():
        genres = row["genre"] if isinstance(row["genre"], list) else [row["genre"]]
        for genre in genres:
            expanded_rows.append({"genre": genre, "rating": row["rating"]})

    df_expanded = pd.DataFrame(expanded_rows)

    # ✅ 5. 장르별 평균 별점 계산
    genre_rating = df_expanded.groupby("genre")["rating"].mean().reset_index()

    # ✅ 6. Plotly 그래프 생성
    fig = px.bar(
        genre_rating,
        x="rating",
        y="genre",
        orientation="h",
        title="장르별 평균 웹툰 별점",
        labels={"rating": "평균 별점", "genre": "장르"},
        height=600 + len(genre_rating) * 20,  # 데이터 개수에 따라 동적 조절
    )

    # ✅ 7. x축 설정 (9점 이상, 0.1 단위 눈금)
    fig.update_layout(
        yaxis=dict(
            automargin=True,
            showgrid=False,
            categoryorder="total ascending",
        ),
        xaxis=dict(
            showgrid=True,
            range=[9, 10],  # 9점부터 10점까지만 표시
            dtick=0.1,  # 한 눈금 단위 0.1
        ),
        margin=dict(l=150, r=50, t=50, b=50),
    )

    # ✅ 8. HTML로 변환하여 반환
    graph_html = fig.to_html(full_html=True)
    return HTMLResponse(content=graph_html)