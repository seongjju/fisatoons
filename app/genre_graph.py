import pymongo
import pandas as pd
import plotly.express as px
from bson import Decimal128

# ✅ 1. MongoDB 연결
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["webtoon_db"]
collection = db["webtoons"]

# ✅ 2. 웹툰 데이터 조회
data = list(collection.find({}, {"_id": 0, "genre": 1, "rating": 1}))

# ✅ 3. DataFrame 변환 & NaN 제거
df = pd.DataFrame(data).dropna()

# ✅ 4. Decimal128 → float 변환
df["rating"] = df["rating"].apply(lambda x: float(str(x)) if isinstance(x, Decimal128) else float(x))

# ✅ 5. 장르별 데이터 확장
expanded_rows = []
for _, row in df.iterrows():
    genres = row["genre"] if isinstance(row["genre"], list) else [row["genre"]]
    for genre in genres:
        expanded_rows.append({"genre": genre, "rating": row["rating"]})

df_expanded = pd.DataFrame(expanded_rows)

# ✅ 6. 장르별 평균 별점 계산
genre_rating = df_expanded.groupby("genre")["rating"].mean().reset_index()

# ✅ 7. Plotly 바 그래프 생성 (세로 길이 자동 조절 & 스크롤 가능)
fig = px.bar(
    genre_rating,
    x="rating",
    y="genre",
    orientation="h",
    title="장르별 평균 웹툰 별점",
    labels={"rating": "평균 별점", "genre": "장르"},
    height=600 + len(genre_rating) * 20,  # 🔹 데이터 개수에 따라 동적 조절
)

# ✅ 8. y축 6점 이상부터 시작하고, 눈금 단위를 0.1로 설정
fig.update_layout(
    yaxis=dict(
        automargin=True,
        showgrid=False,
        categoryorder="total ascending",  # 별점 낮은 순 정렬
    ),
    xaxis=dict(
        showgrid=True,
        range=[9, 10],  # 🔥 6점부터 10점까지만 표시
        dtick=0.1,  # 🔥 한 눈금 단위 0.1 설정
    ),
    margin=dict(l=150, r=50, t=50, b=50),  # 좌우 마진 조절
)

# ✅ 9. 그래프 출력 (인터랙티브 지원)
fig.show()