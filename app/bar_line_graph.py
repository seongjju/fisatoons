import plotly.express as px
import pandas as pd
import os
from fastapi.responses import HTMLResponse
import ast

# CSV 파일 경로
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "webtoon_avg_predictions_with_weekday.csv")

# CSV 데이터 로드
df = pd.read_csv(csv_path)

# 📌 `weekday` 컬럼이 문자열 리스트 형식이라면 변환
df["weekday"] = df["weekday"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

def generate_webtoon_avg_bar_graph():
    """ 요일별 웹툰 평균 별점을 막대그래프로 시각화 """

    models = ["실제별점", "선형회귀_전체예측", "랜덤포레스트_전체예측", "LSTM_전체예측"]

    # 📌 요일 리스트
    weekdays = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    weekday_labels = {
        "mon": "월요일", "tue": "화요일", "wed": "수요일",
        "thu": "목요일", "fri": "금요일", "sat": "토요일", "sun": "일요일"
    }

    # 📌 요일별 그래프 HTML 저장 리스트
    figures = []

    for day in weekdays:
        # 📌 해당 요일에 포함된 웹툰만 필터링
        df_filtered = df[df["weekday"].apply(lambda x: day in x if isinstance(x, list) else False)]

        if df_filtered.empty:
            continue  # 해당 요일에 웹툰이 없으면 스킵

        # 📌 웹툰별 평균 별점 데이터 변환 (웹툰 ID 대신 title 사용)
        df_long = df_filtered.melt(id_vars=["title"], value_vars=models, var_name="모델", value_name="평균 별점")

        # 📌 숫자로 변환 (문자열이면 변환)
        df_long["평균 별점"] = pd.to_numeric(df_long["평균 별점"], errors="coerce")

        # 📌 막대 그래프 생성 (웹툰 ID 대신 title을 x축으로)
        fig = px.bar(
            df_long,
            x="title",
            y="평균 별점",
            color="모델",
            title=f"📊 {weekday_labels[day]} 웹툰 평균 별점 비교",
            barmode="group"
        )

        fig.update_layout(
            xaxis_title="웹툰 제목",
            yaxis_title="평균 별점",
            template="plotly_white",
            xaxis=dict(tickangle=-30),  # x축 레이블 기울기 조정
            yaxis=dict(range=[9, 10]),  # y축 범위 설정
            xaxis_title_standoff=30  # x축 제목과 그래프 사이 간격 조정
        )

        # 📌 HTML 형식으로 그래프 변환 후 저장
        figures.append(fig.to_html(full_html=False))

    # 📌 여러 개의 그래프를 하나의 HTMLResponse로 반환
    combined_html = "<br>".join(figures)
    return HTMLResponse(content=combined_html)