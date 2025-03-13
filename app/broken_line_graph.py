
import plotly.express as px
import pandas as pd
import os
from fastapi.responses import HTMLResponse

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "webtoon_avg_predictions.csv")

df = pd.read_csv(csv_path)

def generate_webtoon_avg_line_graph():
    """ 웹툰별 평균 별점을 꺾은선 그래프로 시각화 """

    models = ["실제별점", "선형회귀_전체예측", "랜덤포레스트_전체예측", "LSTM_전체예측"]

    # 웹툰별 평균 별점 데이터 사용 (이미 평균 데이터라 추가 변환 없음)
    df_long = df.melt(id_vars=["title"], value_vars=models, var_name="모델", value_name="평균 별점")

    # 꺾은선 그래프 생성
    fig = px.line(
        df_long,
        x="title",
        y="평균 별점",
        color="모델",
        title="📊 웹툰별 평균 별점 비교",
        markers=True
    )

    fig.update_layout(
        xaxis_title="웹툰 제목",
        yaxis_title="평균 별점",
        template="plotly_white",
        xaxis=dict(tickangle=-45)  # X축 레이블 기울이기
    )

    return HTMLResponse(content=fig.to_html(full_html=True))