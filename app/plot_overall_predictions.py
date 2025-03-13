import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi.responses import HTMLResponse
import plotly.express as px

# 현재 스크립트(`plot_overall_predictions.py`)가 실행되는 디렉토리
script_dir = os.path.dirname(os.path.abspath(__file__))

# `data/` 폴더에 있는 CSV 파일 로드
csv_path = os.path.join(script_dir, "..", "data", "prediction_all.csv")
df = pd.read_csv(csv_path)


def generate_overall_predictions_graph():
    """ 전체 웹툰 데이터를 기반으로 실제 별점 vs 예측 별점 비교하는 Plotly 그래프 반환 """

    # 사용 모델 리스트
    models = ["선형회귀_전체예측", "랜덤포레스트_전체예측", "LSTM_전체예측"]

    # 데이터프레임 생성 (Plotly 사용을 위해 긴 포맷으로 변환)
    df_long = df.melt(id_vars=["화번호"], value_vars=["실제별점"] + models,
                      var_name="모델", value_name="별점")

    # Plotly 그래프 생성
    fig = px.line(
        df_long,
        x="화번호",
        y="별점",
        color="모델",
        title="📊 전체 웹툰: 실제 별점 vs 예측 별점 비교",
        markers=True
    )

    fig.update_layout(
        xaxis_title="에피소드 번호",
        yaxis_title="별점",
        template="plotly_white"
    )

    # HTML로 변환하여 반환
    graph_html = fig.to_html(full_html=True)
    return HTMLResponse(content=graph_html)