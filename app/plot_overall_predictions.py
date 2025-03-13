
import plotly.express as px
import pandas as pd
import os
from fastapi.responses import HTMLResponse

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "prediction_all.csv")

df = pd.read_csv(csv_path)

def generate_overall_predictions_graph():
    """ 3D Scatter Plot (X: 웹툰 ID, Y: 화번호, Z: 별점) """

    models = ["실제별점", "선형회귀_전체예측", "랜덤포레스트_전체예측", "LSTM_전체예측"]

    # 📌 긴 형식 변환 (melt)
    df_long = df.melt(id_vars=["웹툰", "화번호"], value_vars=models,
                      var_name="모델", value_name="별점")

    # 📌 3D Scatter Plot 생성
    fig = px.scatter_3d(
        df_long,
        x="웹툰",  # X축: 웹툰 ID (웹툰별 그룹화)
        y="화번호",  # Y축: 에피소드 번호
        z="별점",  # Z축: 별점
        color="모델",  # 모델별 색상 구분
        title="📊 전체 웹툰: 3D 실제 별점 vs 예측 별점 비교",
        opacity=0.8
    )
    fig.update_traces(marker=dict(size=1))  # 원 크기를 3으로 설정 (기본값보다 작음)


    fig.update_layout(
        scene=dict(
            xaxis_title="웹툰 ID",
            yaxis_title="화번호",
            zaxis_title="별점"
        ),
        template="plotly_white"
    )

    return HTMLResponse(content=fig.to_html(full_html=True))

