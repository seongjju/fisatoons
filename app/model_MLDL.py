import pandas as pd
import numpy as np
import os
import json

# 현재 스크립트(`model_MLDL.py`)가 실행되는 디렉토리
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "prediction_all.csv")

# CSV 로드
df = pd.read_csv(csv_path)

def get_webtoon_prediction(webtoon_id: int):
    """특정 웹툰 ID의 예측 데이터를 JSON 형태로 반환"""
    webtoon_df = df[df["웹툰"] == webtoon_id].copy()

    if webtoon_df.empty:
        return None

    webtoon_df["화번호"] = pd.to_numeric(webtoon_df["화번호"], errors="coerce")
    webtoon_df = webtoon_df.sort_values(by="화번호")

    return {
        "episode_numbers": webtoon_df["화번호"].tolist(),
        "actual_ratings": webtoon_df["실제별점"].tolist(),
        "model_1_predictions": webtoon_df["선형회귀_전체예측"].tolist(),
        "model_2_predictions": webtoon_df["랜덤포레스트_전체예측"].tolist(),
        "model_3_predictions": webtoon_df["LSTM_전체예측"].tolist()
    }

def get_sentiment_trend(webtoon_id: int):
    """특정 웹툰 ID의 감성 트렌드 데이터를 JSON 형태로 반환"""
    webtoon_df = df[df["웹툰"] == webtoon_id].copy()

    if webtoon_df.empty:
        return None

    webtoon_df["화번호"] = pd.to_numeric(webtoon_df["화번호"], errors="coerce")
    webtoon_df = webtoon_df.sort_values(by="화번호")

    return {
        "episode_numbers": webtoon_df["화번호"].tolist(),
        "positive_ratio": webtoon_df["긍정비율"].tolist(),
        "negative_ratio": webtoon_df["부정비율"].tolist()
    }

# def get_residual_analysis(webtoon_id: int):
#     """특정 웹툰 ID의 오차 분석 데이터를 JSON 형태로 반환"""
#     webtoon_df = df[df["웹툰"] == webtoon_id].copy()
#
#     if webtoon_df.empty:
#         return None
#
#     webtoon_df["화번호"] = pd.to_numeric(webtoon_df["화번호"], errors="coerce")
#     webtoon_df = webtoon_df.sort_values(by="화번호")
#
#     webtoon_df["오차"] = webtoon_df["실제별점"] - webtoon_df["선형회귀_전체예측"]
#
#     return {
#         "episode_numbers": webtoon_df["화번호"].tolist(),
#         "residuals": webtoon_df["오차"].tolist()
#     }
def get_residual_analysis(webtoon_id: int):
    """특정 웹툰 ID의 오차 분석 데이터를 JSON 형태로 반환"""
    webtoon_df = df[df["웹툰"] == webtoon_id].copy()

    if webtoon_df.empty:
        return None

    webtoon_df["화번호"] = pd.to_numeric(webtoon_df["화번호"], errors="coerce")
    webtoon_df = webtoon_df.sort_values(by="화번호")

    # 오차 계산 (실제별점 - 선형회귀 예측)
    webtoon_df["오차"] = webtoon_df["실제별점"] - webtoon_df["선형회귀_전체예측"]

    # 🔹 특정 웹툰의 오차 히스토그램 생성
    hist_values, hist_bins = np.histogram(webtoon_df["오차"], bins=30)

    return {
        "episode_numbers": webtoon_df["화번호"].tolist(),
        "residuals": webtoon_df["오차"].tolist(),
        "histogram_bins": hist_bins[:-1].tolist(),  # 각 bin의 경계값
        "histogram_values": hist_values.tolist()     # 각 bin에 해당하는 빈도 수
    }