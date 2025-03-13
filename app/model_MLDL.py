# pandas matplotlib seaborn scikit-learn numpy tabulate joblib tensorflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from tabulate import tabulate
import joblib
from tensorflow.keras.models import load_model
import matplotlib.font_manager as fm

# ================================================================
# 1. 모델 및 데이터 로드
# ================================================================

# CSV 파일 불러오기 (예측 결과 포함 데이터)
df = pd.read_csv("prediction_all.csv")
# df를 기반으로 데이터 복사본 생성 (이후 분석에 사용)
data = df.copy()

# ================================================================
# 2. 성능 평가: 실제 별점 vs 예측 별점 (평가 지표 산출)
# ================================================================
models = ["선형회귀_전체예측", "랜덤포레스트_전체예측", "LSTM_전체예측"]
results = {}
for model in models:
    mse = mean_squared_error(data["실제별점"], data[model])
    rmse = np.sqrt(mse)
    r2 = r2_score(data["실제별점"], data[model])
    results[model] = {"MSE": mse, "RMSE": rmse, "R²": r2}

results_df = pd.DataFrame(results).T
print("\n🔹 실제 별점 vs 예측 별점 성능 비교")
print(tabulate(results_df, headers="keys", tablefmt="pretty", floatfmt=".4f"))

# ================================================================
# 3. 시각화: 전체 데이터 - 실제 별점과 예측 별점 비교 (라인 플롯)
# ================================================================
plt.figure(figsize=(12, 6))
sns.lineplot(x=data["화번호"], y=data["실제별점"], label="실제별점", marker="o", linestyle="-")
for model in models:
    sns.lineplot(x=data["화번호"], y=data[model], label=model, linestyle="--")
plt.xlabel("에피소드 번호")
plt.ylabel("별점")
plt.title("실제 별점 vs 예측 별점 비교")
plt.legend()
plt.grid()
plt.show()

# ================================================================
# 4. 시각화: 특정 웹툰 (에피소드별 예측 평점 vs 실제 평점)
# ================================================================
# 한글 폰트 설정 (NanumGothic)
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 웹툰 ID 목록 출력
print("파일 내 존재하는 웹툰ID 목록:")
print(data["웹툰"].unique())

# 사용자로부터 분석할 웹툰 ID 입력받기
selected_webtoon = int(input("예측 평점 분석 - 선택할 웹툰ID를 입력하세요: "))
webtoon_df = data[data['웹툰'] == selected_webtoon].copy()
# 화번호가 문자형이면 숫자로 변환 후 정렬
if webtoon_df['화번호'].dtype == 'O':
    webtoon_df['화번호'] = pd.to_numeric(webtoon_df['화번호'], errors='coerce')
webtoon_df = webtoon_df.sort_values(by='화번호')

print("\n🔹 필터링된 데이터 미리보기:")
print(tabulate(webtoon_df.head(), headers="keys", tablefmt="pretty", floatfmt=".4f"))

plt.figure(figsize=(12, 6))
plt.plot(webtoon_df['화번호'], webtoon_df['선형회귀_전체예측'], marker='o', label='선형 회귀 예측')
plt.plot(webtoon_df['화번호'], webtoon_df['랜덤포레스트_전체예측'], marker='s', label='랜덤 포레스트 예측')
plt.plot(webtoon_df['화번호'], webtoon_df['LSTM_전체예측'], marker='^', label='LSTM 예측')
plt.plot(webtoon_df['화번호'], webtoon_df['실제별점'], marker='x', linestyle='--', linewidth=2, color='black', label='실제 별점')
plt.xlabel("에피소드 번호")
plt.ylabel("평점")
plt.title(f"{selected_webtoon} 에피소드 별 예측 평점 vs 실제 평점")
plt.legend()
plt.grid(True)
plt.show()

# ================================================================
# 5. 감성 트렌드 시계열 분석 (에피소드별 감성 비율 변화)
# ================================================================
selected_webtoon_sentiment = int(input("감성 트렌드 분석 - 선택할 웹툰ID를 입력하세요: "))
webtoon_sent_df = data[data['웹툰'] == selected_webtoon_sentiment].copy()
if webtoon_sent_df['화번호'].dtype == 'O':
    webtoon_sent_df['화번호'] = pd.to_numeric(webtoon_sent_df['화번호'], errors='coerce')
webtoon_sent_df = webtoon_sent_df.sort_values(by='화번호')

plt.figure(figsize=(10, 6))
plt.plot(webtoon_sent_df['화번호'], webtoon_sent_df['긍정비율'], marker='o', label='긍정비율')
plt.plot(webtoon_sent_df['화번호'], webtoon_sent_df['부정비율'], marker='s', label='부정비율')
plt.xlabel("화 번호")
plt.ylabel("비율")
plt.title(f"{selected_webtoon_sentiment} 감성 트렌드 (에피소드별 감성 비율)")
plt.legend()
plt.grid(True)
plt.show()

# ================================================================
# 6. 평점 분포 및 통계 비교 (박스 플롯 및 바이올린 플롯)
# ================================================================
# 모델 예측 재계산 (필요시; CSV에 이미 예측값이 있다면 생략 가능)
data['선형회귀_전체예측'] = lr_model.predict(data[['긍정비율', '부정비율']])
data['랜덤포레스트_전체예측'] = rf_model.predict(data[['긍정비율', '부정비율']])
data['LSTM_전체예측'] = lstm_model.predict(data[['긍정비율', '부정비율']])

# 선형 회귀 예측 비교
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['실제별점', '선형회귀_전체예측']])
plt.title("전체 데이터 별점 분포 비교 (실제 vs 선형회귀 예측)")
plt.ylabel("평점")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=data[['실제별점', '선형회귀_전체예측']])
plt.title("전체 데이터 별점 분포 비교 (실제 vs 선형회귀 예측)")
plt.ylabel("평점")
plt.show()

# 랜덤 포레스트 예측 비교
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['실제별점', '랜덤포레스트_전체예측']])
plt.title("전체 데이터 별점 분포 비교 (실제 vs 랜덤포레스트 예측)")
plt.ylabel("평점")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=data[['실제별점', '랜덤포레스트_전체예측']])
plt.title("전체 데이터 별점 분포 비교 (실제 vs 랜덤포레스트 예측)")
plt.ylabel("평점")
plt.show()

# LSTM 예측 비교
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['실제별점', 'LSTM_전체예측']])
plt.title("전체 데이터 별점 분포 비교 (실제 vs LSTM 예측)")
plt.ylabel("평점")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=data[['실제별점', 'LSTM_전체예측']])
plt.title("전체 데이터 별점 분포 비교 (실제 vs LSTM 예측)")
plt.ylabel("평점")
plt.show()

# ================================================================
# 7. 오차(Residual) 분석 (선형 회귀 예측 기준)
# ================================================================
selected_webtoon_residual = int(input("오차 분석 - 선택할 웹툰ID를 입력하세요: "))
webtoon_res_df = data[data['웹툰'] == selected_webtoon_residual].copy()
if webtoon_res_df['화번호'].dtype == 'O':
    webtoon_res_df['화번호'] = pd.to_numeric(webtoon_res_df['화번호'], errors='coerce')
webtoon_res_df = webtoon_res_df.sort_values(by='화번호')

# 오차 컬럼 추가: (실제별점 - 선형회귀 예측)
data['오차'] = data['실제별점'] - data['선형회귀_전체예측']

plt.figure(figsize=(10, 6))
plt.hist(data['오차'], bins=30, color='skyblue', edgecolor='black')
plt.title("선형 회귀 예측 오차 분포 (전체 데이터)")
plt.xlabel("오차 (실제별점 - 예측별점)")
plt.ylabel("빈도")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(webtoon_res_df['화번호'],
            webtoon_res_df['실제별점'] - webtoon_res_df['선형회귀_전체예측'],
            color='red')
plt.axhline(0, color='black', linestyle='--')
plt.title(f"{selected_webtoon_residual} 에피소드별 예측 오차 (실제 - 예측)")
plt.xlabel("화 번호")
plt.ylabel("오차")
plt.grid(True)
plt.show()

# ================================================================
# 8. 상관관계 히트맵 (감성 지표와 실제 별점 간)
# ================================================================
corr_cols = ['긍정비율', '부정비율', '실제별점']
corr_matrix = data[corr_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("감성 비율 및 평점 간 상관관계 히트맵")
plt.show()

# ================================================================
# 9. PCA를 활용한 웹툰 클러스터링 및 시각화
# ================================================================
features = ['긍정비율', '부정비율', '실제별점']
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data[features])
data['PCA1'] = pca_result[:, 0]
data['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(10, 6))
if '장르' in data.columns:
    for genre in data['장르'].unique():
        subset = data[data['장르'] == genre]
        plt.scatter(subset['PCA1'], subset['PCA2'], label=genre, alpha=0.6)
    plt.legend()
    plt.title("PCA를 활용한 웹툰 클러스터링 (감성 및 평점 기반)")
else:
    plt.scatter(data['PCA1'], data['PCA2'], alpha=0.6)
    plt.title("PCA를 활용한 웹툰 클러스터링")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()
