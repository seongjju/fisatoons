# 📌 FISAToon: 웹툰 댓글 감성 분석 및 인기 예측 모델 개발 

### 📆 프로젝트 기간: 2025.03.11 ~ 2025.03.13 (3일)  

---

## 🎯 프로젝트 개요  
FISAToon 프로젝트는 **웹툰 댓글 데이터를 기반으로 감성 분석을 수행하고, 이를 활용하여 웹툰의 인기 추세를 예측하는 것**을 목표로 합니다.  
이를 통해 독자 반응을 정량적으로 분석하고, 감성 변화가 웹툰 인기(별점, 조회수)에 미치는 영향을 예측합니다.  

---

## 👥 팀원 및 역할 분담  

| 이름  | 역할 |
|-------|----------------------------|
| **정성주** | 웹툰 댓글 크롤링 및 전처리, 페이지 제작 |
| **정명희** | 감성 분석 모델 개발 |
| **황지민** | 인기 예측 모델 개발 및 시각화 |

---

## 🛠 개발 환경  
### 🔹 **프레임워크 & 웹 서버**
- FastAPI 
- html, css , javascript

### 🔹 **데이터베이스 & 스토리지**
- MongoDB (댓글 = 비정형 데이터)

- **Google Colab** (GPU 활용)  
- **Google Drive** (모델 및 데이터 저장)  
- **Python 주요 라이브러리:**  
  - Selenium (웹 크롤링)  
  - pandas, numpy (데이터 처리)  
  - Hugging Face Transformers (감성 분석)  
  - matplotlib, seaborn, plotly (데이터 시각화)  
  - scikit-learn, TensorFlow/Keras (머신러닝 & 딥러닝)  

---



## 🗃 **데이터 정보**  

- **크롤링 대상:** 네이버 웹툰 요일별 10개씩 → 총 **70개 웹툰**  
- **수집된 회차:** 각 웹툰별 **약 95회차 (평균)**  
- **댓글 수집 개수:**  
  - 각 회차당 **댓글 5개** 크롤링  
  - 총 댓글 개수 **6,690회차 × 5 = 33,450개**  
- **데이터 구조:**  
  - **웹툰별 메타데이터**
    - `웹툰 ID`, `제목`, `장르`, `작가`, `연재 요일`, `평점`
  - **웹툰 회차별 데이터**
    - `웹툰 ID`, `화번호`, `별점`, `긍정비율`, `부정비율`, `예측 별점 (LSTM, 랜덤포레스트 등)`
  - **웹툰 댓글 데이터**
    - `웹툰 ID`, `화번호`, `댓글 내용`, `감성 분석 결과 (긍정/부정)`, `좋아요 수`, `작성 시간`

---


## 📌 프로젝트 진행 과정  

### 1️⃣ 웹툰 댓글 데이터 크롤링 및 전처리  
- Selenium을 활용하여 웹툰 베스트 댓글 크롤링  
- 텍스트 정제, 불필요한 특수문자 제거  

### 2️⃣ 감성 분석 모델 구축  
- Hugging Face의 **"matthewburke/korean_sentiment"** 모델 활용  
➡️ [Hugging Face 사용 모델 바로가기](https://huggingface.co/matthewburke/korean_sentiment)
- **NSMC 데이터셋으로 Fine-tuning** 후 웹툰 댓글 감성 분석  
- 훈련 전후 성능 비교 → **최적의 모델 선정**  

### 3️⃣ 감성 변화 분석 및 인기 예측  
- 감성 분석 결과를 **웹툰 회차별 별점과 비교**  
- **머신러닝 모델 (Linear Regression, Random Forest) 적용**  
- **딥러닝 모델 (LSTM) 적용**  

### 4️⃣ 웹툰 인기 트렌드 예측 및 시각화  
- 감성 분석 결과와 과거 웹툰 평점을 비교하여 **예측 정확도 검증**  
- 시간별 감성 변화 & 조회수 변화 시각화    

---

## 💡 향후 개선 방향  
- **감성 분석 모델 개선**  
    1. **데이터 차원 개선:**  
       - 영화 댓글 데이터셋이 아닌, 크롤링한 웹툰 댓글 데이터를 활용하여 모델 학습  
       - 데이터의 라벨을 더 다양하게 구성하여 감성 분류 정밀도 향상  
    2. **모델 차원 개선:**  
       - 영화 댓글 데이터를 학습/검증 데이터로 나눠 **Early Stopping**을 적용하여 최적화

---

## 🧑‍💻 추가 개발 계획
**✨ 다음 화 예상 댓글 기능 추가**
- 🤖 ChatGPT를 통해 다음 화의 줄거리를 입력하면 독자의 반응을 예측하여 다음 화 댓글을 모의 생성하는 기능 개발

 **📈 기대 효과**
- 💬 웹툰 콘텐츠의 독자 반응을 사전에 예측하여 효율적인 콘텐츠 관리 및 독자 맞춤형 서비스 제공
- 📊 정교한 감성 분석으로 웹툰 작가와 플랫폼 운영진이 전략적으로 활용할 수 있는 유용한 인사이트 도출

---

## 🚧 트러블슈팅 (문제 해결 과정)  

### 문제 1: 19세 이상 웹툰 크롤링 불가  
**📌 문제:**  
- 네이버에서 **로그인 없이 19금 웹툰 댓글 크롤링 불가**  
- 로그인 자동화 시도 → **네이버가 IP 차단** (로봇 인증 문제 발생)  

**✅ 해결 방법:**  
1. 크롬을 **원격 디버깅 모드로 실행**하여 로그인 유지 후 크롤링  
   ```sh
   open -na "Google Chrome" --args --remote-debugging-port=9222 --user-data-dir="/tmp/chrome_debug"

### 문제 2: 웹툰 사이트의 크롤링 방지 문제 (XPath 변동)  
**📌 문제:**  
- 네이버 웹툰이 **크롤링 방지를 위해 XPath 경로를 웹툰마다 다르게 설정**  
- 하나의 XPath 코드로 모든 웹툰 댓글을 가져오는 것이 불가능  

**✅ 해결 방법:**  
1. **XPath 패턴 분석 후 자동 탐색 로직 적용**  
2. 크롤링 대상 웹툰의 XPath를 매핑하는 **딕셔너리 구조 도입**  
3. XPath가 변경될 경우, 예외 처리를 통해 다른 XPath를 시도  
4. 현재까지 **3가지 다른 XPath 구조를 확인하여 적용 완료**  


### 문제 3: 감성 분석 이전 예측 모델 테스트 문제  
**📌 문제:**  
- 감성 분석이 완료되기 전 **예측 모델을 생성하여 테스트 진행**했으나,  
- 감성 분석 결과가 반영되지 않아 **랜덤 값으로 모델을 평가**하는 문제가 발생  

**✅ 해결 방법:**  
1. 크롤링한 댓글 중 **300개 랜덤 샘플을 수동 라벨링**하여 **테스트 데이터로 활용**  
2. 라벨링된 데이터셋을 감성 분석 모델의 학습 및 평가 과정에 반영  
3. 예측 모델의 결과를 기존 Kaggle 웹툰 데이터셋(과거 평점)과 비교하여 성능 검증  

### 문제 4: 데이터 전처리 및 분할 문제

### ❌ 문제: 무작위 분할로 인해 특정 웹툰의 일부 에피소드만 포함됨  
**✅ 해결 방법:**  
- `train_test_split`은 데이터를 무작위로 분할하므로, 특정 웹툰이 일부만 포함됨
- 무작위 말고 **전체 데이터** 활용


### ❌ 문제: MinMaxScaler 적용 시 `fit_transform()`과 `transform()`의 순서 오류  
**✅ 해결 방법:**  
- 학습 데이터에는 `fit_transform()`을, 테스트 데이터에는 `transform()`을 사용
- 이를 올바르게 적용하지 않으면 **데이터 분포가 달라져 예측 정확도가 떨어질 수 있음**  

---
