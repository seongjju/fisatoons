import time
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from database import webtoon_col  # ✅ MongoDB 연결 import

# ✅ 기존 Chrome 창에 붙어서 실행 (디버그 모드)
options = webdriver.ChromeOptions()
options.debugger_address = "127.0.0.1:9222"  # ✅ 기존 브라우저 사용 (19금 웹툰도 크롤링 가능)

# ✅ 기존 열린 브라우저에 연결
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def get_webtoon_ids_from_db():
    """
    MongoDB에서 웹툰 ID 목록을 가져오는 함수
    """
    webtoon_ids = [webtoon["_id"] for webtoon in webtoon_col.find({}, {"_id": 1})]
    return webtoon_ids


def scrape_webtoon_genres(webtoon_ids):
    """
    네이버 웹툰 ID 리스트를 받아 장르 정보를 크롤링하고, MongoDB에 저장하는 함수.
    """
    for webtoon_id in webtoon_ids:
        print(f"🔹 웹툰 {webtoon_id} 장르 크롤링 시작...")

        # ✅ 웹툰 상세 페이지 접속 (로그인 유지된 Chrome에서 실행)
        driver.get(f"https://comic.naver.com/webtoon/list?titleId={webtoon_id}")
        time.sleep(2)

        try:
            # ✅ 장르 요소 가져오기 (웹툰마다 개수 다름)
            genre_elements = driver.find_elements(By.XPATH, "//*[@id='content']/div[1]/div/div[2]/div/div/a")

            # ✅ 장르 텍스트 추출 (공백 및 '#' 제거)
            genres = [genre.text.strip().replace("#", "") for genre in genre_elements if genre.text.strip()]

            # ✅ MongoDB에 저장 (웹툰별 `genre` 컬럼에 추가)
            webtoon_col.update_one(
                {"_id": webtoon_id},
                {"$set": {"genre": genres}},  # ✅ 장르 리스트 저장
                upsert=True
            )

            print(f"✅ 웹툰 {webtoon_id} 장르 저장 완료: {genres}")

        except NoSuchElementException:
            print(f"❌ 웹툰 {webtoon_id} 장르를 찾을 수 없습니다.")

    print("✅ 모든 웹툰 장르 크롤링 완료!")


# ✅ MongoDB에서 웹툰 ID 가져오기
webtoon_ids = get_webtoon_ids_from_db()

# ✅ 웹툰 장르 크롤링 실행
scrape_webtoon_genres(webtoon_ids)

# ✅ 크롤링 종료 후 기존 Chrome 브라우저는 유지 (창 닫지 않음)
print("✅ 크롤링 완료 (기존 Chrome 유지)")