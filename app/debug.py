import time
from selenium import webdriver
from selenium.common import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from database import webtoon_col, comments_col  # ✅ MongoDB 연결 import

# ✅ 기존 Chrome 세션을 활용하도록 설정
options = webdriver.ChromeOptions()
options.debugger_address = "127.0.0.1:9222"  # 🔹 기존 Chrome 세션에 붙기
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")

# ✅ WebDriver 실행 (기존 세션 사용)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
print("🚀 ChromeDriver 실행 중...")

# ✅ 페이지 로드 테스트
webtoon_id = "807178"
url = f"https://comic.naver.com/webtoon/list?titleId={webtoon_id}&page=2"
driver.get(url)
print(f"🔹 현재 접속된 URL: {driver.current_url}")
print("🔹 현재 페이지 일부 HTML:\n", driver.page_source[:1000])

def get_episode_ratings(webtoon_id):
    """웹툰의 모든 에피소드 및 별점 크롤링 (페이지 버튼 클릭)"""
    print(f"🔹 웹툰 {webtoon_id} 화별 별점 크롤링 시작...")
    episode_data = []

    # ✅ 페이지 버튼 개수 확인
    driver.get(f"https://comic.naver.com/webtoon/list?titleId={webtoon_id}&page=1")
    time.sleep(2)

    # ✅ 전체 페이지 개수 가져오기 (10페이지까지만 크롤링)
    try:
        page_buttons = driver.find_elements(By.XPATH, "//*[@id='content']/div[3]/div[2]/button")
        total_pages = len(page_buttons)
        print(f"🔹 전체 페이지 개수: {total_pages}")

    except NoSuchElementException:
        total_pages = 1

    # ✅ 페이지 URL을 직접 변경하여 접근
    for page in range(1, total_pages + 1):
        print(f"🔹 {webtoon_id} - {page}페이지 크롤링 중...")
        driver.get(f"https://comic.naver.com/webtoon/list?titleId={webtoon_id}&page={page}")
        time.sleep(2)

        # ✅ 현재 페이지의 에피소드 목록 가져오기
        episodes = driver.find_elements(By.XPATH, "//*[@id='content']/div[3]/ul/li")

        if not episodes:
            print(f"❌ {webtoon_id} - {page} 페이지에서 에피소드를 찾지 못했습니다. 크롤링 종료.")
            break

        for episode in episodes:
            try:
                episode_number = episode.find_element(By.XPATH, "./a").get_attribute("href").split("no=")[-1].split("&")[0]
                episode_title = episode.find_element(By.XPATH, "./a/div[2]/p/span").text.strip()
                rating = float(episode.find_element(By.XPATH, "./a/div[2]/div/span[1]/span").text.strip())

                episode_data.append({
                    "episode": episode_number,
                    "title": episode_title,
                    "rating": rating
                })

                print(f"✔ {episode_title} (번호: {episode_number}, 평점: {rating})")

            except Exception as e:
                print(f"❌ 에피소드 크롤링 오류: {e}")

    print("✅ 모든 페이지 크롤링 완료!")

    # ✅ MongoDB에 저장
    if episode_data:
        webtoon_col.update_one(
            {"_id": webtoon_id},
            {"$set": {"episodes": episode_data}},
            upsert=True
        )

    return episode_data

# ✅ 2️⃣ 회차별 댓글 크롤링 (별점 포함)
def get_comments(webtoon_id, episode, rating):
    """웹툰 회차별 댓글 및 별점 저장"""
    print(f"🔹 {webtoon_id} - {episode}화 댓글 크롤링 중...")
    driver.get(f"https://comic.naver.com/webtoon/detail?titleId={webtoon_id}&no={episode}")
    time.sleep(2)

    comments_data = []

    for i in range(1, 6):  # ✅ 최대 5개 댓글 크롤링
        try:
            comment_text = driver.find_element(By.XPATH, f"//*[@id='cbox_module_wai_u_cbox_content_wrap_tabpanel']/ul/li[{i}]/div[1]/div/div[2]/span[2]").text.strip()
            comments_data.append(comment_text)

        except NoSuchElementException:
            break

    # ✅ MongoDB에 저장 (에피소드별 별점과 댓글)
    comments_col.update_one(
        {"webtoon_id": webtoon_id, "episode": episode},
        {"$set": {
            "rating": rating,
            "comments": comments_data
        }},


        upsert=True
    )
    print(f"✅ {webtoon_id} - {episode}화 댓글 및 별점 저장 완료!")


def run_full_scraper():
    webtoon_ids = ["818791"]  # ✅ 크롤링할 웹툰 ID

    for webtoon_id in webtoon_ids:
        episodes = get_episode_ratings(webtoon_id)  # ✅ 에피소드 가져오기

        if not episodes:
            print(f"❌ {webtoon_id} - 에피소드가 없어서 댓글 크롤링을 건너뜁니다.")
            continue

        for episode in episodes:
            get_comments(webtoon_id, episode["episode"], episode["rating"])  # ✅ 댓글 크롤링

    print("✅ 모든 크롤링 완료!")

# ✅ 실행
if __name__ == "__main__":
    run_full_scraper()
    driver.quit()