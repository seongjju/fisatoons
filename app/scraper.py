import time
from selenium import webdriver
from selenium.common import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from database import webtoon_col, comments_col  # ✅ MongoDB 연결 import
from dotenv import load_dotenv
import os

load_dotenv()
NAVER_ID = os.getenv("NAVER_ID")
NAVER_PW = os.getenv("NAVER_PW")

# ✅ Chrome WebDriver 설정
options = webdriver.ChromeOptions()
# options.add_argument("--headless")  #  창 안 띄우기
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

def naver_login():
    print("🔹 네이버 로그인 시도...")
    driver.get("https://nid.naver.com/nidlogin.login")
    time.sleep(2)  # ✅ 페이지 로딩 대기

    try:
        # ✅ 아이디 입력 필드 찾기
        id_input = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.ID, "id"))
        )
        driver.execute_script("arguments[0].value = arguments[1];", id_input, NAVER_ID)
        time.sleep(1)

        # ✅ 비밀번호 입력 필드 찾기
        pw_input = driver.find_element(By.ID, "pw")
        driver.execute_script("arguments[0].value = arguments[1];", pw_input, NAVER_PW)
        time.sleep(1)

        # ✅ 로그인 버튼 클릭 (JavaScript 사용)
        login_button = driver.find_element(By.ID, "log.login")
        driver.execute_script("arguments[0].click();", login_button)

        time.sleep(3)  # ✅ 로그인 대기

        # ✅ 로그인 성공 확인
        driver.get("https://comic.naver.com/webtoon")  # ✅ 다시 웹툰 페이지로 이동


    except Exception as e:
        print(f"❌ 로그인 중 오류 발생: {e}")
        return False

chrome_options = webdriver.ChromeOptions()
chrome_options.debugger_address = "127.0.0.1:9222"  # 🔹 위에서 실행한 디버그 포트와 맞춰야 함

# ✅ 기존 브라우저에 붙어서 실행
driver = webdriver.Chrome(options=chrome_options)
# ✅ 네이버 웹툰 접속 (로그인 유지됨)
driver.get("https://comic.naver.com/webtoon")

# ✅ WebDriver 실행
print("🚀 ChromeDriver 실행 중...")


# ✅ 1️⃣ 웹툰 목록 크롤링 (중복 방지 + 요일별 10개 제한)
def get_webtoon_list():
    print("🔹 네이버 웹툰 목록 크롤링 시작...")
    nw_url = "https://comic.naver.com/webtoon/weekday"
    driver.get(nw_url)
    time.sleep(2)

    weekdays = [
       ("mon", 1), ("tue", 2),
        ("wed", 3), ("thu", 4), ("fri", 5), ("sat", 6), ("sun", 7)
    ]

    for day_name, day_index in weekdays:
        print(f"🔹 {day_name.upper()} 요일 크롤링 중...")

        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, f"//*[@id='container']/div[3]/div[2]/div[{day_index}]/ul/li"))
            )
        except TimeoutException:
            print(f"❌ {day_name.upper()} 요일 크롤링 실패. 스킵합니다.")
            continue

        titles = driver.find_elements(By.XPATH, f"//*[@id='container']/div[3]/div[2]/div[{day_index}]/ul/li/div/a/span/span")[:10]
        links = driver.find_elements(By.XPATH, f"//*[@id='container']/div[3]/div[2]/div[{day_index}]/ul/li/div/a")[:10]

        for title, link in zip(titles, links):
            webtoon_title = title.text.strip()
            webtoon_id = link.get_attribute("href").split("titleId=")[-1].split("&")[0]

            # ✅ 중복 방지 (업데이트 또는 삽입)
            webtoon_col.update_one(
                {"_id": webtoon_id},
                {"$set": {"title": webtoon_title}, "$addToSet": {"weekday": day_name}},
                upsert=True
            )

    print("✅ 모든 웹툰 저장 완료!")

def get_episode_ratings(webtoon_id):
    print(f"🔹 웹툰 {webtoon_id} 화별 별점 크롤링 시작...")
    episode_data = []

    driver.get(f"https://comic.naver.com/webtoon/list?titleId={webtoon_id}&page=1")
    time.sleep(2)

    try:
        # ✅ 전체 페이지 버튼 가져오기
        page_buttons = driver.find_elements(By.XPATH, "//*[@id='content']/div[3]/div[3]/button")
        page_numbers = []
        print(page_numbers)

        for btn in page_buttons:
            try:
                page_number = int(btn.text.strip())  # ✅ 숫자로 변환 가능한 경우만 리스트에 추가
                page_numbers.append(page_number)
            except ValueError:
                continue  # ✅ "다음 페이지" 같은 문자 버튼은 무시

        max_page = max(page_numbers) if page_numbers else 1  # ✅ 페이지 버튼이 없으면 1페이지뿐
        print(f"🔹 전체 페이지 개수: {max_page}")

    except NoSuchElementException:
        max_page = 1  # ✅ 페이지 버튼이 없으면 1페이지뿐

    for page in range(1, max_page + 1):  # ✅ 첫 페이지부터 마지막 페이지까지 반복
        print(f"🔹 {webtoon_id} - {page}페이지 크롤링 중...")

        driver.get(f"https://comic.naver.com/webtoon/list?titleId={webtoon_id}&page={page}")
        time.sleep(2)


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

            except Exception as e:
                print(f"❌ 에피소드 크롤링 오류: {e}")

    # ✅ MongoDB에 저장 (웹툰 ID별 `episodes` 필드에 추가)
    if episode_data:
        webtoon_col.update_one(
            {"_id": webtoon_id},
            {"$set": {"episodes": episode_data}},
            upsert=True
        )

    return episode_data


# ✅ 3️⃣ 웹툰 회차별 댓글 크롤링 (별점 포함)
def get_comments(webtoon_id, episode, rating):
    print(f"🔹 {webtoon_id} - {episode}화 댓글 크롤링 중...")
    driver.get(f"https://comic.naver.com/webtoon/detail?titleId={webtoon_id}&no={episode}")
    time.sleep(2)

    comments_data = []

    for i in range(1, 6):  # ✅ 최대 5개 댓글 크롤링
        try:
            comment_text = driver.find_element(By.XPATH, f"//*[@id='cbox_module_wai_u_cbox_content_wrap_tabpanel']/ul/li[{i}]/div[1]/div/div[2]/span[2]").text.strip()
            comments_data.append({"text": comment_text})
        except NoSuchElementException:
            break

    # ✅ MongoDB에 별점과 댓글 저장 (업데이트 or 삽입)
    comments_col.update_one(
        {"webtoon_id": webtoon_id, "episode": episode},
        {"$set": {
            "rating": rating,  # ✅ 별점 추가
            "comments": comments_data
        }},
        upsert=True
    )
    print(f"✅ {webtoon_id} - {episode}회차 댓글 및 별점 저장 완료!")


# ✅ 4️⃣ 전체 크롤링 실행
# def run_full_scraper():
#     get_webtoon_list()  # ✅ 웹툰 ID 크롤링
#
#     # webtoon_ids = [webtoon["_id"] for webtoon in webtoon_col.find({}, {"_id": 1})]  # ✅ DB에 있는 웹툰 ID 가져오기
#     webtoon_ids = [webtoon["_id"] for webtoon in webtoon_col.find(
#         {"weekday": {"$in": ["wed", "thu", "fri", "sat", "sun"]}},  # ✅ 수요일부터 일요일까지만 가져오기
#         {"_id": 1}
#     )]
#     for webtoon_id in webtoon_ids:
#         episodes = get_episode_ratings(webtoon_id)
#
#         for episode in episodes:
#             get_comments(webtoon_id, episode["episode"], episode["rating"])  # ✅ 별점과 함께 저장
#
#     print("✅ 모든 크롤링 완료!")
def run_full_scraper():
    get_webtoon_list()  # ✅ 웹툰 ID 크롤링

    # 🔹 특정 웹툰 ID만 수동으로 지정
    webtoon_ids = ["807178"]  # 여기에 수동으로 추가할 웹툰 ID 입력

    for webtoon_id in webtoon_ids:
        episodes = get_episode_ratings(webtoon_id)

        for episode in episodes:
            get_comments(webtoon_id, episode["episode"], episode["rating"])  # ✅ 별점과 함께 저장

    print("✅ 모든 크롤링 완료!")


# ✅ 실행
if __name__ == "__main__":
    # ✅ 크롤링 시작 전에 로그인 실행
    naver_login()

    run_full_scraper()
    driver.quit()