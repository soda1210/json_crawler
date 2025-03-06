import os
import json
import requests
import time
import glob
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# 設定 Selenium WebDriver
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 無頭模式，不開啟瀏覽器視窗
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")  # 防止被偵測
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# 下載圖片
def download_image(url, folder_path, image_name):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(os.path.join(folder_path, image_name), 'wb') as file:
                file.write(response.content)
            return True
    except Exception as e:
        print(f"❌ 下載圖片失敗: {url}，錯誤: {str(e)}")
    return False

# 記錄 log
def write_log(log_path, message):
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")

# 處理 JSON 檔案
def process_json_file(json_file_path, folder_path):
    driver = setup_driver()  # 啟動 Selenium WebDriver
    log_path = os.path.join(folder_path, "log.txt")  # log 檔案

    # 初始化計數
    total_count = 0
    success_count = 0
    fail_count = 0
    fail_details = []

    # 讀取 JSON
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        total_count = len(data)

        for ad in data:
            ad_id = ad['id']
            ad_snapshot_url = ad['ad_snapshot_url']
            
            print(f"📢 處理廣告 {ad_id}，網址：{ad_snapshot_url}")
            write_log(log_path, f"📢 開始處理廣告 {ad_id} - {ad_snapshot_url}")

            try:
                driver.get(ad_snapshot_url)
                time.sleep(5)  # 等待頁面加載

                # 尋找廣告圖片
                image_element = driver.find_element(By.CLASS_NAME, "xz62fqu.xh8yej3.x9ybwvh.x19kjcj4")
                img_url = image_element.get_attribute("src")

                if img_url and download_image(img_url, folder_path, f"{ad_id}.jpg"):
                    success_count += 1
                    print(f"✅ 圖片下載成功: {ad_id}.jpg")
                    write_log(log_path, f"✅ 成功下載: {ad_id}.jpg")
                else:
                    fail_count += 1
                    fail_details.append((ad_id, ad_snapshot_url))
                    print(f"⚠ 無法找到圖片: {ad_id}")
                    write_log(log_path, f"⚠ 失敗: {ad_id} - 沒有圖片")

            except Exception as e:
                fail_count += 1
                fail_details.append((ad_id, ad_snapshot_url))
                print(f"❌ 發生錯誤（廣告 {ad_id}）: {str(e)}")
                write_log(log_path, f"❌ 失敗: {ad_id} - {str(e)}")

    driver.quit()  # 關閉瀏覽器

    # 記錄總結
    write_log(log_path, f"\n📊 結果總結:")
    write_log(log_path, f"➡ 總數: {total_count}")
    write_log(log_path, f"✅ 成功: {success_count}")
    write_log(log_path, f"❌ 失敗: {fail_count}")

    if fail_count > 0:
        write_log(log_path, f"\n🔻 失敗清單:")
        for fail_id, fail_url in fail_details:
            write_log(log_path, f"- {fail_id}: {fail_url}")

    print(f"\n📄 日誌已儲存至: {log_path}")

if __name__ == "__main__":
    # 讀取 json/ 目錄內的所有 JSON 檔案
    json_files = glob.glob("json/*.json")

    if not json_files:
        print("❌ 找不到任何 JSON 檔案，請確認 json/ 目錄內有檔案")
        exit(1)

    for json_file_path in json_files:
        # 取得 JSON 檔名（不含副檔名）
        json_file_name = os.path.splitext(os.path.basename(json_file_path))[0]

        # 建立對應的資料夾
        folder_path = os.path.join("images", json_file_name)
        os.makedirs(folder_path, exist_ok=True)

        print(f"\n🚀 開始處理 {json_file_name}，圖片將儲存於 {folder_path}")
        process_json_file(json_file_path, folder_path)
