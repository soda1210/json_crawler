import os
import json
import requests
import glob
import signal
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class AdScraper:
    def __init__(self, json_dir="data/json", output_dir="data/images", class_name="x168nmei x13lgxp2 x5pf9jr xo71vjh xz62fqu xh8yej3 x9ybwvh x19kjcj4"):
        # 全域變數來控制是否停止
        self.stop_processing = False
        
        # 設定目錄
        self.json_dir = json_dir
        self.output_dir = output_dir
        
        # 設定廣告圖片元素的 class name
        self.class_name = class_name
        
        # 確保目錄存在
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # 設定信號處理器
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """處理停止信號"""
        self.stop_processing = True
        print("\n⚠️ 收到停止指令，正在停止處理...")
    
    def _setup_driver(self):
        """設定 Selenium WebDriver"""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # 無頭模式，不開啟瀏覽器視窗
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")  # 防止被偵測
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    
    def _download_image(self, url, folder_path, image_name):
        """下載圖片"""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(os.path.join(folder_path, image_name), 'wb') as file:
                    file.write(response.content)
                return True
        except Exception as e:
            print(f"❌ 下載圖片失敗: {url}，錯誤: {str(e)}")
        return False
    
    def _write_log(self, log_path, message):
        """記錄 log"""
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")
    
    def _process_ad(self, driver, ad, folder_path, log_path):
        """處理單個廣告"""
        if self.stop_processing:
            return False, None

        ad_id = ad['id']
        ad_snapshot_url = ad['ad_snapshot_url']
        
        print(f"📢 處理廣告 {ad_id}，網址：{ad_snapshot_url}")
        self._write_log(log_path, f"📢 開始處理廣告 {ad_id} - {ad_snapshot_url}")

        try:
            driver.get(ad_snapshot_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, self.class_name))
            )

            # 尋找廣告圖片
            image_element = driver.find_element(By.CLASS_NAME, self.class_name)
            img_url = image_element.get_attribute("src")

            if img_url and self._download_image(img_url, folder_path, f"{ad_id}.jpg"):
                print(f"✅ 圖片下載成功: {ad_id}.jpg")
                self._write_log(log_path, f"✅ 成功下載: {ad_id}.jpg")
                return True, None
            else:
                print(f"⚠ 無法找到圖片: {ad_id}")
                self._write_log(log_path, f"⚠ 失敗: {ad_id} - 沒有圖片")
                return False, ad_snapshot_url

        except Exception as e:
            print(f"❌ 發生錯誤（廣告 {ad_id}）: {str(e)}")
            self._write_log(log_path, f"❌ 失敗: {ad_id} - {str(e)}")
            return False, ad_snapshot_url
    
    def process_json_file(self, json_file_path, folder_path):
        """處理 JSON 檔案"""
        driver = self._setup_driver()  # 啟動 Selenium WebDriver
        log_path = os.path.join(folder_path, "log.txt")  # log 檔案

        # 初始化計數
        total_count = 0
        success_count = 0
        fail_count = 0
        fail_details = []

        # 檢查檔案是否存在且可讀
        if not os.path.isfile(json_file_path):
            print(f"❌ 檔案不存在: {json_file_path}")
            return False
            
        # 嘗試讀取 JSON
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    total_count = len(data)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 格式錯誤: {json_file_path}, 錯誤: {str(e)}")
                    return False

                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(self._process_ad, driver, ad, folder_path, log_path) for ad in data]
                    for future in as_completed(futures):
                        if self.stop_processing:
                            break
                        success, fail_detail = future.result()
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                            if fail_detail:
                                fail_details.append(fail_detail)
        except PermissionError:
            print(f"❌ 無法讀取檔案 (權限被拒絕): {json_file_path}")
            print("請確認檔案未被其他程式使用，且您有足夠的權限讀取該檔案。")
            driver.quit()
            return False
        except Exception as e:
            print(f"❌ 讀取檔案時發生錯誤: {json_file_path}, 錯誤: {str(e)}")
            driver.quit()
            return False

        driver.quit()  # 關閉瀏覽器

        # 記錄總結
        self._write_log(log_path, f"\n📊 結果總結:")
        self._write_log(log_path, f"➡ 總數: {total_count}")
        self._write_log(log_path, f"✅ 成功: {success_count}")
        self._write_log(log_path, f"❌ 失敗: {fail_count}")

        if fail_count > 0:
            self._write_log(log_path, f"\n🔻 失敗清單:")
            for fail_url in fail_details:
                if fail_url:
                    self._write_log(log_path, f"- {fail_url}")

        print(f"\n📄 日誌已儲存至: {log_path}")
        return True
    
    def run(self):
        """執行爬蟲程序"""
        # 讀取 json/ 目錄內的所有 JSON 檔案
        try:
            json_files = glob.glob(f"{self.json_dir}/*.json")
        except Exception as e:
            print(f"❌ 讀取 JSON 目錄時發生錯誤: {str(e)}")
            return False

        if not json_files:
            print(f"❌ 找不到任何 JSON 檔案，請確認 {self.json_dir}/ 目錄內有檔案")
            return False

        success_count = 0
        total_count = len(json_files)
        
        for json_file_path in json_files:
            if self.stop_processing:
                break
                
            # 取得 JSON 檔名（不含副檔名）
            json_file_name = os.path.splitext(os.path.basename(json_file_path))[0]

            # 建立對應的資料夾
            folder_path = os.path.join(self.output_dir, json_file_name)
            try:
                os.makedirs(folder_path, exist_ok=True)
            except Exception as e:
                print(f"❌ 無法建立目錄 {folder_path}: {str(e)}")
                continue

            print(f"\n🚀 開始處理 {json_file_name}，圖片將儲存於 {folder_path}")
            if self.process_json_file(json_file_path, folder_path):
                success_count += 1
        
        print(f"\n📊 總結: 成功處理 {success_count}/{total_count} 個 JSON 檔案")
        return success_count > 0 