# 📌 Facebook 廣告圖片抓取工具

## 📖 專案介紹
本專案使用 `Selenium` 和 `requests` 自動化從 Facebook 廣告頁面抓取圖片。它會讀取 `json/` 目錄內的 JSON 檔案，
並自動下載廣告圖片到對應的資料夾，記錄成功與失敗的資訊。

---

## 🚀 功能特色
- **📂 自動處理 `json/` 內所有 `.json` 檔案**，不需手動指定
- **📁 每個 JSON 檔案建立專屬資料夾** `images/{json_file_name}/`
- **🖼️ 下載圖片到對應的資料夾**
- **📄 記錄 `log.txt`**（包含成功/失敗數量、失敗網址與 ID）
- **🛡️ 防止中途崩潰**（即時記錄 log）
- **⚡ Selenium 自動模擬瀏覽器，確保抓取的廣告圖片**

---

## 📦 安裝與環境設定
### 1️⃣ 安裝 Python 依賴套件
請先安裝必要的 Python 套件:
```sh
pip install -r requirements.txt
```
（或手動安裝）
```sh
pip install selenium webdriver-manager requests beautifulsoup4
```

### 2️⃣ 安裝 Chrome 瀏覽器與 WebDriver
請確保您的電腦已安裝 [Google Chrome](https://www.google.com/chrome/)，並且 WebDriver 版本與瀏覽器版本相符。

---

## 🔧 使用方法
### 1️⃣ 準備 JSON 檔案
將廣告 JSON 檔案放入 `json/` 目錄，例如：
```
json/
  ├── example.json
  ├── another_ad.json
```

JSON 格式範例：
```json
[
  {
    "id": "12345",
    "ad_snapshot_url": "https://www.facebook.com/ads/xyz123"
  },
  {
    "id": "67890",
    "ad_snapshot_url": "https://www.facebook.com/ads/abc456"
  }
]
```

### 2️⃣ 執行腳本
```sh
python main.py
```
程式會自動讀取 `json/` 內的所有 JSON 檔案，下載對應的廣告圖片，並儲存到 `images/{json_file_name}/`。

### 3️⃣ 檢查下載結果
每個 JSON 檔案會建立對應的資料夾，下載的圖片與 `log.txt` 會存放於：
```
images/
  ├── example/
  │   ├── 12345.jpg
  │   ├── 67890.jpg
  │   ├── log.txt
  ├── another_ad/
      ├── abc.jpg
      ├── def.jpg
      ├── log.txt
```

---

## 📜 Log 紀錄格式
- `✅ 成功: 12345.jpg` → 下載成功
- `❌ 失敗: 67890 - 找不到圖片` → 下載失敗
- `📊 結果總結: 總數: 10 | 成功: 8 | 失敗: 2`

---

## ❗ 注意事項
1. **請確保 JSON 檔案內的 `ad_snapshot_url` 有效**
2. **Facebook 可能會變更 HTML 結構**，導致 `Selenium` 無法找到圖片，請適時調整 `find_element` 條件
3. **若 Selenium 下載圖片失敗**，可能是 `headless` 模式受限制，可嘗試移除 `--headless` 參數

---

## 🛠️ 技術細節
- `Selenium WebDriver` 用於模擬瀏覽器開啟 Facebook 廣告頁面
- `requests` 用於下載圖片
- `glob` 自動掃描 `json/` 內的所有 JSON 檔案
- `log.txt` 即時紀錄下載狀況，防止中途崩潰導致資料遺失

---

## 🏆 授權 & 版權
本專案僅供學術與研究用途，請勿用於違反 Facebook 條款的行為。

