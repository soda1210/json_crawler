# 影像分析研究專案

這個專案包含兩個主要部分：Facebook廣告爬蟲和影像分析工具。專案旨在收集廣告圖片並進行影像分析研究，提供豐富的視覺化報告和數據分析。

## 專案特點

- **高效的爬蟲系統**：使用Selenium自動化收集Facebook廣告圖片
- **強大的影像分析**：提取圖片顏色特徵、分布和統計數據
- **記憶體優化**：支援處理大量高解析度圖片而不會耗盡記憶體
- **批次處理**：可分批處理大量圖片，避免記憶體溢出
- **視覺化報告**：自動生成HTML報告和多種視覺化圖表
- **中斷恢復**：即使在處理中途停止，也會生成部分結果報告

## 專案結構

```
.
├── data/                  # 資料目錄
│   ├── json/              # 存放JSON資料檔案
│   └── images/            # 存放下載的圖片
├── results/               # 分析結果目錄
│   └── analysis/          # 影像分析結果
├── src/                   # 源碼目錄
│   ├── scrapers/          # 爬蟲相關模組
│   │   └── ad_scraper.py  # Facebook廣告爬蟲
│   ├── analysis/          # 分析相關模組
│   │   └── image_analyzer.py  # 影像分析器
│   ├── run_scraper.py     # 執行爬蟲的入口點
│   └── run_analyzer.py    # 執行影像分析的入口點
├── requirements.txt       # 依賴項列表
└── README.md              # 專案說明文件
```

## 安裝

1. 克隆此專案到本地：

```bash
git clone <專案URL>
cd <專案目錄>
```

2. 建立虛擬環境並安裝依賴項：

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## 使用方法

### 爬蟲工具

爬蟲工具用於從Facebook廣告庫中下載廣告圖片。

```bash
python src/run_scraper.py [--json-dir JSON_DIR] [--output-dir OUTPUT_DIR]
```

參數說明：
- `--json-dir`: JSON檔案目錄，預設為 `data/json`
- `--output-dir`: 圖片輸出目錄，預設為 `data/images`

### 影像分析工具

影像分析工具用於分析圖片的顏色分布和特徵。

```bash
python src/run_analyzer.py --image-dir IMAGE_DIR [--output-dir OUTPUT_DIR] [--threads THREADS] [--batch-size BATCH_SIZE] [--max-memory MAX_MEMORY]
```

參數說明：
- `--image-dir`: 圖片來源目錄（必須指定）
- `--output-dir`: 分析結果輸出目錄，預設為 `results/analysis`
- `--threads`: 執行緒數量，預設為 4
- `--batch-size`: 批次處理的圖片數量，預設為 0（處理所有圖片）
- `--max-memory`: 最大記憶體使用量(GB)，預設為 0（不限制）

例如：
```bash
python src/run_analyzer.py --image-dir images/Donald_J_Trump_FACEBOOK(1) --output-dir test_analyzer --threads 8 --batch-size 20 --max-memory 4
```

## 功能說明

### 爬蟲功能

- **自動化收集**：從JSON檔案中讀取廣告資訊，使用Selenium訪問Facebook廣告快照頁面
- **多執行緒下載**：支援多執行緒同時處理多個廣告，提高效率
- **錯誤處理**：完善的錯誤處理和日誌記錄，確保穩定運行
- **優雅中斷**：支援通過Ctrl+C或停止檔案安全地中斷處理

### 影像分析功能

- **顏色分析**：
  - 提取圖片主要顏色（使用K-Means聚類）
  - 計算圖片唯一顏色數量
  - 計算平均顏色
  - 生成RGB直方圖

- **視覺化報告**：
  - 唯一顏色數量分布圖
  - 主要顏色佔比分布圖
  - 平均顏色RGB分量分布
  - 最常見的10種主要顏色
  - 唯一顏色數量vs主要顏色佔比散點圖
  - 平均顏色的RGB 3D散點圖

- **記憶體優化**：
  - 支援批次處理大量圖片
  - 自動縮放大型圖片以節省記憶體
  - 監控記憶體使用量，避免溢出
  - 定期執行垃圾回收

- **中斷處理**：
  - 支援通過Ctrl+C安全地中斷處理
  - 即使在中途停止，也會生成部分結果報告
  - 可以通過建立`stop.txt`檔案來停止處理

## 分析報告

分析完成後，會在輸出目錄生成以下檔案：

- `result.csv`：包含所有圖片分析結果的詳細數據
- `charts/`目錄：包含所有生成的圖表
- `summary_report.html`：包含所有圖表和摘要的HTML報告

HTML報告提供了直觀的視覺化展示，包括：
- 分析摘要（總圖片數量、平均唯一顏色數量、平均主要顏色佔比等）
- 所有生成的圖表，並附有說明

## 注意事項

- 請確保在執行爬蟲前，`data/json/`目錄中已有正確格式的JSON檔案
- 影像分析需要指定有效的圖片目錄
- 處理大量圖片時，建議使用批次處理和記憶體限制
- 對於8GB RAM的系統，建議設置`--max-memory 4`
- 對於16GB RAM的系統，建議設置`--max-memory 8`
- 批次大小可以從10-50開始嘗試，根據實際情況調整
- 如果遇到記憶體問題，可以嘗試減少執行緒數量

## 依賴項

主要依賴項包括：
- selenium：用於網頁自動化
- requests：用於HTTP請求
- opencv-python：用於圖片處理
- numpy：用於數值計算
- matplotlib：用於繪製圖表
- pandas：用於數據處理
- scikit-learn：用於K-Means聚類
- seaborn：用於統計數據視覺化
- psutil：用於監控記憶體使用量

完整依賴項請參見`requirements.txt`。

