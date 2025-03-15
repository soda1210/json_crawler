import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import threading
import signal
import time
from queue import Queue
from sklearn.cluster import KMeans
import seaborn as sns
from collections import Counter

# 設定 Matplotlib 使用無 GUI 的後端
matplotlib.use('Agg')

class ImageAnalyzer:
    def __init__(self, image_folder="data/images/Donald_J_Trump_FACEBOOK(1)", 
                 output_folder="results/analysis", num_threads=4,
                 batch_size=0, max_memory_gb=0):
        """
        初始化影像分析器
        
        Args:
            image_folder: 圖片來源資料夾
            output_folder: 分析結果輸出資料夾
            num_threads: 執行緒數量
            batch_size: 批次處理的圖片數量 (0表示處理所有圖片)
            max_memory_gb: 最大記憶體使用量 (GB) (0表示不限制)
        """
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb
        self.stop_flag = False
        self.threads = []
        
        # 確保輸出目錄存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 設定信號處理器
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """設定信號處理器"""
        # 使用主執行緒處理信號
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """處理停止信號"""
        self.stop_flag = True
        print("\n🛑 Interrupt signal received, stopping processing...")
        print("Please wait, allowing threads to safely finish current tasks...")
        print("Reports will be generated with partial results before exiting.")
        
        # 不再使用 os._exit(0) 強制結束程式
        # 而是讓主程式流程繼續執行到可以生成報告的地方
    
    def _check_stop_file(self):
        """檢查是否存在停止檔案"""
        if os.path.exists("stop.txt"):
            print("\n🛑 Stop file detected, stopping processing...")
            print("Reports will be generated with partial results before exiting.")
            return True
        return self.stop_flag
    
    def _extract_colors(self, image, num_colors=5):
        """
        使用 K-Means 提取主要顏色
        
        Args:
            image: 輸入圖片
            num_colors: 要提取的顏色數量
            
        Returns:
            colors: 顏色列表
            percentages: 各顏色佔比
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=num_colors, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        percentages = np.bincount(labels) / len(labels)

        return colors, percentages
    
    def _count_unique_colors(self, image):
        """計算圖片中的唯一顏色數量"""
        unique_colors = np.unique(image.reshape(-1, 3), axis=0)
        return len(unique_colors)
    
    def _plot_color_histogram(self, image, image_name):
        """繪製並儲存 RGB 直方圖"""
        colors = ('b', 'g', 'r')
        plt.figure(figsize=(8, 4))

        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)

        plt.title(f"{image_name} - RGB Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Pixel Count")
        plt.savefig(os.path.join(self.output_folder, f"{image_name}_histogram.png"))
        plt.close()
    
    def _average_color(self, image):
        """計算圖片的平均顏色"""
        avg_color = image.mean(axis=(0, 1))
        return avg_color[::-1]  # 轉換為 RGB 格式
    
    def _process_images(self, queue, results, thread_id):
        """處理圖片的工作函數（用於多執行緒）"""
        while not queue.empty() and not self.stop_flag:
            try:
                # 非阻塞方式獲取任務，允許檢查停止標誌
                try:
                    image_file = queue.get(block=False)
                except:
                    # 隊列可能為空，但其他執行緒仍在填充
                    if self.stop_flag:
                        break
                    time.sleep(0.1)  # 短暫休眠避免CPU過度使用
                    continue
                
                if self._check_stop_file():
                    self.stop_flag = True
                    print(f"\n🛑 執行緒 {thread_id}: 檢測到停止檔案，正在退出...")
                    break

                image_path = os.path.join(self.image_folder, image_file)
                
                # 使用記憶體優化的方式讀取圖片
                try:
                    # 先獲取圖片尺寸
                    img_info = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_2)  # 縮小一半讀取
                    
                    # 如果圖片太大，則縮小讀取
                    h, w = img_info.shape[:2] if img_info is not None else (0, 0)
                    scale = 1.0
                    
                    # 根據圖片大小決定縮放比例
                    if h * w > 4000000:  # 大於 400萬像素
                        scale = 0.5
                    elif h * w > 1000000:  # 大於 100萬像素
                        scale = 0.75
                    
                    # 根據縮放比例讀取圖片
                    if scale < 1.0:
                        print(f"🔍 執行緒 {thread_id}: 圖片 {image_file} 較大，縮放至 {scale*100:.0f}% 進行分析")
                        image = cv2.resize(img_info, (0, 0), fx=scale, fy=scale)
                    else:
                        image = img_info
                        
                    # 釋放不需要的記憶體
                    img_info = None
                    
                except Exception as e:
                    print(f"❌ 執行緒 {thread_id}: 讀取圖片時發生錯誤: {str(e)}")
                    image = None

                if image is None:
                    print(f"❌ 執行緒 {thread_id}: 無法讀取圖片: {image_file}")
                    queue.task_done()
                    continue

                print(f"🔍 執行緒 {thread_id}: 處理 {image_file} ...")

                # 提取顏色 (使用較少的樣本)
                try:
                    # 隨機取樣以減少計算量
                    h, w = image.shape[:2]
                    if h * w > 500000:  # 如果像素數量大於50萬
                        # 建立縮小的圖片用於顏色分析
                        sample_img = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                    else:
                        sample_img = image
                        
                    colors, percentages = self._extract_colors(sample_img)
                    
                    # 釋放不需要的記憶體
                    if sample_img is not image:
                        sample_img = None
                except Exception as e:
                    print(f"❌ 執行緒 {thread_id}: 提取顏色時發生錯誤: {str(e)}")
                    queue.task_done()
                    continue

                # 計算唯一顏色數量 (使用縮小的圖片)
                try:
                    # 縮小圖片以加速唯一顏色計算
                    small_img = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
                    num_unique_colors = self._count_unique_colors(small_img)
                    small_img = None  # 釋放記憶體
                except Exception as e:
                    print(f"❌ 執行緒 {thread_id}: 計算唯一顏色時發生錯誤: {str(e)}")
                    num_unique_colors = 0

                # 儲存 RGB 直方圖
                try:
                    # 使用縮小的圖片生成直方圖
                    hist_img = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                    self._plot_color_histogram(hist_img, os.path.splitext(image_file)[0])
                    hist_img = None  # 釋放記憶體
                except Exception as e:
                    print(f"❌ 執行緒 {thread_id}: 生成直方圖時發生錯誤: {str(e)}")

                # 計算平均顏色
                try:
                    avg_color = self._average_color(image)
                except Exception as e:
                    print(f"❌ 執行緒 {thread_id}: 計算平均顏色時發生錯誤: {str(e)}")
                    avg_color = [0, 0, 0]

                # 釋放圖片記憶體
                image = None

                # 儲存結果
                results.append({
                    "Image Name": image_file,
                    "Main Color 1": colors[0].tolist(),
                    "Main Color 1 %": round(percentages[0] * 100, 2),
                    "Main Color 2": colors[1].tolist() if len(colors) > 1 else None,
                    "Main Color 2 %": round(percentages[1] * 100, 2) if len(colors) > 1 else None,
                    "Main Color 3": colors[2].tolist() if len(colors) > 2 else None,
                    "Main Color 3 %": round(percentages[2] * 100, 2) if len(colors) > 2 else None,
                    "Unique Color Count": num_unique_colors,
                    "Average Color": avg_color.tolist()
                })
                
                # 強制執行垃圾回收
                if thread_id == 1 and len(results) % 10 == 0:
                    import gc
                    gc.collect()
                
                queue.task_done()
                
                # 定期檢查停止標誌
                if self.stop_flag:
                    break
                    
            except Exception as e:
                print(f"❌ 執行緒 {thread_id}: 處理圖片時發生錯誤: {str(e)}")
                continue

    def _generate_summary_charts(self, results_df):
        """生成全部影像的分析圖表"""
        print("📊 Generating image analysis charts...")
        
        # 建立圖表目錄
        charts_dir = os.path.join(self.output_folder, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # 設定更好的視覺風格
        sns.set(style="whitegrid")
        
        try:
            # 1. 唯一顏色數量分布圖
            if self.stop_flag and len(results_df) < 5:
                print("⚠️ Too few results to generate charts.")
                return
                
            plt.figure(figsize=(10, 6))
            sns.histplot(results_df["Unique Color Count"], kde=True, bins=20)
            plt.title("Unique Colors Distribution")
            plt.xlabel("Number of Unique Colors")
            plt.ylabel("Number of Images")
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "unique_colors_distribution.png"))
            plt.close()
            
            # 2. 主要顏色佔比分布圖
            plt.figure(figsize=(10, 6))
            sns.histplot(results_df["Main Color 1 %"], kde=True, bins=20, color="blue")
            plt.title("Main Color Percentage Distribution")
            plt.xlabel("Main Color Percentage (%)")
            plt.ylabel("Number of Images")
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "main_color_percentage_distribution.png"))
            plt.close()
            
            # 3. 平均顏色 RGB 分量分布
            avg_colors = np.array(results_df["Average Color"].tolist())
            
            # 建立子圖
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # RGB 分量直方圖
            for i, color, title in zip(range(3), ['r', 'g', 'b'], ['Red', 'Green', 'Blue']):
                sns.histplot(avg_colors[:, i], kde=True, color=color, ax=axes[i])
                axes[i].set_title(f"{title} Component Distribution")
                axes[i].set_xlabel(f"{title} Value")
                axes[i].set_ylabel("Number of Images")
            
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "average_color_rgb_distribution.png"))
            plt.close()
            
            # 4. 主要顏色視覺化
            # 將主要顏色轉換為十六進制格式以便視覺化
            def rgb_to_hex(rgb):
                if rgb is None or not isinstance(rgb, list):
                    return "#FFFFFF"  # 預設白色
                return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
            
            # 獲取每張圖片的主要顏色
            main_colors = [rgb_to_hex(color) for color in results_df["Main Color 1"].tolist()]
            
            # 計算顏色出現頻率
            color_counter = Counter(main_colors)
            top_colors = color_counter.most_common(10)
            
            # 繪製主要顏色頻率圖
            plt.figure(figsize=(12, 6))
            colors = [color for color, _ in top_colors]
            counts = [count for _, count in top_colors]
            
            bars = plt.bar(range(len(top_colors)), counts, color=colors)
            
            # 添加顏色代碼標籤
            for i, (color, count) in enumerate(top_colors):
                plt.text(i, count + 0.5, color, ha='center', rotation=90)
            
            plt.title("Top 10 Most Common Main Colors")
            plt.xlabel("Color")
            plt.ylabel("Frequency")
            plt.xticks([])  # 隱藏 x 軸刻度
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "top_main_colors.png"))
            plt.close()
            
            # 5. 散點圖：唯一顏色數量 vs 主要顏色佔比
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x="Unique Color Count", y="Main Color 1 %", data=results_df, alpha=0.7)
            plt.title("Unique Colors vs Main Color Percentage")
            plt.xlabel("Number of Unique Colors")
            plt.ylabel("Main Color Percentage (%)")
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "unique_colors_vs_main_color_percentage.png"))
            plt.close()
            
            # 6. 平均顏色的 RGB 散點圖
            plt.figure(figsize=(10, 8))
            ax = plt.axes(projection='3d')
            ax.scatter(
                avg_colors[:, 0], 
                avg_colors[:, 1], 
                avg_colors[:, 2], 
                c=avg_colors/255.0,  # 正規化顏色值
                s=50,
                alpha=0.7
            )
            ax.set_xlabel('R')
            ax.set_ylabel('G')
            ax.set_zlabel('B')
            ax.set_title('RGB Distribution of Average Colors')
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "average_color_rgb_3d.png"))
            plt.close()
            
            print(f"✅ Charts saved to: {charts_dir}")
            
            # 生成摘要報告
            self._generate_summary_report(results_df, charts_dir)
                
        except Exception as e:
            print(f"❌ Error generating charts: {str(e)}")
    
    def _generate_summary_report(self, results_df, charts_dir):
        """生成摘要報告"""
        report_path = os.path.join(self.output_folder, "summary_report.html")
        
        try:
            # 計算一些統計數據
            total_images = len(results_df)
            avg_unique_colors = results_df["Unique Color Count"].mean()
            avg_main_color_percentage = results_df["Main Color 1 %"].mean()
            
            # 生成 HTML 報告
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Image Analysis Summary Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .chart-container {{ margin-bottom: 30px; }}
                    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <h1>Image Analysis Summary Report</h1>
                
                <div class="summary">
                    <h2>Analysis Summary</h2>
                    <p>Analysis Folder: <strong>{self.image_folder}</strong></p>
                    <p>Total Images: <strong>{total_images}</strong></p>
                    <p>Average Unique Colors: <strong>{avg_unique_colors:.2f}</strong></p>
                    <p>Average Main Color Percentage: <strong>{avg_main_color_percentage:.2f}%</strong></p>
                </div>
                
                <div class="chart-container">
                    <h2>Unique Colors Distribution</h2>
                    <img src="charts/unique_colors_distribution.png" alt="Unique Colors Distribution">
                </div>
                
                <div class="chart-container">
                    <h2>Main Color Percentage Distribution</h2>
                    <img src="charts/main_color_percentage_distribution.png" alt="Main Color Percentage Distribution">
                </div>
                
                <div class="chart-container">
                    <h2>RGB Component Distribution</h2>
                    <img src="charts/average_color_rgb_distribution.png" alt="RGB Component Distribution">
                </div>
                
                <div class="chart-container">
                    <h2>Top 10 Most Common Main Colors</h2>
                    <img src="charts/top_main_colors.png" alt="Top 10 Most Common Main Colors">
                </div>
                
                <div class="chart-container">
                    <h2>Unique Colors vs Main Color Percentage</h2>
                    <img src="charts/unique_colors_vs_main_color_percentage.png" alt="Unique Colors vs Main Color Percentage">
                </div>
                
                <div class="chart-container">
                    <h2>RGB Distribution of Average Colors</h2>
                    <img src="charts/average_color_rgb_3d.png" alt="RGB Distribution of Average Colors">
                </div>
            </body>
            </html>
            """
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            print(f"✅ Summary report saved to: {report_path}")
            
        except Exception as e:
            print(f"❌ Error generating summary report: {str(e)}")
    
    def analyze(self):
        """執行影像分析"""
        try:
            # 重置停止標誌
            self.stop_flag = False
            
            # 獲取圖片檔案列表
            try:
                image_files = [f for f in os.listdir(self.image_folder) 
                              if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            except Exception as e:
                print(f"❌ Error reading image directory: {str(e)}")
                return False

            if not image_files:
                print(f"❌ No images found in folder: {self.image_folder}")
                return False
                
            # 如果設定了批次大小，分批處理
            if self.batch_size > 0 and len(image_files) > self.batch_size:
                print(f"⚠️ Number of images ({len(image_files)}) exceeds batch size ({self.batch_size}), processing in batches")
                
                all_results = []
                batch_count = (len(image_files) + self.batch_size - 1) // self.batch_size
                
                for batch_idx in range(batch_count):
                    if self.stop_flag:
                        break
                        
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(image_files))
                    batch_files = image_files[start_idx:end_idx]
                    
                    print(f"\n--- Processing batch {batch_idx+1}/{batch_count} (images {start_idx+1}-{end_idx}/{len(image_files)}) ---")
                    
                    # 處理當前批次
                    batch_results = self._process_batch(batch_files)
                    
                    if batch_results:
                        all_results.extend(batch_results)
                    
                    # 強制執行垃圾回收
                    import gc
                    gc.collect()
                    
                    # 檢查記憶體使用量
                    if self.max_memory_gb > 0:
                        mem_usage = self._get_memory_usage()
                        print(f"📊 Current memory usage: {mem_usage:.2f} GB")
                        
                        if mem_usage > self.max_memory_gb:
                            print(f"⚠️ Memory usage ({mem_usage:.2f} GB) exceeds limit ({self.max_memory_gb} GB)")
                            print("Pausing processing, waiting for memory to be released...")
                            
                            # 等待記憶體釋放
                            while self._get_memory_usage() > self.max_memory_gb * 0.8 and not self.stop_flag:
                                gc.collect()
                                time.sleep(2)
                
                results = all_results
            else:
                # 一次處理所有圖片
                results = self._process_batch(image_files)
            
            # 即使被中斷，也嘗試生成報告
            if results:
                # 儲存結果到 CSV
                df = pd.DataFrame(results)
                csv_path = os.path.join(self.output_folder, "result.csv")
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                print(f"✅ Analysis results saved to: {csv_path}")
                
                # 生成全部影像的分析圖表
                self._generate_summary_charts(df)
                
                if not self.stop_flag:
                    print(f"✅ Processing complete! Results saved in '{self.output_folder}' folder.")
                else:
                    print(f"⚠️ Processing was interrupted, but partial results have been saved in '{self.output_folder}' folder.")
                
                return not self.stop_flag
            else:
                print("❌ No images were successfully processed")
                return False
            
        except KeyboardInterrupt:
            self.stop_flag = True
            print("\n🛑 Keyboard interrupt received, stopping processing...")
            return False
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            return False
            
    def _process_batch(self, image_files):
        """處理一批圖片"""
        # 建立任務佇列
        queue = Queue()
        for image_file in image_files:
            queue.put(image_file)

        results = []
        self.threads = []

        # 建立並啟動執行緒
        for i in range(self.num_threads):
            thread = threading.Thread(
                target=self._process_images, 
                args=(queue, results, i+1),
                daemon=True  # 設為守護執行緒，主執行緒結束時會自動終止
            )
            thread.start()
            self.threads.append(thread)

        # 等待所有任務完成或收到停止信號
        try:
            # 定期檢查停止標誌，而不是無限期等待
            while any(t.is_alive() for t in self.threads) and not self.stop_flag:
                for t in self.threads:
                    t.join(timeout=0.5)  # 短暫等待，允許檢查停止標誌
                
                # 檢查停止檔案
                if os.path.exists("stop.txt"):
                    self.stop_flag = True
                    print("\n🛑 Stop file detected, stopping processing...")
                    print("Reports will be generated with partial results before exiting.")
                    break
                    
                # 檢查記憶體使用量
                if self.max_memory_gb > 0:
                    mem_usage = self._get_memory_usage()
                    if mem_usage > self.max_memory_gb:
                        print(f"⚠️ Memory usage ({mem_usage:.2f} GB) exceeds limit ({self.max_memory_gb} GB)")
                        self.stop_flag = True
                        print("Stopping processing to free memory...")
                        print("Reports will be generated with partial results before exiting.")
                        break
        except KeyboardInterrupt:
            # 捕獲鍵盤中斷
            self.stop_flag = True
            print("\n🛑 Keyboard interrupt received, stopping processing...")
            print("Reports will be generated with partial results before exiting.")
        
        # 如果收到停止信號，等待執行緒安全結束
        if self.stop_flag:
            print("Waiting for threads to safely finish...")
            for thread in self.threads:
                thread.join(timeout=2.0)
            print("Batch processing stopped.")
        
        return results

    def _get_memory_usage(self):
        """獲取當前程式的記憶體使用量 (GB)"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024 / 1024  # 轉換為 GB 