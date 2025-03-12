import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import threading
import signal
from queue import Queue
from sklearn.cluster import KMeans


matplotlib.use('Agg')  # 設定 Matplotlib 使用無 GUI 的後端


# Set input and output folders
IMAGE_FOLDER = "images/Donald_J_Trump_FACEBOOK(1)"  # 圖片來源資料夾
OUTPUT_FOLDER = "output"  # 分析結果輸出資料夾
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set number of threads
NUM_THREADS = 4  # 你可以根據 CPU 調整此數值

# Global stop flag
stop_flag = False

# Handle stop signal (Ctrl+C)
def signal_handler(sig, frame):
    global stop_flag
    print("\n🛑 Stop signal received. Exiting...")
    stop_flag = True

# Register stop signal
signal.signal(signal.SIGINT, signal_handler)

# Check for stop file
def check_stop_file():
    return os.path.exists("stop.txt")

# Extract main colors using K-Means
def extract_colors(image, num_colors=5):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    percentages = np.bincount(labels) / len(labels)

    return colors, percentages

# Count unique colors in the image
def count_unique_colors(image):
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)
    return len(unique_colors)

# Plot RGB histogram and save it
def plot_color_histogram(image, image_name):
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(8, 4))

    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)

    plt.title(f"{image_name} - RGB Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Pixel Count")
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{image_name}_histogram.png"))
    plt.close()

# Compute the average color of the image
def average_color(image):
    avg_color = image.mean(axis=(0, 1))
    return avg_color[::-1]  # Convert to RGB format

# Worker function to process images
def process_images(queue, results):
    global stop_flag
    while not queue.empty() and not stop_flag:
        image_file = queue.get()
        if check_stop_file():
            stop_flag = True
            print("\n🛑 Stop file detected. Exiting...")
            break

        image_path = os.path.join(IMAGE_FOLDER, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Cannot read image: {image_file}")
            continue

        print(f"🔍 Processing {image_file} ...")

        # Extract colors
        colors, percentages = extract_colors(image)

        # Count unique colors
        num_unique_colors = count_unique_colors(image)

        # Save RGB histogram
        plot_color_histogram(image, os.path.splitext(image_file)[0])

        # Compute average color
        avg_color = average_color(image)

        # Store results
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

# Main function
def main():
    global stop_flag
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    if not image_files:
        print("❌ No images found in the folder.")
        return

    queue = Queue()
    for image_file in image_files:
        queue.put(image_file)

    results = []
    threads = []

    # Create and start threads
    for _ in range(NUM_THREADS):
        thread = threading.Thread(target=process_images, args=(queue, results))
        thread.start()
        threads.append(thread)

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_FOLDER, "result.csv"), index=False, encoding="utf-8-sig")

    print("✅ Processing complete! Results saved in 'output' folder.")

# Run the script
if __name__ == "__main__":
    main()
